import os
import sqlite3
import json
import pandas as pd
import asyncio
import time

from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from typing import TypedDict
from sandbox import run_sandbox
from mcp_helpers import fetch_dataframe
from langchain.mcp import MCPAdapter
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.messages import ToolMessage
from log_system import init_log, log_event
from oversight import add_oversight, CODEBOOK
from report_tables import settings_summary, stats_table, model_table, model_facts, render_report, direction_summary

load_dotenv()

if os.getenv("LLM_PROVIDER") == "claude":
    llm_key = os.getenv("ANTHROPIC_API_KEY")
    base_url = "https://api.anthropic.com/v1/"
    model_name = os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")
    extra_headers = {}
else:
    llm_key = "dummy"
    base_url = os.getenv("OSBOT_BASE_URL")
    model_name = os.getenv("LOCAL_MODEL")
    extra_headers = {"x-api-key": os.getenv("OSBOT_API_KEY")}

def make_model():
    #we create a new model every run, because the async client gets tied to the event loop
    #asyncio.run() closes the loop after each run, so a reused model would crash on the next /run
    return ChatOpenAI(
        model=model_name,
        api_key=llm_key,
        base_url=base_url,
        max_tokens=5000,
        default_headers=extra_headers,
    )

client = OpenAI(
    api_key=llm_key,
    base_url=base_url,
    default_headers=extra_headers,
)

def trace(agent, event, **details):
    #one place for all step logging:
    #prints a short readable line in the docker logs, and writes the full details to the jsonl run log
    #everything from parts to print is just for docker log
    parts = []
    for key, value in details.items():
        text = str(value)
        if len(text) > 400:
            text = text[:400] + "..."
        parts.append(f"{key}={text}")
    print(f"[{agent}] {event}: " + ", ".join(parts), flush=True)
    log_event(agent, event, **details)

def log_usage(agent, response):
    #token usage per LLM call, so we can sum it per run and compare models
    usage = getattr(response, "usage", None)
    if usage:
        trace(agent, "token_usage",
              prompt=usage.prompt_tokens,
              completion=usage.completion_tokens,
              total=usage.total_tokens)

def log_langchain_usage(agent, response):
    #LangChain responses store token usage in usage_metadata instead of usage, this is specifically for data agent
    usage = getattr(response, "usage_metadata", None) or {}
    if usage:
        trace(agent, "token_usage",
              prompt=usage.get("input_tokens"),
              completion=usage.get("output_tokens"),
              total=usage.get("total_tokens"))

def clean_number(text):
    text = text.strip()
    #numpy 2 prints numbers like np.float64(937.05), so keep only what is inside the brackets
    if "(" in text and text.endswith(")"):
        start = text.index("(") + 1
        text = text[start:-1]
    return float(text)

def call_reviewer(agent_name, prompt, max_tokens):
    #ask the LLM for a review. if it runs out of tokens or returns nothing, that is the
    #reviewer's own problem, not a problem with the work it checks. so instead of rejecting
    #the work, we ask the reviewer once more with twice as much room
    response_text = None
    for attempt in range(2):
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
        )
        log_usage(agent_name, response)

        response_text = response.choices[0].message.content
        finish_reason = response.choices[0].finish_reason

        if response_text and response_text.strip() and finish_reason != "length":
            return response_text

        trace(agent_name, "reviewer_ran_out", attempt=attempt + 1, finish_reason=finish_reason, max_tokens=max_tokens)
        max_tokens = max_tokens * 2

    #still nothing after two tries: return what we have, the caller treats it as not approved
    return response_text

def normalize_for_matching(text):
    #LLMs often change special characters when they copy text (narrow spaces, unicode hyphens,
    #curly quotes), so compare a simplified version of both texts instead of the raw strings
    text = str(text).lower()
    for special in [" ", " ", " "]:
        text = text.replace(special, " ")
    for dash in ["‐", "‑", "‒", "–", "—", "−"]:
        text = text.replace(dash, "-")
    for quote_char in ["“", "”", "‘", "’", "\"", "'", "*", "`"]:
        text = text.replace(quote_char, "")
    return " ".join(text.split())

def quote_is_in_text(quote, text):
    #true if every part of the quote appears in the text. a reviewer may shorten a long
    #sentence with "..." so each part between the dots is checked on its own
    parts = quote.replace("…", "...").split("...")
    normalized_text = normalize_for_matching(text)
    found_any = False
    for part in parts:
        part = normalize_for_matching(part).strip(" .,;:")
        if not part:
            continue
        if part not in normalized_text:
            return False
        found_any = True
    return found_any

def get_review(agent_name, prompt, max_tokens, reviewed_text=None):
    #ask a reviewer and turn its answer into {"verdict": bool, "reason": str, "quote": str}.
    #two things are checked by code, and in both cases we ask the reviewer once more,
    #because like an empty answer it is the reviewer's own problem, not the work's:
    #  1) a verdict without a reason: the reviewer never said what it checked
    #  2) a rejection that quotes text which is not in the reviewed text: the reviewer
    #     invented the quote, so the rejection is based on something that was never written
    problems = []   #what went wrong in each attempt, so the final message says what really happened
    review = None

    for attempt in range(2):
        response_text = call_reviewer(agent_name, prompt, max_tokens)

        if not response_text:
            #call_reviewer already retried, so don't ask a third time
            return {"verdict": False, "reason": f"{agent_name} returned no content to parse.", "quote": ""}

        try:
            review = json.loads(response_text)
        except Exception as e:
            return {"verdict": False, "reason": f"Could not parse {agent_name} response as JSON: {e}", "quote": ""}

        verdict = bool(review.get("verdict", False))
        reason = str(review.get("reason", "")).strip()
        quote = str(review.get("quote", "") or "").strip()
        review = {"verdict": verdict, "reason": reason, "quote": quote}

        if not reason:
            problems.append(f"attempt {attempt + 1}: gave a verdict ({verdict}) without a reason")
            trace(agent_name, "empty_reason", attempt=attempt + 1, verdict=verdict)
            continue

        #only rejections need a quote. an empty quote is allowed, it means something is missing
        if not verdict and quote and reviewed_text is not None:
            if not quote_is_in_text(quote, reviewed_text):
                problems.append(f"attempt {attempt + 1}: rejected with a quote that is not in the text (\"{quote}\")")
                trace(agent_name, "quote_not_found", attempt=attempt + 1, quote=quote, reason=reason)
                continue
            trace(agent_name, "quote_verified", attempt=attempt + 1, quote=quote)
        elif not verdict and not quote:
            trace(agent_name, "no_quote", attempt=attempt + 1, reason=reason)

        return review

    #both attempts were unusable. we still reject (we can't approve work that was never really
    #checked), but the message says exactly what went wrong in each attempt
    summary = "; ".join(problems)
    trace(agent_name, "review_unusable", problems=problems)

    if review is not None and review["reason"] and review["quote"]:
        #the last answer had a real reason but a quote that doesn't exist: pass the criticism on,
        #with a warning so the writer checks it against its own text instead of rewriting blindly
        reason = (f"{review['reason']} (Note: the quoted text \"{review['quote']}\" was not found in what you wrote, "
                  f"so this criticism may not apply. Check it against your own text.)")
        return {"verdict": False, "reason": reason, "quote": review["quote"]}

    return {"verdict": False, "reason": f"{agent_name} did not give a usable verdict, so the work is not approved ({summary}).", "quote": ""}

def shorten(text, max_chars=1000):
    #long error messages (for example when the agent's code printed a whole table)
    #must never go into a prompt in full, or the prompt becomes too big for the model
    text = str(text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"... [cut, {len(text) - max_chars} more characters]"

#this is stategraph, it's a data container (or shared memory) that gets passed around the agents.
class ResearchState(TypedDict):
    chosen_columns: list[str]
    method_choice: dict
    calculated_data: dict
    interpretation: str
    reviewer_feedback: str
    reviewer_verdict: bool
    revision_counter: int
    modelling_counter: int
    decision_log: list[dict]
    scientific_report: str
    report_feedback: str          
    report_verdict: bool          
    report_revision_counter: int  
    

def data_agent(state: ResearchState) -> dict:
    #conversation history, we will add to this as tools are called to LLM gets a history of messages
    messages = [HumanMessage(content="Find out which columns exist in the table, then fetch the data for what you percieve to be important factors that could cause diabetes, and decide when you are done.")]

    async def explore_and_select():
        model = make_model()
        async with MCPAdapter(Path("diabetes-db-server.py")) as adapter:
            tools = await adapter.list_tools()
            model_with_tools = model.bind_tools(tools)

            response = await model_with_tools.ainvoke(messages)
            log_langchain_usage("data_agent", response)
            print(f"[data_agent] first response, tool_calls = {response.tool_calls}")

            final_columns = None
            #had issue of being stuck while running program, seeing if this fixes
            loop_count = 0
            max_loops = 10
    
            while response.tool_calls:
                loop_count = loop_count + 1
                messages.append(response)

                for tool_call in response.tool_calls:
                    requested_tool_name = tool_call["name"].split("<|")[0].strip()
                    matching_tool = next((t for t in tools if t.name == requested_tool_name), None)

                    print(f"[data_agent] loop {loop_count}: model called '{requested_tool_name}' with args {tool_call['args']}")
                    trace("data_agent", "tool_call", loop=loop_count, tool=requested_tool_name, args=tool_call["args"])
                    
                    if matching_tool is None:   #more debugging
                        print(f"[data_agent] WARNING: no tool named '{requested_tool_name}' exists, skipping")  
                        continue   

                    raw_result = await matching_tool.ainvoke(tool_call["args"])
                    log_event("data_agent", "tool_result", tool=requested_tool_name, result=raw_result)
                    
                    if isinstance(raw_result, list):
                        result_text = "".join(block.get("text", "") for block in raw_result if isinstance(block, dict))
                    else:
                        result_text = raw_result

                    tool_message = ToolMessage(content=result_text, tool_call_id=tool_call["id"])
                    messages.append(tool_message)

                    if requested_tool_name == "finalized_selection":
                        final_columns = tool_call["args"]["chosen_columns"]

                if final_columns is not None:
                    break
                if loop_count >= max_loops:
                    break

                response = await model_with_tools.ainvoke(messages)
                log_langchain_usage("data_agent", response)

            if final_columns is None:
                #sometimes agent decides to stop before calling the final tool, so this forces it to do it
                print("[data_agent] model stopped without calling finalized_selection, nudging it to finish")
                messages.append(response)
                messages.append(HumanMessage(content="You explored the data but did not call finalized_selection. Please call finalized_selection now with the columns you have decided to use."))
                response = await model_with_tools.ainvoke(messages)
                log_langchain_usage("data_agent", response)
                if response.tool_calls:
                    for tool_call in response.tool_calls:
                        requested_tool_name = tool_call["name"].split("<|")[0].strip()
                        if requested_tool_name == "finalized_selection":
                            final_columns = tool_call["args"]["chosen_columns"]

            print(f"[data_agent] loop ended, final_columns = {final_columns}")
            return final_columns

    final_columns = asyncio.run(explore_and_select())
    if final_columns is None: #safeguard incase we get nothing 
        final_columns = []

    trace("data_agent", "final_columns", columns=final_columns)

    new_log_entry = {"agent": "data_agent", "action": f"agent gets the data it wants from diabetes.db, it chose: {final_columns}"}
    updated_log = state["decision_log"] + [new_log_entry] 
    
    #we return the stuff we want to update in our state
    return {"chosen_columns": final_columns, "decision_log": updated_log}

def methods_agent(state: ResearchState) -> dict:
    chosen_columns = state["chosen_columns"]

    async def fetch_column_info():
        async with MCPAdapter(Path("diabetes-db-server.py")) as adapter:
            tools = await adapter.list_tools()

            describe_tool = None
            for tool in tools:
                if tool.name == "describe_dtype_column":
                    describe_tool = tool

            raw_result = await describe_tool.ainvoke({"chosen_columns": chosen_columns})

            if isinstance(raw_result, list):
                result_text = ""
                for block in raw_result:
                    if isinstance(block, dict):
                        result_text = result_text + block.get("text", "")
            else:
                result_text = raw_result

            return result_text

    column_description = asyncio.run(fetch_column_info())

    prompt = f"""You are a biostatistics assistant helping choose appropriate statistical tests.

Here are the candidate features and their data types, and how many unique values each one has:
{column_description}

The outcome you are testing against is Diabetes_binary (0 = no diabetes/prediabetes, 1 = yes), a binary variable.

For EACH feature listed above, decide which statistical method is most appropriate to test
whether there is a significant association between that feature and Diabetes_binary. Different
features may need different tests depending on their type (e.g. a binary feature needs a
different test than a continuous one like BMI) — reason about each feature individually rather
than picking one test for all of them.

Briefly explain your reasoning for each feature (variable types, sample size, what the test
assumes), then end your answer with one line per feature in exactly this format, with no
other text in those lines:

METHOD: <feature name> = <name of the test>

For example (do not copy the content, only the structure):
METHOD: BMI = Mann-Whitney U test
METHOD: HighBP = Chi-square test
"""
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=9000,
    )

    log_usage("methods_agent", response)
    response_text = response.choices[0].message.content
    method_choice = {}

    if response_text:
        for line in response_text.split("\n"):
            if "METHOD:" in line:
                rest = line.split("METHOD:", 1)[1]
                parts = rest.split("=")
                if len(parts) == 2:
                    feature = parts[0].strip(" *")
                    method = parts[1].strip(" *")
                    if feature != "Diabetes_binary":
                        method_choice[feature] = method

    trace("methods_agent", "methods_chosen", methods=method_choice)
    new_log_entry = {"agent": "methods_agent", "action": f"agent decided what method is best to use for calculating the statistics, and the decision it made: {response_text}"}
    updated_log = state["decision_log"] + [new_log_entry]

    return {"method_choice": method_choice, "decision_log": updated_log}

def modelling_agent(state: ResearchState) -> dict:
    method_choice = state["method_choice"]
    given_data = fetch_dataframe(state["chosen_columns"])
    previous = state["calculated_data"]  #results from the last attempt, empty on the first one
    results = {}

    if not method_choice:
        results["_pipeline_error"] = {"error": "methods_agent produced no usable METHOD lines; method_choice was empty."}
        trace("modelling_agent", "pipeline_error", attempt=state["modelling_counter"] + 1, error=results["_pipeline_error"]["error"])
    else:
        for feature, method in method_choice.items():#we put this in a loop cause we running statistics for each method the LLM chose
            #keep results that already worked, only retry the ones that failed
            if feature in previous and "error" not in previous[feature]:
                results[feature] = previous[feature]
                continue

            #tell the agent what went wrong last time, so it doesn't repeat it
            previous_error = previous.get(feature, {}).get("error")
            if previous_error:
                error_note = f"\n    Your previous code for this feature failed with this error: {shorten(previous_error)}\n    Write new code that avoids this problem."
            else:
                error_note = ""

            prompt = f"""
    You are a Python data analyst. Write Python code that performs the following statistical
    test. Do not explain anything — output ONLY valid Python code, with no markdown
    formatting, no triple backticks, and no comments outside the code.

    Feature to test: {feature}
    Method to use: {method}

    A variable named `given_data` already exists in your execution environment as a pandas
    DataFrame. It contains the column `{feature}` and the outcome column `Diabetes_binary`
    (0 = no diabetes/prediabetes, 1 = yes), along with others you should ignore. Use
    `given_data` directly by that exact name — do not redefine it, rebuild it, or load it
    from anywhere else.

    Requirements:
    - Use only these libraries: scipy.stats, numpy, pandas, and statsmodels (for any test not
      available in scipy, such as a trend test). Do not import anything else, and do not attempt
      any file access, network access, or database connection — work only with the `given_data`
      variable already provided to you.
    - Only use the `{feature}` and `Diabetes_binary` columns from `given_data` for this test.
    - Only pass arguments that the function actually accepts. Do not invent arguments.
    - Run the test named above and print the result on its own line, in exactly this format:
      RESULT: statistic=<value>, p_value=<value>
      Print both values at full precision. Do not round them.
    - Also print a second line showing the direction of the relationship, so it can be
      reported alongside the test result. This line must describe what the numbers mean in
      plain words, not just a bare dictionary, so it can be understood correctly without
      needing to see this code:
      - If the test is a Chi-square test: compute the proportion of Diabetes_binary==1 within
        each unique value of `{feature}`, and print it as:
        GROUP_SUMMARY: proportion with diabetes for each value of {feature}: <your result here>
      - If the test is a Mann-Whitney U test: compute the median of `{feature}` separately for
        Diabetes_binary==0 and Diabetes_binary==1, and print it as:
        GROUP_SUMMARY: median {feature} for Diabetes_binary=0 group: <value>; median {feature}
        for Diabetes_binary=1 group: <value>
      - For any other test (e.g. an independent-samples t-test, ANOVA, or anything else not
        listed above): choose whichever group summary best matches what the test is comparing
        (for example, the mean of `{feature}` for each Diabetes_binary group, if the test
        compares means), and print it in the same clearly labeled style:
        GROUP_SUMMARY: <a clearly labeled description of the group difference>
      Always print a GROUP_SUMMARY line, no matter which test was used above — never skip it.
    - Finally print one line listing the arguments you passed to the test function, in exactly
      this format:
      SETTINGS: <argument>=<value>, <argument>=<value>
      Only list arguments you actually passed. Do not add extra arguments just to report them.
      For a chi-square test, always pass correction=True or correction=False explicitly.
    - The code must run standalone from top to bottom with no undefined variables, other than
      `given_data` itself.
    {error_note}
    """

            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=5000,
            )
            log_usage("modelling_agent", response)

            response_text = response.choices[0].message.content
            sandbox_result = run_sandbox(response_text, given_data)

            if sandbox_result["success"]:
                text = sandbox_result["stdout"]
                result_line = None
                summary_line = None
                settings_line = None

                for line in text.split("\n"):
                    if line.startswith("RESULT:"):
                        result_line = line
                    if line.startswith("GROUP_SUMMARY:"):
                        summary_line = line
                    if line.startswith("SETTINGS:"):
                        settings_line = line

                if result_line:
                    try:
                        rest = result_line.split("RESULT:", 1)[1]
                        parts = rest.split(",")
                        statistic_value = parts[0].split("=")
                        p_value = parts[1].split("=")
                        calculated = {"statistic": clean_number(statistic_value[1]), "p_value": clean_number(p_value[1])}
                        if summary_line:
                            calculated["group_summary"] = summary_line.split("GROUP_SUMMARY:", 1)[1].strip()
                        if settings_line:
                            calculated["settings"] = settings_line.split("SETTINGS:", 1)[1].strip()
                    except Exception as e:
                        calculated = {"error": f"Could not parse output '{shorten(text)}': {e}"}
                else:
                    calculated = {"error": f"No RESULT line found in output '{shorten(text)}'"}
            else:
                calculated = {"error": shorten(sandbox_result["error"])}

            #labelled group numbers and an independent check, see oversight.py.
            #outside the try, so a crash here is never blamed on the agent's code
            if "error" not in calculated:
                calculated = add_oversight(calculated, given_data, feature, method)

            results[feature] = calculated #return in the dict

            trace("modelling_agent", "test_result", attempt=state["modelling_counter"] + 1, feature=feature, method=method, result=calculated)

    kept = sum(1 for f in results if f in previous and results[f] is previous[f])
    new_log_entry = {"agent": "modelling_agent", "action": f"Ran one statistical test per feature ({kept} kept from the previous attempt). Results: {results}"}
    updated_log = state["decision_log"] + [new_log_entry]

    return {"calculated_data": results, "decision_log": updated_log, "modelling_counter": state["modelling_counter"] + 1}

def route_after_modelling(state: ResearchState) -> str:
    calculated_data = state["calculated_data"]
    has_error = any("error" in result for result in calculated_data.values())

    #if any feature has an error we go back to modelling_agent,
    #which only re-runs the features that failed (the rest are kept)
    if not has_error:
        return "approved"
    elif state["modelling_counter"] < 3:
        return "needs_revision"
    else:
        return "failed"

def interpretation_agent(state: ResearchState) -> dict:
    calculated= state["calculated_data"]#read what modelling_agent produced
    result_list = []

    for feature, result in calculated.items():
        if "error" in result:
            line = f"- {feature}: test failed with error: {result['error']}"
        else:
            #pass everything, including the code-computed "groups" with labelled direction
            line = f"- {feature}: {json.dumps(result)}"
        result_list.append(line)
    
    results_description = "\n".join(result_list)#we turn the result list from a list to a string

    if state["reviewer_feedback"] and not state["reviewer_verdict"]:#if feedback was not empty and verdit is false 
        error_note = f"\nYour previous interpretation was not approved by the reviewer, and the reason was: {state['reviewer_feedback']}\nInterpret the info again."
        trace("interpretation_agent", "revision", attempt=state["revision_counter"] + 1, feedback=state["reviewer_feedback"])
    else:
        error_note = ""

    prompt = f"""Here are statistical results from a health survey. Each of the following features
was tested individually against diabetes (Diabetes_binary), using the method shown for it:

{results_description}

Methods used (feature = test):
{state['method_choice']}

Based on these results, what can we say about the relationship between these features and
diabetes? Keep in mind this dataset has a very large sample size, so even a small, practically
unimportant difference can produce a statistically significant p-value, be careful not to
overstate any finding. This data is observational and cross-sectional, so describe any
relationship only as a statistical association, never as causation, do not use words like
"causes", "leads to", or "results in". Answer briefly, covering the features together, in a
maximum of 5 sentences.

Dataset coding: Age is an ordinal age-category variable (1–13), not age in years.
Report Age differences as differences in categories, never as years.
Do not describe differences in medians as differences in means.
Before describing any association, check the actual values in group_summary.
For categorical variables, each numeric category code maps to its own
specific value in group_summary. Verify this mapping before reporting
any percentage or identifying the lowest or highest category.
Never assign a value from one category to another category.
If the prevalence is not monotonic across categories, do not describe
it as a steady increase or decrease.
Never reverse the direction of a difference between groups.
If you mention a numerical difference, calculate it directly from the
reported group values and express it in percentage points when appropriate.
Do not invent or estimate numerical ranges that are not supported by the data.
If the direction or magnitude cannot be determined from the provided results,
state that it cannot be determined instead of guessing.

Variable meanings (codebook):
{make_codebook_text()}

Before interpreting, check that the data itself is plausible. Look at the group numbers for
each feature (mean, median, n_missing_dropped, diabetes_prevalence_pct) and compare them with
the codebook and with what is realistic for what the variable measures, including how much data
is missing. If any value looks impossible or suspicious,
say so clearly and do not present results that depend on it as a normal finding.
{error_note}
"""

    response = client.chat.completions.create(
    model=model_name,
    messages=[{"role": "user", "content": prompt}],
    max_tokens=5000,
    )
    
    log_usage("interpretation_agent", response)
    response_text = response.choices[0].message.content
    trace(
        "interpretation_agent", "interpretation",
        attempt=state["revision_counter"] + 1,
        text=response_text,
        finish_reason=response.choices[0].finish_reason,
    )

    new_log_entry = {"agent": "interpretation_agent", "action": f"interpreting the data that was given from data_agent, response was: {response_text}"}
    updated_log = state["decision_log"] + [new_log_entry]

    return {"interpretation": response_text, "decision_log": updated_log}

def reviewer_agent(state: ResearchState) -> dict:
    interpreted_text = state["interpretation"]

    prompt = f"""You are reviewing a written interpretation of a statistical result, to check whether it is accurate, not to re-run or re-judge the statistics yourself.

Here is the data and method that were used to produce the interpretation:
Calculated result: {state['calculated_data']}
Method used: {state['method_choice']}

Here is the interpretation text that was written based on that result:
{interpreted_text}

Variable meanings (codebook):
{make_codebook_text()}

Assume the statistical method and the computed numbers are correct, that is not what you are checking. But the data itself can be wrong. Your job is to decide whether the interpretation text above accurately and faithfully reflects the given data and method, and whether it notices data that is not plausible. Reject it (REVISE) if the text:
- misrepresents or contradicts the actual numbers or method
- ignores the data entirely and doesn't meaningfully engage with it
- draws a conclusion the data doesn't support
- is vague, off-topic, or unrelated to the actual statistical result
- presents a result as a normal finding even though the group numbers (mean, median, n_missing_dropped, prevalence) are impossible or suspicious for that variable according to the codebook and what is realistic for what the variable measures, without mentioning the problem

Respond with ONLY a single JSON object and nothing else, no markdown code fences, no text before or after it. The object must have exactly these three keys:
- "verdict": a boolean. true if the interpretation accurately reflects the data, false if it does not.
- "reason": a string explaining why you reached that verdict, referencing the specific data/method. This part is needed every single time, do not put a empty reason, but explain why you decided what do you did.
- "quote": if verdict is false, the exact sentence or phrase from the interpretation text that is wrong, copied word for word. If the problem is that something is missing from the text, use an empty string "". If verdict is true, use an empty string "".

Example of the exact shape required (do not copy the content it only serves as an example, copy only the structure):
{{"verdict": false, "reason": "The interpretation claims a significant effect, but the p-value of 0.42 does not support that.", "quote": "there is a significant effect of X on diabetes"}}
"""

    json_reformat = get_review("reviewer_agent", prompt, 8000, reviewed_text=interpreted_text)
    #the feedback the interpretation agent gets on a revision, always with a reason
    response_text = json.dumps(json_reformat)

    if json_reformat["verdict"]:
        reviewer_verdict = True
    else:
        reviewer_verdict = False

    revision_counter = state["revision_counter"]

    if not reviewer_verdict and revision_counter+1 == 3:
        failed_log = "Interpretation rejected after max revisions. Workflow will stop."
    else:
        failed_log =""

    trace("reviewer_agent", "verdict", attempt=revision_counter + 1, approved=reviewer_verdict, reason=json_reformat["reason"])

    new_log_entry = {"agent": "reviewer_agent", "action": f"Deciding if the interpretation was valid or not, this time the verdict was {reviewer_verdict} and reasoning was: {json_reformat['reason']}. {failed_log}"}
    updated_log = state["decision_log"] + [new_log_entry]

    return {"reviewer_feedback": response_text, "reviewer_verdict": reviewer_verdict, "revision_counter": state["revision_counter"]+1, "decision_log": updated_log}
    
def route_after_review(state: ResearchState) -> str:
    if state["reviewer_verdict"]:
        return "approved"
    if state["revision_counter"] < 3:
        return "needs_revision"
    return "failed"

import glob

def load_ml_results():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_files = glob.glob(os.path.join(script_dir, "Maskinlaeringsmodeller", "logs", "*.json"))
    if not log_files:
        return None

    log_files_sorted = sorted(log_files, key=os.path.getmtime, reverse=True)

    for log_file in log_files_sorted:
        with open(log_file, "r") as f:
            ml_data = json.load(f)
        if ml_data.get("check", {}).get("report_submitted"):
            return ml_data

    return None


def make_codebook_text():
    #the codebook as a list with one line per variable, for the prompt
    lines = []
    for name, meaning in CODEBOOK.items():
        lines.append(f"- {name}: {meaning}")
    return "\n".join(lines)


def report_agent(state: ResearchState) -> dict:
    ml_data = load_ml_results()

    if ml_data:
        ml_section_text = f"""
A separate predictive modeling component also tested several machine learning classifiers
on the same dataset. Results per model tried, including each model's feature importance
where available:
{ml_data['trained']}

The model selected as best, with its reasoning:
{ml_data['report']}
"""
    else:
        ml_section_text = "No predictive modeling results are available."
    
    #facts computed by code, put into the prompt so the LLM doesn't have to work them out itself
    codebook_text = make_codebook_text()
    settings_text = settings_summary(state["calculated_data"])
    facts_text = model_facts(ml_data)
    directions_text = direction_summary(state["calculated_data"])

    #the placeholders the LLM should write instead of tables.
    #kept in variables, because writing {{ }} inside an f-string needs confusing double braces
    stats_placeholder = "{{STATS_TABLE}}"
    model_placeholder = "{{MODEL_TABLE}}"

    if state["report_feedback"] and not state["report_verdict"]:
        error_note = f"\nYour previous report was not approved by the reviewer, and the reason was: {state['report_feedback']}\nWrite the report again, making sure to fix this specific issue."
        trace("report_agent", "revision", attempt=state["report_revision_counter"] + 1, feedback=state["report_feedback"])
    else:
        error_note = ""

    prompt = f"""You are writing the Methods, Results, and Discussion sections of a scientific
report, based on a statistical analysis of a health survey dataset, plus a separate predictive
modeling component.

Features analyzed (statistical testing): {state['chosen_columns']}

Statistical tests used (feature = test):
{state['method_choice']}

Raw statistical results (feature: statistic and p-value):
{state['calculated_data']}

Test settings actually used per feature (checked by code against a reference computation):
{settings_text}

Variable meanings (codebook), use these when describing any variable:
{codebook_text}

Model comparison facts (computed by code). Use ONLY these for any "highest", "best",
"lowest" or ranking claim about the models:
{facts_text}

Direction of each tested feature (computed by code from the data). Every direction you state
anywhere in the report, including the Discussion, must agree with these sentences, even when
a result is different from what you would expect:
{directions_text}

Only describe a feature as tested or statistically significant if it is listed above.
Features in the line "used by the models but not included in the univariate tests" were not tested.
Describe every variable with the codebook wording above (for example DiffWalk means difficulty walking).
Do not use the word "protective" or other causal wording.

Dataset coding: Sex = 0 means female; Sex = 1 means male.
When interpreting Sex, use these labels exactly. Do not reverse them.
Dataset coding: Age is an ordinal age-category variable (1–13), not age in years.
Report Age values as category codes, never as years.
A difference of two Age categories does not mean a difference of two years.
Do not describe differences in medians as differences in means.

Reviewed interpretation of these results:
{state['interpretation']}

Predictive modeling results:
{ml_section_text}

Write this as a short scientific report with these parts, in this order:

Methods: Describe the statistical tests performed for each feature against diabetes status.
State the test settings exactly as listed under "Test settings actually used per feature".
Never say a setting was used for all tests unless every feature lists it.
Say that directions and effect sizes come from descriptive group comparisons computed
alongside each test. Name the criterion used to select the best model before the results.
Describe predictive modeling only if predictive modeling results are provided above.
If no predictive modeling results are available, explicitly state that predictive modeling
results were unavailable and do not invent models or performance metrics.
Unavailable predictive modeling results do not prove that modeling was not performed.
Never claim that predictive models were not trained, generated, or conducted unless explicitly supported by the provided data.

Results: Do NOT write any tables yourself. Put the line {stats_placeholder} on its own line
where the statistical results table belongs, and the line {model_placeholder} on its own line
where the predictive modelling table belongs. Code replaces them with the real tables, which
include every statistic, p-value, effect size, direction, model metric, the baseline accuracy
and the feature importances. Around the placeholders, write a few sentences summarizing the
main findings. Any direction you mention must match the Direction column wording from the
results (the "direction" or "trend" fields). For ordinal variables, follow the "trend" field
exactly: if it says NOT monotonic, do not describe a steady increase or decrease.
If no predictive modeling results are available, do not invent any model results.

Discussion: Based only on the reviewed interpretation above, discuss what each statistical
result suggests about a possible association with diabetes status.
Only if predictive modeling results are available, discuss how well diabetes can be predicted
and whether the best model's most important features align with the statistically significant
features. If predictive modeling results are unavailable, do not discuss model performance.
Do not introduce any claims that are not supported by the provided results or interpretation.

Limitations: State that this data is observational and cross-sectional, so these are
associations only, not causal relationships.
Only if predictive modeling results are available, discuss the limitations of training
models on the raw, imbalanced dataset without class-imbalance handling, including why
accuracy may be misleading relative to the majority-class baseline.
Mention recall for the diabetic class only if that metric is provided in the results.
If predictive modeling results are unavailable, state that predictive model performance
could not be evaluated.

Do not use causal language anywhere (no "causes", "leads to", "results in"). Write in a formal
academic tone.
Do not state how many variables or features were analysed; list them by name instead.
Do not refer to supplementary tables, appendices, earlier literature or any source that is
not provided above.
Do not call an effect "large", "sizeable" or "marked" unless the effect size in the table
supports it (Cramér's V or rank-biserial r of about 0.3 or more).
Do not write any tables anywhere in the report, including Methods. The settings for each
feature are already shown in the statistics table.
Do not mention the placeholders or that tables will be inserted.

Data quality: before writing, check the group numbers in the raw results (mean, median,
n_missing_dropped, prevalence) against the codebook and against what is realistic for each
variable. End the Methods section with a short paragraph on data quality. If any value looks
impossible or suspicious, name the feature and the problem there,
and say in the Discussion and Limitations how it may affect the results. Never present a result
that depends on implausible values as a normal finding. If nothing looks implausible, say so.
{error_note}
"""
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=20000,
    )
    log_usage("report_agent", response)

    response_text = response.choices[0].message.content

    if not response_text or not response_text.strip():
        raise ValueError("Report agent returned an empty scientific report")

    #the LLM only wrote placeholders, now code builds the real tables and puts them in
    statistics_table = stats_table(state["calculated_data"])
    models_table = model_table(ml_data, state["calculated_data"])
    #the LLM sometimes puts a placeholder in the middle of a sentence ("is presented in {{STATS_TABLE}}."),
    #and then the table breaks the text. a markdown table only works on its own lines,
    #so always put blank lines around the placeholders before the tables go in
    response_text = response_text.replace(stats_placeholder, f"\n\n{stats_placeholder}\n\n")
    response_text = response_text.replace(model_placeholder, f"\n\n{model_placeholder}\n\n")
    response_text = render_report(response_text, statistics_table, models_table)

    trace("report_agent", "report_written", attempt=state["report_revision_counter"] + 1, report=response_text)
    new_log_entry = {"agent": "report_agent", "action": "Generated the full report, combining statistical testing and predictive modeling results."}
    updated_log = state["decision_log"] + [new_log_entry]

    return {"scientific_report": response_text, "decision_log": updated_log, "report_revision_counter": state["report_revision_counter"] + 1}

def report_reviewer_agent(state: ResearchState) -> dict:
    report_text = state["scientific_report"]
    ml_data = load_ml_results()

    if ml_data:
        #summary was too bix for the context window so we are dropping duplicate and uncesecary stuff
        model_summaries = []
        for trained_model_name, result in ml_data["trained"].items():
            model_summaries.append(f"{trained_model_name}: accuracy={result['accuracy']}, auc={result['auc']}")

        best_model_name = ml_data["report"].get("best_model")
        best_model_result = ml_data["trained"].get(best_model_name, {})
        top_features = None
        if best_model_result.get("feature_importance"):
            top_features = sorted(
                best_model_result["feature_importance"].items(),
                key=lambda x: -abs(x[1])
            )[:5]

        ml_ground_truth = f"""Models actually trained: {list(ml_data['trained'].keys())}
Accuracy/AUC per model: {'; '.join(model_summaries)}
Best model submitted: {ml_data['report']}
Best model's top 5 features by importance: {top_features}"""
    else:
        ml_ground_truth = "No predictive modeling results were available when the report was written."

    #the same facts the report agent got, so the reviewer can check the prose against them
    facts_text = model_facts(ml_data)
    settings_text = settings_summary(state["calculated_data"])

    prompt = f"""You are reviewing a scientific report for internal accuracy, not to re-run any
statistics yourself.

Here is the ground-truth data the report should be based on:
Statistical tests used (feature = test): {state['method_choice']}
Raw statistical results (feature: statistic and p-value): {state['calculated_data']}
Reviewed interpretation: {state['interpretation']}
Predictive modeling ground truth:
{ml_ground_truth}
Model comparison facts: {facts_text}
Test settings actually used per feature: {settings_text}

The tables in the report were generated by code directly from the raw results and are correct.
Check the prose around them against the tables and the data above.

Here is the report text to check:
{report_text}

Check specifically for:
- Any feature described with a different test name than the one listed above (e.g. calling a
  Mann-Whitney U test result a "chi-square statistic").
- Any statistic, p-value, or other number in the report that does not match the raw results
  given above.
- Any claim about predictive modeling (model count, model names, accuracy, AUC, feature
  importance) that isn't grounded in the predictive modeling ground truth given above.
- Internal contradictions, where one part of the report states something that a different part
  of the same report contradicts (including the prose contradicting a table).
- Any direction claim that does not match the "direction" or "trend" field for that feature,
  for example calling a non-monotonic trend a steady increase, or reading GenHlth as
  "higher = better" when higher means worse health.
- Any "best", "highest" or "lowest" claim about a model that contradicts the model comparison facts.
- A Methods claim that one test setting was used for all features when the per-feature
  settings above say otherwise.
- A count of variables that does not match the features listed, or a reference to a
  supplementary table, appendix or literature that does not exist.
- Implausible data that the report does not mention: compare the group numbers in the raw results
  (mean, median, n_missing_dropped, prevalence) with the variable meanings below and with what is
  realistic for what the variable measures, including how much data is missing.
  If such a problem exists and the report presents the affected result as a normal finding, reject it.

Variable meanings (codebook):
{make_codebook_text()}

Respond with ONLY a single JSON object and nothing else, no markdown code fences, no text
before or after it. The object must have exactly these three keys:
- "verdict": a boolean. true if the report is accurate and internally consistent, false if not.
- "reason": a string explaining why, citing the specific mismatch if verdict is false.
- "quote": if verdict is false, the exact sentence or phrase from the report that is wrong, copied
  word for word. If the problem is that something is missing from the report, use an empty string "".
  If verdict is true, use an empty string "".

Example of the exact shape required (do not copy the content, only the structure):
{{"verdict": false, "reason": "The report describes BMI as tested with a chi-square test, but method_choice lists BMI as Mann-Whitney U test.", "quote": "BMI was compared with a chi-square test"}}
"""
    #16000, not 8000: the report prompt is large and this reviewer used 7000-8000 tokens and ran out
    json_reformat = get_review("report_reviewer_agent", prompt, 16000, reviewed_text=report_text)

    report_verdict = json_reformat["verdict"]
    if not report_text or not report_text.strip():
        report_verdict = False
        json_reformat["verdict"] = False
        json_reformat["reason"] = "Scientific report is empty and cannot be approved."

    #the feedback the report agent gets on a revision, always with a reason
    response_text = json.dumps(json_reformat)

    revision_counter = state["report_revision_counter"]
    if not report_verdict and revision_counter + 1 == 3:
        failed_log = "Scientific report rejected after max revisions. Workflow will stop."
    else:
        failed_log = ""

    trace("report_reviewer_agent", "verdict", attempt=revision_counter, approved=report_verdict, reason=json_reformat["reason"])
    new_log_entry = {"agent": "report_reviewer_agent", "action": f"Deciding if the report was accurate or not, this time the verdict was {report_verdict} and reasoning was: {json_reformat['reason']}. {failed_log}"}
    updated_log = state["decision_log"] + [new_log_entry]

    return {"report_feedback": response_text, "report_verdict": report_verdict, "decision_log": updated_log}

def route_after_report_review(state: ResearchState) -> str:
    if state["report_verdict"]:
        return "approved"
    if state["report_revision_counter"] < 3:
        return "needs_revision"
    return "failed"

#create a mini graph using LangGraph
builder = StateGraph(ResearchState) #we tie it to our state schema

#register the nodes with name asigned
builder.add_node("data_agent", data_agent)
builder.add_node("methods_agent", methods_agent)
builder.add_node("modelling_agent", modelling_agent)
builder.add_node("interpretation_agent", interpretation_agent)
builder.add_node("reviewer_agent", reviewer_agent)
builder.add_node("report_agent", report_agent)
builder.add_node("report_reviewer_agent", report_reviewer_agent)

#wire the steps
builder.add_edge(START, "data_agent")          
builder.add_edge("data_agent", "methods_agent")  
builder.add_edge("methods_agent", "modelling_agent")

builder.add_conditional_edges(
    "modelling_agent", route_after_modelling,
    {
        "approved": "interpretation_agent",
        "needs_revision": "modelling_agent",
        "failed": END
    }
)
 
builder.add_edge("interpretation_agent", "reviewer_agent")

#this conditional node is so that if we get false from reviewer, we can loop back until its satisfied
builder.add_conditional_edges(
    "reviewer_agent", route_after_review,
    {
        #if route_after_review function returns approved we end, or we loop back into interpretation
        "approved": "report_agent",
        "needs_revision": "interpretation_agent",
        "failed": END
    },
)

builder.add_edge("report_agent", "report_reviewer_agent")

builder.add_conditional_edges(
    "report_reviewer_agent", route_after_report_review,
    {
        "approved": END,
        "needs_revision": "report_agent",
        "failed": END
    },
)

#compile to something runnable
graph = builder.compile()


def run_workflow(run_id):
    init_log(run_id)  #new log file for every run, same id as the report
    trace("workflow", "run_started")

    #generate graph visualization when the workflow is triggered.
    #the picture is only for documentation, a failure here should not stop the run
    try:
        graph.get_graph().draw_mermaid_png(output_file_path="/app/output/graph.png")
    except Exception as e:
        trace("workflow", "graph_png_failed", error=e)

    result = graph.invoke({
        "chosen_columns": [],
        "method_choice": {},
        "calculated_data": {},
        "interpretation": "",
        "reviewer_feedback": "",
        "reviewer_verdict": False,
        "revision_counter": 0,
        "modelling_counter": 0,
        "decision_log": [],
        "scientific_report": "",
        "report_feedback": "",
        "report_verdict": False,
        "report_revision_counter": 0,
    })

    if any("error" in r for r in result["calculated_data"].values()):
        final_status = "modelling_failed"
    elif not result["reviewer_verdict"]:
        final_status = "interpretation_rejected"
    elif not result["report_verdict"]:
        final_status = "report_rejected"
    else:
        final_status = "completed"

    trace(
        "workflow", "run_finished",
        status=final_status,
        interpretation_attempts=result["revision_counter"],
        report_attempts=result["report_revision_counter"],
    )

    #save the report as a markdown file in the output folder (mounted to ./output on the host)
    if result["scientific_report"]:
        report_path = f"/app/output/report_{run_id}.md"
        with open(report_path, "w") as f:
            f.write(result["scientific_report"])
        trace("workflow", "report_saved", path=report_path)
        
    return result