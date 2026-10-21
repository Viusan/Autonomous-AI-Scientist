import os
import sqlite3
import io
import contextlib
import json
import pandas as pd
import asyncio

from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from typing import TypedDict
from mcp_helpers import fetch_dataframe
from langchain.mcp import MCPAdapter
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.messages import ToolMessage

load_dotenv()

api_key = os.getenv("OSBOT_API_KEY")
base_url = os.getenv("OSBOT_BASE_URL")
model_name = os.getenv("LOCAL_MODEL")

model = ChatOpenAI( #this is for the langchain openai object (so we gonna use this for the tools)
    model=model_name,
    api_key="dummy",
    base_url=base_url,
    max_tokens=5000,
    default_headers={"x-api-key": api_key},
)

client = OpenAI(
    api_key="dummy",
    base_url=base_url,
    default_headers={"x-api-key": api_key},
)

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

def data_agent(state: ResearchState) -> dict:
    #conversation history, we will add to this as tools are called to LLM gets a history of messages
    messages = [HumanMessage(content="Find out which columns exist in the table, then fetch the data for what you percieve to be important factors that could cause diabetes, and decide when you are done.")]

    async def explore_and_select():
        async with MCPAdapter(Path("diabetes-db-server.py")) as adapter:
            tools = await adapter.list_tools()
            model_with_tools = model.bind_tools(tools)

            response = await model_with_tools.ainvoke(messages)
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
                    raw_result = await matching_tool.ainvoke(tool_call["args"])

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
            return final_columns

    final_columns = asyncio.run(explore_and_select())
    if final_columns is None: #safeguard incase we get nothing 
        final_columns = []

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

Here are the candidate features and their data types:
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
                    method_choice[feature] = method

    new_log_entry = {"agent": "methods_agent", "action": f"agent decided what method is best to use for calculating the statistics, and the decision it made: {response_text}"}
    updated_log = state["decision_log"] + [new_log_entry]

    return {"method_choice": method_choice, "decision_log": updated_log}

#this wont be in final code, this should be in docker sandbox
def run_sandbox(code: str, given_data) -> dict: #utility function
    buffer = io.StringIO() #empty fake in-memory
    try:
        with contextlib.redirect_stdout(buffer): #whatever is printed inside here goes into the memory
            exec(code, {"given_data": given_data})
            captured_text = buffer.getvalue()   #pulls everything from buffer to string
            return {"success": True, "stdout": captured_text, "error": None}
    except Exception as e:
        return {"success": False, "stdout": buffer.getvalue(), "error": str(e)}

def modelling_agent(state: ResearchState) -> dict:
    method_choice = state["method_choice"]
    given_data = fetch_dataframe(state["chosen_columns"])
    results = {}

    if not method_choice:
            results["_pipeline_error"] = {"error": "methods_agent produced no usable METHOD lines; method_choice was empty."}
    else:
        for feature, method in method_choice.items():#we put this in a loop cause we running statistics for each method the LLM chose
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
    - Use only these libraries: scipy.stats, numpy, pandas. Do not import anything else, and do
      not attempt any file access, network access, or database connection — work only with the
      `given_data` variable already provided to you.
    - Only use the `{feature}` and `Diabetes_binary` columns from `given_data` for this test.
    - Run the test named above and print the result in exactly this format, with no other
      output:
      RESULT: statistic=<value>, p_value=<value>
    - The code must run standalone from top to bottom with no undefined variables, other than
      `given_data` itself.
    """

            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1500,
            )

            response_text = response.choices[0].message.content
            sandbox_result = run_sandbox(response_text, given_data)

            if sandbox_result["success"]:
                text = sandbox_result["stdout"]
                try:
                    parts = text.split(",")
                    statistic_value = parts[0].split("=")
                    p_value = parts[1].split("=")
                    calculated = {"statistic": float(statistic_value[1]), "p_value": float(p_value[1])}
                except Exception as e:
                    calculated = {"error": f"Could not parse output '{text}': {e}"}
            else:
                calculated = {"error": sandbox_result["error"]}

            results[feature] = calculated #return in the dict 

    new_log_entry = {"agent": "modelling_agent", "action": f"Ran one statistical test per feature. Results: {results}"}
    updated_log = state["decision_log"] + [new_log_entry]

    return {"calculated_data": results, "decision_log": updated_log, "modelling_counter": state["modelling_counter"] + 1}

def route_after_modelling(state: ResearchState) -> str:
    calculated_data = state["calculated_data"]
    has_error = any("error" in result for result in calculated_data.values())

    #if one of the methods returned as a error we just run the modelling agent again
    #idk maybe we can make this more effective by only re-running the method that got the error
    #cause now we re-running the whole method list, maybe waste of tokens?
    if not has_error:
        return "approved"
    elif state["modelling_counter"] < 3:
        return "needs_revision"
    else:
        return "failed"

def interpretation_agent(state: ResearchState) -> dict:
    calculated= state["calculated_data"]#read what modelling_agent produced
    result_list = []

    for feature, result in calculated.items():#just checking through which methods failed and which didnt
        if "error" in result:
            line = f"- {feature}: test failed with error: {result['error']}"
        else:
            line = f"- {feature}: statistic={result['statistic']}, p_value={result['p_value']}"
        result_list.append(line)
    
    results_description = "\n".join(result_list)#we turn the result list from a list to a string

    if state["reviewer_feedback"] and not state["reviewer_verdict"]:#if feedback was not empty and verdit is false 
        error_note = f"\nYour previous interpretation was not approved by the reviewer, and the reason was: {state['reviewer_feedback']}\nInterpret the info again."
    else:
        error_note = ""

    prompt = f"""Here are statistical results from a health survey. Each of the following features
was tested individually against diabetes (Diabetes_binary), using the method shown for it:

{results_description}

Methods used (feature = test):
{state['method_choice']}

Based on these results, what can we say about the relationship between these features and
diabetes? Keep in mind this dataset has a very large sample size, so even a small, practically
unimportant difference can produce a statistically significant p-value — be careful not to
overstate any finding. Answer briefly, covering the features together, in a maximum of 5
sentences.
{error_note}
"""

    response = client.chat.completions.create(
    model=model_name,
    messages=[{"role": "user", "content": prompt}],
    max_tokens=1500,
    )
    
    response_text = response.choices[0].message.content

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

Assume the data and method themselves are already correct, that is not what you are checking. Your only job is to decide whether the interpretation text above accurately and faithfully reflects the given data and method. Reject it (REVISE) if the text:
- misrepresents or contradicts the actual numbers or method
- ignores the data entirely and doesn't meaningfully engage with it
- draws a conclusion the data doesn't support
- is vague, off-topic, or unrelated to the actual statistical result

Respond with ONLY a single JSON object and nothing else, no markdown code fences, no text before or after it. The object must have exactly these two keys:
- "verdict": a boolean. true if the interpretation accurately reflects the data, false if it does not.
- "reason": a string explaining why you reached that verdict, referencing the specific data/method. This part is needed every single time, do not put a empty reason, but explain why you decided what do you did.

Example of the exact shape required (do not copy the content it only serves as an example, copy only the structure):
{{"verdict": false, "reason": "The interpretation claims a significant effect, but the p-value of 0.42 does not support that."}}
"""

    response = client.chat.completions.create(
    model=model_name,
    messages=[{"role": "user", "content": prompt}],
    max_tokens=1500,
    )

    response_text = response.choices[0].message.content
    json_reformat = json.loads(response_text)

    if json_reformat["verdict"]:
        reviewer_verdict = True
    else:
        reviewer_verdict = False

    revision_counter = state["revision_counter"]

    if not reviewer_verdict and revision_counter+1 == 3:
        failed_log = "revision counter hit its limit, despite having a false verdict we will still proceed."
    else:
        failed_log =""

    new_log_entry = {"agent": "reviewer_agent", "action": f"Deciding if the interpretation was valid or not, this time the verdict was {reviewer_verdict} and reasoning was: {json_reformat['reason']}. {failed_log}"}
    updated_log = state["decision_log"] + [new_log_entry]

    return {"reviewer_feedback": response_text, "reviewer_verdict": reviewer_verdict, "revision_counter": state["revision_counter"]+1, "decision_log": updated_log}
    
def route_after_review(state: ResearchState) -> str:
    if state["revision_counter"] < 3: #just so we dont create a infinite loop
        if state["reviewer_verdict"]:
            return "approved"
        else:
            return "needs_revision"
    else:
        return "approved"

#create a mini graph using LangGraph
builder = StateGraph(ResearchState) #we tie it to our state schema

#register the nodes with name asigned
builder.add_node("data_agent", data_agent)
builder.add_node("methods_agent", methods_agent)
builder.add_node("modelling_agent", modelling_agent)
builder.add_node("interpretation_agent", interpretation_agent)
builder.add_node("reviewer_agent", reviewer_agent)

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
        "approved": END,
        "needs_revision": "interpretation_agent"
    },
)

#compile to something runnable
graph = builder.compile()

graph.get_graph().draw_mermaid_png(output_file_path="graph.png") #create a graph to vizualise how the nodes interact with each other

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
})

#print("Interpretation: ", result["interpretation"])
#print("Method choice: ", result["method_choice"])
#print("Given data: ", result["given_data"])
print("Decision log: ", result["decision_log"])
#print("Calculated data: ", result["calculated_data"])