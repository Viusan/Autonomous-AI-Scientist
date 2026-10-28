import os
import sys
import json
import time
import asyncio
from dotenv import load_dotenv
from openai import OpenAI, InternalServerError, APIConnectionError
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from ml_bibliotek import TILLATTE_TABELLER

load_dotenv()

client = OpenAI(
    api_key="dummy",
    base_url=os.getenv("OSBOT_BASE_URL"),
    default_headers={"x-api-key": os.getenv("OSBOT_API_KEY")},
)
model_name = os.getenv("LOCAL_MODEL")

#finn ut hvilken modell som best forutsier diabetes, se på dataen først, prøv minst 4 modeller, og avslutt med submit_report. Ikke finn på tall.

SYSTEM = """You are an autonomous research assistant working on a health survey dataset (BRFSS 2015).
The goal: find out which machine learning model best predicts whether a person has diabetes
(target column Diabetes_binary), and report it for the results section of a research article.

You have tools. Decide yourself which tools to call and in which order. Look at the data first, then
try at least 4 different models (mix simple and strong ones), compare them, and finish by calling submit_report.
Only use numbers that train_model returned. Do not invent numbers."""

#beskrivelsen av tool-et LLM-en bruker for å levere svaret: beste modell, accuracy, AUC, begrunnelse og eventuelle notater om dataen.
SUBMIT_TOOL = {"type": "function", "function": {
    "name": "submit_report",
    "description": "Submit your final conclusion. Call this once when you are done. Use only numbers returned by train_model.",
    "parameters": {"type": "object", "properties": {
        "best_model": {"type": "string", "description": "exact model name"},
        "accuracy": {"type": "number"}, "auc": {"type": "number"},
        "reasoning": {"type": "string", "description": "3 to 4 sentences on why this model is best"},
        "data_notes": {"type": "string", "description": "anything about the data you think is worth mentioning, or empty"}},
        "required": ["best_model", "accuracy", "auc", "reasoning"]}}}


#Hvis modellen gir en feil eller mister forbindelsen, venter den noen sekunder og prøver på nytt, opptil 4 ganger. Hvis det fortsatt feiler, gir den None, og agenten stopper.
def spor_llm(messages, tools, forsok=4):
    for i in range(forsok):
        try:
            return client.chat.completions.create(model=model_name, messages=messages, tools=tools,
                                                  tool_choice="auto", max_tokens=4000)
        except (InternalServerError, APIConnectionError) as e:
            print(f"LLM error (attempt {i + 1}/{forsok}): {str(e)[:200]}")
            time.sleep(3 * (i + 1))
    return None


async def kjor_agent(tabell, maks_steg=25):
    if tabell not in TILLATTE_TABELLER:
        raise SystemExit(f"Unknown table {tabell}. Choose one of {TILLATTE_TABELLER}")

    mappe = os.path.dirname(os.path.abspath(__file__))
    server = StdioServerParameters(
        command=sys.executable, args=[os.path.join(mappe, "ml_server.py")],
        env={**os.environ, "ML_DB": os.path.join(mappe, "diabetes.db"), "PYTHONPATH": mappe})

    rapport, trent, log = {}, {}, []

    async with stdio_client(server) as (les, skriv):
        async with ClientSession(les, skriv) as session:
            await session.initialize()

            # Agenten spør serveren hvilke tools den har, og gjør dem om til formatet LLM-en forstår
            mcp_tools = (await session.list_tools()).tools
            tools = [{"type": "function", "function": {"name": t.name, "description": t.description or "",
                                                       "parameters": getattr(t, "inputSchema", None) or t.input_schema}} for t in mcp_tools]  # mcp 1.x / 2.x
            tools.append(SUBMIT_TOOL)
            print("Tools from MCP server:", [t.name for t in mcp_tools])

            messages = [{"role": "system", "content": SYSTEM}, # Skriver det om slik at LLM-en forstår hva som er skrevet og lagrer resultatene.
                        {"role": "user", "content": f"Use the table '{tabell}'."}]

            for steg in range(maks_steg):
                svar = spor_llm(messages, tools)
                if svar is None:
                    print("The LLM kept failing, stopping here.")
                    break
                msg = svar.choices[0].message
                kall = msg.tool_calls or []
                
                for c in kall:
                    c.function.name = c.function.name.split("<|")[0].strip()

                melding = {"role": "assistant", "content": msg.content or ""}
                if kall:
                    melding["tool_calls"] = [{"id": c.id, "type": "function",
                                              "function": {"name": c.function.name, "arguments": c.function.arguments}}
                                             for c in kall]
                messages.append(melding)

                if not kall:
                    if rapport:
                        break
                    messages.append({"role": "user", "content": "Continue with the tools and finish by calling submit_report."})
                    continue

                for c in kall:
                    navn = c.function.name
                    try:
                        args = json.loads(c.function.arguments or "{}")
                        if navn == "submit_report":
                            rapport.update(args)
                            resultat = {"status": "report received, you are done"}
                        else:
                            ut = await session.call_tool(navn, args)   # her går kallet til MCP-serveren
                            tekst = "".join(b.text for b in ut.content if hasattr(b, "text"))
                            feilet = getattr(ut, "isError", None) or getattr(ut, "is_error", False)  # mcp 1.x / 2.x
                            resultat = {"error": tekst} if feilet else json.loads(tekst)
                            if navn == "train_model" and "model" in resultat:
                                trent[resultat["model"]] = resultat
                    except Exception as e:
                        args, resultat = c.function.arguments, {"error": str(e)}
                    log.append({"step": steg, "tool": navn, "args": args, "result": resultat})
                    print(f"[{steg}] {navn}({args}) -> {str(resultat)[:200]}")
                    messages.append({"role": "tool", "tool_call_id": c.id, "content": json.dumps(resultat)[:6000]})

                if rapport:
                    break

    return rapport, trent, log


def kontroller(rapport, trent, toleranse=0.005):
    # Sjekker LLM-ens svar mot det tool-ene faktisk returnerte
    if not rapport:
        return {"report_submitted": False}
    valgt = rapport["best_model"]
    gyldige = {n: r for n, r in trent.items() if r["auc"] is not None}
    ref = max(gyldige.values(), key=lambda r: r["auc"]) if gyldige else None
    tallene_stemmer = (valgt in trent and abs(rapport["accuracy"] - trent[valgt]["accuracy"]) < 0.001
                       and trent[valgt]["auc"] is not None and abs(rapport["auc"] - trent[valgt]["auc"]) < 0.001)
    return {
        "report_submitted": True,
        "models_tried": len(trent),
        "model_was_trained": valgt in trent,
        "reference_best_by_auc": ref["model"] if ref else None,
        "exact_match_best_auc": bool(ref) and valgt == ref["model"],
        "within_tolerance": bool(ref) and valgt in trent and trent[valgt]["auc"] is not None
                            and ref["auc"] - trent[valgt]["auc"] <= toleranse + 1e-9,
        "numbers_match_tool_output": tallene_stemmer,
    }


if __name__ == "__main__":
    tabell = sys.argv[1] if len(sys.argv) > 1 else "diabetes_binary"
    rapport_ut, trent, log = asyncio.run(kjor_agent(tabell))
    sjekk = kontroller(rapport_ut, trent)

    print("\n=== RAPPORT ===")
    print(json.dumps(rapport_ut, indent=2, ensure_ascii=False))
    print("\n=== KONTROLL ===")
    print(json.dumps(sjekk, indent=2))

    mappe = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(mappe, "logs"), exist_ok=True)
    fil = os.path.join(mappe, "logs", f"{tabell}_{int(time.time())}.json")
    with open(fil, "w") as f:
        json.dump({"table": tabell, "report": rapport_ut, "check": sjekk, "trained": trent, "log": log},
                  f, indent=2, ensure_ascii=False, default=str)
    print("\nLog saved to", fil)