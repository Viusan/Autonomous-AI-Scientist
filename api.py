import importlib.util

from fastapi import FastAPI, HTTPException

spec = importlib.util.spec_from_file_location(
    "langgraph_orc",
    "langgraph-orc.py"
)
langgraph_orc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(langgraph_orc)

app = FastAPI()


@app.get("/")
def health_check():
    return {"status": "API is running"}


@app.post("/run")
def run_analysis():
    try:
        result = langgraph_orc.run_workflow()

        return {
            "status": "completed",
            "scientific_report": result["scientific_report"]
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
