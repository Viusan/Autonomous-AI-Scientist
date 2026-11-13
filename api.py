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
    except Exception as e:
        #only real crashes give 500 now
        raise HTTPException(status_code=500, detail=str(e))

    if any("error" in r for r in result["calculated_data"].values()):
        return {
            "status": "modelling_failed",
            "approved": False,
            "calculated_data": result["calculated_data"],
        }

    if not result["reviewer_verdict"]:
        return {
            "status": "interpretation_rejected",
            "approved": False,
            "interpretation": result["interpretation"],
            "reviewer_feedback": result["reviewer_feedback"],
        }

    if not result["report_verdict"]:
        return {
            "status": "report_rejected",
            "approved": False,
            "scientific_report": result["scientific_report"],
            "report_feedback": result["report_feedback"],
        }

    return {
        "status": "completed",
        "approved": True,
        "scientific_report": result["scientific_report"],
    }
