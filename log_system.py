import json, os, time
from datetime import datetime, timezone

_LOG_PATH = None

def init_log(run_id: str | None = None) -> str:
    #this makes sure that log folder exists
    global _LOG_PATH #inside func variables are new local, but this says use existing global
    script_dir = os.path.dirname(os.path.abspath(__file__))
    #create the logs folder if it doesnt exist
    logs_dir = os.path.join(script_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    run_id = run_id or str(int(time.time()))
    _LOG_PATH = os.path.join(logs_dir, f"run_{run_id}.jsonl")
    return _LOG_PATH

def log_event(agent: str, event: str, **fields):
    if _LOG_PATH is None:
        init_log()
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "agent": agent,
        "event": event,
        **fields,
    }
    with open(_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str) + "\n")