import json

try:
    from mcp.server.mcpserver import MCPServer as Server
except ImportError:
    from mcp.server.fastmcp import FastMCP as Server

import ml_tools

mcp = Server("diabetes-ml")


def kjor(funksjon, *args):
    # Kjører funksjonen. Hvis den feiler, får LLM-en feilmeldingen som tekst, så den kan rette seg selv.
    try:
        return json.dumps(funksjon(*args))
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
def list_tables() -> str:
    """List the database tables you may use, with row counts."""
    return kjor(ml_tools.tool_list_tables)


@mcp.tool()
def describe_table(table: str) -> str:
    """Summary of a table: for every column the min, max, mean and number of missing values."""
    return kjor(ml_tools.tool_describe_table, table)


@mcp.tool()
def list_models() -> str:
    """List the machine learning models you can train, with notes (e.g. slow models)."""
    return kjor(ml_tools.tool_list_models)


@mcp.tool()
def train_model(table: str, model: str, params: dict = {}) -> str:
    """Train one model on the table (80/20 split, rows with missing values dropped) and return
    accuracy, precision, recall, f1 and auc on the test set, plus the baseline accuracy.
    Optional params are hyperparameters for the model, e.g. {"n_estimators": 100}."""
    return kjor(ml_tools.tool_train_model, table, model, params)


if __name__ == "__main__":
    mcp.run()   # snakker med agenten via stdin/stdout