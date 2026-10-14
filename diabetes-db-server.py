import pandas as pd
import sqlite3
import json

from mcp.server.mcpserver import MCPServer

mcp = MCPServer("diabetes-db")

DB_PATH = "diabetes.db"
TABLE_NAME = "diabetes_binary"

@mcp.tool()
def list_columns() -> str:
    """List all the column names in the table"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(f"PRAGMA table_info({TABLE_NAME})")#gets table info from diabtes_binary (pragma kinda diff check out online to see visualization)
    columns = [row[1] for row in cursor.fetchall()]#row 1 from pragma displays column names 
    names_in_column = [c for c in columns if c != "Diabetes_binary"]#saw that diabetes_binary was included, which is not a feature, so we avoid that
    
    conn.close()
    return "\n".join(names_in_column)

@mcp.tool()
def describe_dtype_column() -> str:
    """Return data type and unique-value-count summary for each column."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(f"PRAGMA table_info({TABLE_NAME})")#gets table info from diabtes_binary (pragma kinda diff check out online to see visualization)
    columns = [row[1] for row in cursor.fetchall()]#row 1 from pragma displays column names 
    names_in_column = [c for c in columns if c != "Diabetes_binary"]#saw that diabetes_binary was included, which is not a feature, so we avoid that
    
    column_list = []

    for column_name in names_in_column:
        df = pd.read_sql(f"SELECT {column_name} FROM {TABLE_NAME}", conn)
        dtype = df[column_name].dtype
        unique_count = df[column_name].nunique() #returns how many distinct values show up in column like if only 0 and 1 we get 2.

        line = f"- {column_name}: {dtype}, {unique_count} unique values"
        column_list.append(line)

    conn.close()
    return "\n".join(column_list)
    
@mcp.tool()
def run_query(chosen_columns: list[str]) -> str: #taking in column name instead of sql string is safer incase of sql injection
    """Run a read-only SELECT query against the diabetes_binary table."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(f"PRAGMA table_info({TABLE_NAME})")
    columns = [row[1] for row in cursor.fetchall()]
    names_in_column = [c for c in columns if c != "Diabetes_binary"]

    validated_features = [c for c in chosen_columns if c in names_in_column]#we do this incase the LLM returns slighty diff names for columns, then SQL query will be broken
    if validated_features:
        columns_str = ", ".join(validated_features)
    else:
        #if our earlier fails then we still get a valid sql string from this
        columns_str = ", ".join(names_in_column)
    
    query = f"SELECT {columns_str}, Diabetes_binary FROM {TABLE_NAME}"

    #dataframe is a returned as a python object we can work with
    #good thing here is column name is attached to data (ask claude to visualize if you dont understand)
    data_frame = pd.read_sql(query, conn)
    df_dictionary = data_frame.to_dict(orient="records")#we turn dataframe into a python dictionary

    conn.close()
    return json.dumps(df_dictionary)

if __name__ == "__main__":
    mcp.run(transport="stdio")