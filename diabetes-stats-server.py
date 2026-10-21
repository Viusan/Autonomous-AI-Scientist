import sqlite3
import json
import pandas as pd

from scipy import stats
from mcp.server.mcpserver import MCPServer

mcp = MCPServer("diabetes-stats")

DB_PATH = "diabetes.db"
TABLE_NAME = "diabetes_binary"

@mcp.tool()
def run_statistical_test(feature: str, test_name: str) -> str:
    """Run a statistical test comparing one feature against diabetes status (Diabetes_binary), and return the test's statistic and p-value.

test_name must be exactly "chi_square" or "mann_whitney":
- Use "chi_square" when the feature is categorical (e.g. a 0/1 flag like HighBP, or a small number of discrete categories).
- Use "mann_whitney" when the feature is continuous or has many possible values (e.g. BMI, Age).
"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(f"PRAGMA table_info({TABLE_NAME})")
    columns = [row[1] for row in cursor.fetchall()]

    names_in_column = [c for c in columns if c != "Diabetes_binary"]

    if feature not in names_in_column:
        conn.close()
        return "not valid feature, does not exist."

    valid_tests = ["chi_square", "mann_whitney"]

    if test_name not in valid_tests:
        conn.close()
        return "test is not valid, does not exist in avaliable tools"

    query = f"SELECT {feature}, Diabetes_binary FROM {TABLE_NAME}"
    data = pd.read_sql(query, conn)

    if test_name == "chi_square":
        #chi-square test compares two categorical variables (the feature and diabetes status) to see if they're related, using counts of how often each combination occurs
        contingency_table = pd.crosstab(data[feature], data["Diabetes_binary"])
        chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
        result = {"statistic": float(chi2), "p_value": float(p)}
        conn.close()
        return json.dumps(result)
    elif test_name == "mann_whitney":
        #mann-whitney u test: compares the feature's values between the diabetes group and non-diabetes group, without assuming the data is normally distributed
        group0 = data[data["Diabetes_binary"] == 0][feature]
        group1 = data[data["Diabetes_binary"] == 1][feature]
        statistic, p = stats.mannwhitneyu(group0, group1)
        result = {"statistic": float(statistic), "p_value": float(p)}
        conn.close()
        return json.dumps(result)