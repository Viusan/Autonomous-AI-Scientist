import sqlite3
import pandas as pd

DB_PATH = "diabetes.db"
TABLE_NAME = "diabetes_binary"


def fetch_dataframe(chosen_columns):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({TABLE_NAME})")
    columns = [row[1] for row in cursor.fetchall()]
    names_in_column = [c for c in columns if c != "Diabetes_binary"]

    validated_features = []
    for c in chosen_columns:
        if c in names_in_column:
            validated_features.append(c)

    if validated_features:
        columns_str = ", ".join(validated_features)
    else:
        columns_str = ", ".join(names_in_column)

    query = f"SELECT {columns_str}, Diabetes_binary FROM {TABLE_NAME}"
    data_frame = pd.read_sql(query, conn)
    conn.close()
    return data_frame