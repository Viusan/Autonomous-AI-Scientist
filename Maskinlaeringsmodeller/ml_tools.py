import os
import time
import sqlite3
import importlib
import inspect
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from ml_bibliotek import MODELLER, TILLATTE_TABELLER, tilgjengelige_modeller

DB = os.environ.get("ML_DB", "diabetes.db")
_cache = {}  # tabell -> (X_train, X_test, y_train, y_test, info), så vi ikke laster dataen på nytt hver gang


#tools som programmet kan kalle. 
def sjekk_tabell(tabell): # Stopper alt hvis tabellnavnet ikke står på listen over tillate tabeller. Agenten kan ikke be om en tilfeldig tabell.
    if tabell not in TILLATTE_TABELLER:
        raise ValueError(f"Unknown table '{tabell}'. Allowed: {TILLATTE_TABELLER}")


def hent_tabell(tabell): # Leser hele tabellen fra databasen inn en pandas-tabell.
    sjekk_tabell(tabell)
    conn = sqlite3.connect(DB)
    df = pd.read_sql(f"SELECT * FROM {tabell}", conn)
    conn.close()
    return df


def tool_list_tables(): # Viser tabellen som finnes med antall rader.
    conn = sqlite3.connect(DB)
    ut = []
    for t in TILLATTE_TABELLER:
        try:
            n = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            ut.append({"table": t, "rows": n})
        except Exception:
            pass  # tabellen finnes ikke i databasen
    conn.close()
    return ut


def tool_describe_table(table): # viser for hver kolonne minste, største, snitt og antall manglende verdier.
    df = hent_tabell(table)
    kolonner = {}
    for c in df.columns:
        kolonner[c] = {"min": round(float(df[c].min()), 3), "max": round(float(df[c].max()), 3),
                       "mean": round(float(df[c].mean()), 3), "missing": int(df[c].isna().sum())}
    return {"table": table, "rows": len(df), "target": "Diabetes_binary", "columns": kolonner}


def tool_list_models():
    return tilgjengelige_modeller()


def last_og_del(tabell): #Gjør tabellen klar til trening og testing.
    # Samme data og samme deling hver gang: fjern tomme rader, 80/20, random_state=42
    if tabell in _cache:
        return _cache[tabell]
    df = hent_tabell(tabell)
    rader_for = len(df)
    df = df.dropna()
    X = df.drop(columns=["Diabetes_binary"])
    y = df["Diabetes_binary"].astype(int)
    info = {"rows_total": rader_for, "rows_used": len(df), "rows_dropped_missing": rader_for - len(df),
            "baseline_accuracy": round(float(max(y.mean(), 1 - y.mean())), 3)}
    _cache[tabell] = (*train_test_split(X, y, test_size=0.2, random_state=42), info)
    return _cache[tabell]


def tool_train_model(table, model, params=None):
    if model not in MODELLER: # Sjekker at modellen finnes.
        raise ValueError(f"Unknown model '{model}'. Use list_models.")
    m = MODELLER[model]
    if m.get("ikke_klassifikator"):
        raise ValueError(f"{model} is an anomaly detector, not a classifier.")

    X_train, X_test, y_train, y_test, info = last_og_del(table)

    # Finn klassen ut fra stien i ml_biblotek.
    modul, klassenavn = m["klasse"].rsplit(".", 1)
    klasse = getattr(importlib.import_module(modul), klassenavn)

    innstillinger = {**m.get("standard", {}), **(params or {})}
    if "n_jobs" in inspect.signature(klasse.__init__).parameters and model != "Logistic Regression":
        innstillinger["n_jobs"] = 2  # maskinen er delt
    estimator = klasse(**innstillinger)
    modell = make_pipeline(StandardScaler(), estimator) if m.get("skala") else estimator

    # Trege modeller trenes på et utvalg
    if m.get("treg"):
        X_fit, y_fit = X_train.iloc[:20000], y_train.iloc[:20000]
    else:
        X_fit, y_fit = X_train, y_train

    start = time.time()
    modell.fit(X_fit, y_fit)
    pred = modell.predict(X_test)
    if hasattr(modell, "predict_proba"):
        score = modell.predict_proba(X_test)[:, 1]
    elif hasattr(modell, "decision_function"):
        score = modell.decision_function(X_test)
    else:
        score = None

    # Hent feature importance der modellen støtter det
    faktisk_modell = modell[-1] if m.get("skala") else modell  # siste steg i pipeline, hvis scaling ble brukt
    feature_importance = None
    if hasattr(faktisk_modell, "feature_importances_"):
        verdier = faktisk_modell.feature_importances_
        feature_importance = dict(sorted(zip(X_train.columns, verdier.tolist()), key=lambda x: -x[1]))
    elif hasattr(faktisk_modell, "coef_"):
        verdier = faktisk_modell.coef_[0]
        feature_importance = dict(sorted(zip(X_train.columns, verdier.tolist()), key=lambda x: -abs(x[1])))

    resultat = {
        "model": model, "table": table, "params": innstillinger,
        "trained_on_rows": len(X_fit), "test_rows": len(X_test),
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "precision": round(float(precision_score(y_test, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, pred)), 4),
        "f1": round(float(f1_score(y_test, pred)), 4),
        "auc": round(float(roc_auc_score(y_test, score)), 4) if score is not None else None,
        "baseline_accuracy": info["baseline_accuracy"], "rows_dropped_missing": info["rows_dropped_missing"],
        "seconds": round(time.time() - start, 1),
        "feature_importance": feature_importance,
    }
    return resultat