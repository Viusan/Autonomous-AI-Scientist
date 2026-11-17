"""Oversight layer for the AI Scientist pipeline.

The agents make every decision (columns, test, code). This module never decides anything:
it only labels the agent's results with descriptive group numbers and checks the agent's
statistic against a reference computation.

Set the environment variable OVERSIGHT to choose the level:
  off      - nothing is added, the flow is exactly as without this module
  log      - group numbers and verification are added and logged, but never trigger a retry
  enforce  - like log, and a wrong statistic is marked as an error so modelling_agent retries
"""
import os
import math
import pandas as pd
from scipy import stats

OUTCOME = "Diabetes_binary"
OVERSIGHT = os.getenv("OVERSIGHT", "enforce").lower()


def _label(value):
    #1.0 -> "1", so labels read like the codebook
    value = float(value)
    return str(int(value)) if value.is_integer() else str(value)


def describe_groups(df, feature):
    #descriptive numbers, computed by code for EVERY feature no matter which test the agent picked.
    #this is not a methodological choice, just labelled facts, so it doesn't reduce autonomy
    data = df[[feature, OUTCOME]].dropna() #this just means select from dataframe, the feature and then what dibaetes binary is, and dropna dropps every row with NaN val
    #seperate who has diabetes and not with feature
    no_diabetes = data.loc[data[OUTCOME] == 0, feature]
    diabetes = data.loc[data[OUTCOME] == 1, feature]

    summary = {
        #summary of how many people comparison is based on
        "n": len(data),
        "n_missing_dropped": int(df[feature].isna().sum()), #how many dropped
        "median": {"no_diabetes": float(no_diabetes.median()), "diabetes": float(diabetes.median())}, #mediam valie of those with and without diabetes
        "mean": {"no_diabetes": round(float(no_diabetes.mean()), 3), "diabetes": round(float(diabetes.mean()), 3)},
    }

    if data[feature].nunique() <= 13:  #binary and ordinal codes: prevalence per value
        prevalence = data.groupby(feature)[OUTCOME].mean() * 100 #splits feature into groups so of u have highBP or not for example
        summary["diabetes_prevalence_pct"] = {f"{feature}={_label(v)}": round(float(p), 2) for v, p in prevalence.items()}#will show the split features in dict
        if set(prevalence.index) == {0, 1}:
            word = "HIGHER" if prevalence[1] > prevalence[0] else "LOWER"
            #this is the line that actually tells what the comparison means, so llm doesnt need to work itself out. 
            summary["direction"] = f"diabetes prevalence is {word} when {feature}=1 than when {feature}=0"

    if "direction" not in summary:
        word = "HIGHER" if diabetes.mean() > no_diabetes.mean() else "LOWER"
        summary["direction"] = f"mean {feature} is {word} in the diabetes group"
    return summary


def verify_result(df, feature, method, statistic):
    #independent check of the agent's number. it only recomputes, it never decides anything.
    #both legitimate variants are accepted (Yates on/off, U for either group),
    #so a mismatch means the generated code is actually wrong, not just a different valid choice
    #this doesnt rewrite but it just checks against what was computed.
    data = df[[feature, OUTCOME]].dropna()
    m = method.lower()

    if "chi" in m:
        table = pd.crosstab(data[feature], data[OUTCOME])
        references = {
            "yates_correction=True": stats.chi2_contingency(table, correction=True)[0],
            "yates_correction=False": stats.chi2_contingency(table, correction=False)[0],
        }
    elif "mann" in m or "whitney" in m:
        group0 = data.loc[data[OUTCOME] == 0, feature]
        group1 = data.loc[data[OUTCOME] == 1, feature]
        u = stats.mannwhitneyu(group0, group1).statistic
        references = {"U_for_no_diabetes_group": u, "U_for_diabetes_group": len(group0) * len(group1) - u}
    else:
        return {"status": "unverified", "reason": f"no reference implementation for '{method}'"}

    for variant, reference in references.items():
        if math.isclose(statistic, float(reference), rel_tol=1e-4):
            return {"status": "verified", "matched_variant": variant}
    return {"status": "mismatch", "reference_statistics": {k: float(v) for k, v in references.items()}}


def add_oversight(calculated, df, feature, method):
    """Add labelled group numbers and a verification to one test result from modelling_agent."""
    if OVERSIGHT == "off":
        return calculated

    calculated["method"] = method
    calculated["oversight_level"] = OVERSIGHT
    calculated["groups"] = describe_groups(df, feature)
    calculated["verification"] = verify_result(df, feature, method, calculated["statistic"])

    #only in enforce mode does a wrong number count as an error,
    #so the existing retry loop in route_after_modelling sends it back to the agent
    if OVERSIGHT == "enforce" and calculated["verification"]["status"] == "mismatch":
        calculated["error"] = (
            f"statistic {calculated['statistic']} does not match a reference computation: "
            f"{calculated['verification']['reference_statistics']}"
        )
    return calculated