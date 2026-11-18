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

#what each variable means, and which way its scale goes (CDC BRFSS 2015 codebook).
#used to label directions, so "mean GenHlth is HIGHER" can never be read as "better health"
CODEBOOK = {
    "HighBP": "has been told they have high blood pressure (1 = yes)",
    "HighChol": "has been told they have high cholesterol (1 = yes)",
    "CholCheck": "had a cholesterol check in the last 5 years (1 = yes)",
    "BMI": "body mass index (continuous, higher = heavier)",
    "Smoker": "has smoked at least 100 cigarettes in their lifetime (1 = yes), not current smoking",
    "Stroke": "has ever had a stroke (1 = yes)",
    "HeartDiseaseorAttack": "coronary heart disease or heart attack (1 = yes)",
    "PhysActivity": "physical activity in the past 30 days, not counting job (1 = yes)",
    "Fruits": "eats fruit 1+ times per day (1 = yes)",
    "Veggies": "eats vegetables 1+ times per day (1 = yes)",
    "HvyAlcoholConsump": "heavy drinker, men >14 / women >7 drinks per week (1 = yes)",
    "AnyHealthcare": "has any kind of health coverage (1 = yes)",
    "NoDocbcCost": "could not see a doctor in the past 12 months because of cost (1 = yes)",
    "GenHlth": "self-rated general health, 1 = excellent ... 5 = poor (higher = WORSE health)",
    "MentHlth": "days of poor mental health in the past 30 days, 0-30 (higher = worse)",
    "PhysHlth": "days of physical illness or injury in the past 30 days, 0-30 (higher = worse)",
    "DiffWalk": "serious difficulty walking or climbing stairs (1 = yes)",
    "Sex": "0 = female, 1 = male",
    "Age": "13 age categories, 1 = 18-24 ... 13 = 80+ (higher = older), not years",
    "Education": "1 = never attended school ... 6 = college graduate (higher = more education)",
    "Income": "1 = less than $10k ... 8 = $75k or more (higher = higher income)",
}


def get_meaning(feature):
    #the codebook text for a feature, or a fallback if the feature isn't in the codebook
    if feature in CODEBOOK:
        return CODEBOOK[feature]
    return "see codebook"


def make_label(value):
    #1.0 -> "1", so labels read like the codebook
    value = float(value)
    if value.is_integer():
        return str(int(value))
    return str(value)


# ---------------------------------------------------------------------------
# Part 1: labelled group numbers
# ---------------------------------------------------------------------------

def describe_trend(prevalence, feature):
    #for ordinal codes (Age, GenHlth, Income...): describe the shape across the categories,
    #so a peak in the middle can't be reported as "highest in the oldest group"
    categories = []
    percents = []
    for value, percent in prevalence.items():
        categories.append(make_label(value))
        percents.append(float(percent))

    #check every step from one category to the next
    always_up = True
    always_down = True
    for i in range(1, len(percents)):
        if percents[i] <= percents[i - 1]:
            always_up = False
        if percents[i] >= percents[i - 1]:
            always_down = False

    first = f"{percents[0]:.1f}% at {feature}={categories[0]}"
    last = f"{percents[-1]:.1f}% at {feature}={categories[-1]}"

    if always_up:
        return f"diabetes prevalence increases at every step, from {first} to {last}"
    if always_down:
        return f"diabetes prevalence decreases at every step, from {first} to {last}"

    #goes up and down: report where the highest and lowest are
    highest = 0
    lowest = 0
    for i in range(len(percents)):
        if percents[i] > percents[highest]:
            highest = i
        if percents[i] < percents[lowest]:
            lowest = i

    return (f"diabetes prevalence is NOT monotonic: {first}, {last}; "
            f"highest {percents[highest]:.1f}% at {feature}={categories[highest]}, "
            f"lowest {percents[lowest]:.1f}% at {feature}={categories[lowest]}")


def describe_groups(df, feature):
    #descriptive numbers, computed by code for EVERY feature no matter which test the agent picked.
    #this is not a methodological choice, just labelled facts, so it doesn't reduce autonomy
    data = df[[feature, OUTCOME]].dropna()
    no_diabetes = data.loc[data[OUTCOME] == 0, feature]
    diabetes = data.loc[data[OUTCOME] == 1, feature]

    summary = {
        "meaning": get_meaning(feature),
        "n": len(data),
        "n_missing_dropped": int(df[feature].isna().sum()),
        "median": {"no_diabetes": float(no_diabetes.median()), "diabetes": float(diabetes.median())},
        "mean": {"no_diabetes": round(float(no_diabetes.mean()), 3), "diabetes": round(float(diabetes.mean()), 3)},
    }

    number_of_values = data[feature].nunique()

    if number_of_values <= 13:
        #binary and ordinal codes: diabetes prevalence (%) for each value
        prevalence = data.groupby(feature)[OUTCOME].mean() * 100

        summary["diabetes_prevalence_pct"] = {}
        for value, percent in prevalence.items():
            summary["diabetes_prevalence_pct"][f"{feature}={make_label(value)}"] = round(float(percent), 2)

        #effect size: Cramér's V, from 0 (no association) to 1 (perfect association)
        table = pd.crosstab(data[feature], data[OUTCOME])
        chi2 = stats.chi2_contingency(table, correction=False)[0]
        smallest_side = min(table.shape) - 1
        cramers_v = math.sqrt(chi2 / (len(data) * smallest_side))
        summary["effect_size"] = {"cramers_v": round(cramers_v, 4)}

        if number_of_values == 2 and 0 in prevalence.index and 1 in prevalence.index:
            #binary 0/1 feature: write the direction in words
            if prevalence[1] > prevalence[0]:
                word = "HIGHER"
            else:
                word = "LOWER"
            summary["direction"] = f"diabetes prevalence is {word} when {feature}=1 than when {feature}=0"
        else:
            #ordinal feature: describe the shape, and say what the scale means
            summary["trend"] = describe_trend(prevalence, feature) + f" ({get_meaning(feature)})"
    else:
        #continuous feature (BMI, MentHlth, PhysHlth): rank-biserial r as effect size,
        #from -1 to 1, positive means the diabetes group tends to have higher values
        u = stats.mannwhitneyu(no_diabetes, diabetes).statistic
        all_pairs = len(no_diabetes) * len(diabetes)
        rank_biserial = 1 - 2 * u / all_pairs
        summary["effect_size"] = {"rank_biserial_r": round(float(rank_biserial), 4)}

    if "direction" not in summary:
        #everything that isn't binary: direction based on the mean
        if diabetes.mean() > no_diabetes.mean():
            word = "HIGHER"
        else:
            word = "LOWER"
        summary["direction"] = f"mean {feature} is {word} in the diabetes group ({get_meaning(feature)})"

    return summary


# ---------------------------------------------------------------------------
# Part 2: checking the agent's numbers
# ---------------------------------------------------------------------------

def verify_result(df, feature, method, statistic, p_value):
    #independent check of the agent's numbers. it only recomputes, it never decides anything.
    #both legitimate variants are accepted (Yates on/off, U for either group),
    #so a mismatch means the generated code is actually wrong, not just a different valid choice
    data = df[[feature, OUTCOME]].dropna()
    method_lower = method.lower()

    # step 1: compute the correct answers, one per valid variant
    #each reference is {"statistic": ..., "p_value": ...}
    references = {}

    if "chi" in method_lower:
        table = pd.crosstab(data[feature], data[OUTCOME])
        with_yates = stats.chi2_contingency(table, correction=True)
        without_yates = stats.chi2_contingency(table, correction=False)

        if with_yates[0] == without_yates[0]:
            #Yates only changes 2x2 tables, so for bigger tables both are identical
            references["yates_not_applicable"] = {"statistic": with_yates[0], "p_value": with_yates[1]}
        else:
            references["yates_correction=True"] = {"statistic": with_yates[0], "p_value": with_yates[1]}
            references["yates_correction=False"] = {"statistic": without_yates[0], "p_value": without_yates[1]}

    elif "mann" in method_lower or "whitney" in method_lower:
        group0 = data.loc[data[OUTCOME] == 0, feature]
        group1 = data.loc[data[OUTCOME] == 1, feature]
        result = stats.mannwhitneyu(group0, group1)
        #U depends on which group comes first, but the two versions always add up to n0 * n1.
        #the p-value is the same for both
        u_other_group = len(group0) * len(group1) - result.statistic
        references["U_for_no_diabetes_group"] = {"statistic": result.statistic, "p_value": result.pvalue}
        references["U_for_diabetes_group"] = {"statistic": u_other_group, "p_value": result.pvalue}

    else:
        #no reference for this test (t-test, logistic regression...): allowed, just not checked
        return {"status": "unverified", "reason": f"no reference implementation for '{method}'"}

    # step 2: find the variant closest to the agent's statistic
    closest_variant = None
    smallest_difference = None
    for variant, reference in references.items():
        difference = abs(statistic - float(reference["statistic"]))
        if smallest_difference is None or difference < smallest_difference:
            smallest_difference = difference
            closest_variant = variant

    closest = references[closest_variant]

    # step 3: is the statistic close enough? (0.001% difference allowed, for rounding)
    if not math.isclose(statistic, float(closest["statistic"]), rel_tol=1e-5):
        all_statistics = {}
        for variant, reference in references.items():
            all_statistics[variant] = float(reference["statistic"])
        return {"status": "mismatch", "reference_statistics": all_statistics}

    # step 4: is the p-value right too? 0.0 is only fine if the correct value is also 0.0 (underflow)
    if not math.isclose(p_value, float(closest["p_value"]), rel_tol=1e-3, abs_tol=1e-300):
        return {
            "status": "mismatch",
            "matched_variant": closest_variant,
            "reference_statistics": {closest_variant: float(closest["statistic"])},
            "reason": f"statistic is right but p_value {p_value} should be {float(closest['p_value'])}",
        }

    return {"status": "verified", "matched_variant": closest_variant}


# ---------------------------------------------------------------------------
# Part 3: the one function the orchestrator calls
# ---------------------------------------------------------------------------

def add_oversight(calculated, df, feature, method):
    """Add labelled group numbers and a verification to one test result from modelling_agent."""
    if OVERSIGHT == "off":
        return calculated

    calculated["method"] = method
    calculated["oversight_level"] = OVERSIGHT
    calculated["groups"] = describe_groups(df, feature)
    calculated["verification"] = verify_result(df, feature, method, calculated["statistic"], calculated["p_value"])

    #only in enforce mode does a wrong number count as an error,
    #so the existing retry loop in route_after_modelling sends it back to the agent
    if OVERSIGHT == "enforce" and calculated["verification"]["status"] == "mismatch":
        verification = calculated["verification"]
        if "reason" in verification:
            calculated["error"] = verification["reason"]
        else:
            calculated["error"] = (f"statistic {calculated['statistic']} does not match a reference "
                                   f"computation: {verification['reference_statistics']}")
    return calculated