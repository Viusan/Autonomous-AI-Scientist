"""Tables and facts for the scientific report.

The idea: the LLM is bad at copying and comparing many numbers, so code does that part.
- Before the LLM writes, it gets two lists of facts: the test settings and the model rankings.
- The LLM writes {{STATS_TABLE}} and {{MODEL_TABLE}} instead of writing tables itself.
- After the LLM writes, render_report() puts the real tables where those placeholders are.
"""

#how the ML part splits the data and predicts (see Maskinlaeringsmodeller/ml_tools.py).
#update this text if the ML code changes
ML_SETUP = ("80/20 train/test split (not stratified, random_state=42), rows with missing values dropped, "
            "default decision threshold of 0.5, no class-imbalance handling")

#the model metrics we show, as (key in the ML log, column name in the table)
METRICS = [
    ("accuracy", "Accuracy"),
    ("auc", "AUC"),
    ("precision", "Precision"),
    ("recall", "Recall"),
    ("f1", "F1"),
]


# ---------------------------------------------------------------------------
# Small helpers for formatting numbers
# ---------------------------------------------------------------------------

def format_p(p):
    #one p-value format for the whole report
    if p == 0:
        #0.0 means the number was too small for the computer to store (underflow), not exactly zero
        return "< 1e-300"
    if p < 0.001:
        return f"{p:.2e}"  #scientific notation, e.g. 8.64e-206
    return f"{p:.3f}"      #normal decimals, e.g. 0.042


def format_statistic(value):
    if abs(value) >= 1_000_000:
        return f"{value:.4e}"  #very big numbers (Mann-Whitney U) in scientific notation
    return f"{value:.2f}"


def get_settings_text(result):
    #which test settings were used, as checked by verify_result() in oversight.py
    verification = result.get("verification", {})

    if verification.get("status") == "verified":
        variant = verification["matched_variant"]
        if variant == "yates_not_applicable":
            return "Yates not applicable, table larger than 2x2 (verified)"
        return variant + " (verified)"

    #not checked (for example a t-test): show what the agent said it used
    if result.get("settings"):
        return result["settings"] + " (not verified)"

    return "not reported"


def get_effect_size_text(result):
    effect = result.get("groups", {}).get("effect_size", {})
    if "cramers_v" in effect:
        return "Cramér's V = " + str(effect["cramers_v"])
    if "rank_biserial_r" in effect:
        return "rank-biserial r = " + str(effect["rank_biserial_r"])
    return "-"


def get_direction_text(result):
    groups = result.get("groups", {})
    #ordinal variables have a "trend" (the shape across categories), use that if it exists
    if "trend" in groups:
        return groups["trend"]
    if "direction" in groups:
        return groups["direction"]
    return "-"


def get_tested_features(calculated_data):
    #only features whose test worked. names starting with "_" are internal error entries
    features = []
    for feature, result in calculated_data.items():
        if feature.startswith("_"):
            continue
        if "error" in result:
            continue
        features.append(feature)
    return features


# ---------------------------------------------------------------------------
# Facts that go INTO the prompt, before the LLM writes
# ---------------------------------------------------------------------------

def settings_summary(calculated_data):
    #one line per feature, so Methods can state the real settings instead of "all tests used X"
    lines = []
    for feature in get_tested_features(calculated_data):
        result = calculated_data[feature]
        method = result.get("method", "unknown test")
        lines.append(f"- {feature}: {method}, settings: {get_settings_text(result)}")
    return "\n".join(lines)


def direction_summary(calculated_data):
    #one line per tested feature with the direction or trend sentence computed by code,
    #so the Discussion can't "correct" a direction to what the LLM expects
    lines = []
    for feature in get_tested_features(calculated_data):
        groups = calculated_data[feature].get("groups", {})
        if "trend" in groups:
            lines.append(f"- {feature}: {groups['trend']}")
        elif "direction" in groups:
            lines.append(f"- {feature}: {groups['direction']}")
    return "\n".join(lines)


def get_models_with_metrics(ml_data):
    #skip models that failed to train (they have no accuracy)
    models = {}
    for name, result in ml_data["trained"].items():
        if result.get("accuracy") is not None:
            models[name] = result
    return models


def model_facts(ml_data):
    #for every metric: which model is highest, which is lowest, and where the selected model ranks.
    #then the LLM never has to compare numbers itself to write "best" or "highest"
    if not ml_data:
        return "No predictive modelling results."

    models = get_models_with_metrics(ml_data)
    selected = ml_data["report"].get("best_model")
    reasoning = ml_data["report"].get("reasoning", "")

    lines = [f"Selected model (chosen by the ML agent): {selected}. Its reasoning: {reasoning}"]

    for key, label in METRICS:
        #collect this metric for every model that has it
        scores = {}
        for name, result in models.items():
            if result.get(key) is not None:
                scores[name] = result[key]
        if not scores:
            continue

        highest_value = max(scores.values())
        lowest_value = min(scores.values())

        #several models can share the highest or lowest value
        highest_models = []
        lowest_models = []
        for name, value in scores.items():
            if value == highest_value:
                highest_models.append(name)
            if value == lowest_value:
                lowest_models.append(name)

        #rank of the selected model: 1 + how many models are strictly better
        rank = 1
        selected_value = scores.get(selected)
        if selected_value is not None:
            for value in scores.values():
                if value > selected_value:
                    rank = rank + 1

        lines.append(
            f"- {label}: highest = {', '.join(highest_models)} ({highest_value}); "
            f"lowest = {', '.join(lowest_models)} ({lowest_value}); "
            f"{selected} ranks {rank} of {len(scores)}"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tables that replace the placeholders, AFTER the LLM writes
# ---------------------------------------------------------------------------

def stats_table(calculated_data):
    #markdown table: a header row, a separator row, then one row per feature
    rows = [
        "| Feature | Test | Settings | Statistic | p-value | Effect size | Direction |",
        "|---|---|---|---|---|---|---|",
    ]

    for feature in get_tested_features(calculated_data):
        result = calculated_data[feature]
        row = (
            f"| {feature} "
            f"| {result.get('method', 'unknown test')} "
            f"| {get_settings_text(result)} "
            f"| {format_statistic(result['statistic'])} "
            f"| {format_p(result['p_value'])} "
            f"| {get_effect_size_text(result)} "
            f"| {get_direction_text(result)} |"
        )
        rows.append(row)

    return "\n".join(rows)


def get_importance_type(model_name):
    #what kind of feature importance each model type gives
    name = model_name.lower()
    if "xgboost" in name or "lightgbm" in name or "catboost" in name:
        return "built-in importance of the boosting library (XGBoost default: gain)"
    if "logistic" in name or "ridge" in name or "linear" in name:
        return "coefficient on standardized features (sign shows direction)"
    return "mean decrease in impurity"


def model_table(ml_data, calculated_data):
    if not ml_data:
        return "Predictive modelling results were unavailable."

    models = get_models_with_metrics(ml_data)
    selected = ml_data["report"].get("best_model")

    # step 1: find the highest value in each column, so only that one gets bold
    best = {}
    for key, label in METRICS:
        best[key] = None
        for result in models.values():
            value = result.get(key)
            if value is None:
                continue
            if best[key] is None or value > best[key]:
                best[key] = value

    # step 2: the header row and separator row
    header = "| Model |"
    separator = "|---|"
    for key, label in METRICS:
        header = header + f" {label} |"
        separator = separator + "---|"
    rows = [header, separator]

    # step 3: one row per model
    for name, result in models.items():
        if name == selected:
            row = f"| {name} (selected) |"
        else:
            row = f"| {name} |"

        for key, label in METRICS:
            value = result.get(key)
            if value is None:
                row = row + " - |"
            elif value == best[key]:
                row = row + f" **{value:.4f}** |"  #bold: highest in this column
            else:
                row = row + f" {value:.4f} |"
        rows.append(row)

    table = "\n".join(rows)

    # step 4: the text below the table
    first_model = list(models.values())[0]  #baseline and test size are the same for every model
    notes = [
        "Bold marks the highest value in each column. Precision, recall and F1 are for the diabetes class.",
        f"Majority-class baseline accuracy (always predicting no diabetes): {first_model['baseline_accuracy']}. "
        f"Test set: {first_model['test_rows']} rows. Setup: {ML_SETUP}.",
    ]

    # step 5: top 10 features of the selected model, and which of them were never tested
    selected_result = models.get(selected)
    if selected_result and selected_result.get("feature_importance"):
        importance = selected_result["feature_importance"]  #already sorted, highest first

        top_features = []
        position = 1
        for feature, value in importance.items():
            if position > 10:
                break
            top_features.append(f"{position}. {feature} ({value:.4f})")
            position = position + 1
        notes.append(f"Top 10 features of {selected} ({get_importance_type(selected)}): " + ", ".join(top_features) + ".")

        tested = get_tested_features(calculated_data)
        not_tested = []
        for feature in importance:
            if feature not in tested:
                not_tested.append(feature)
        if not_tested:
            not_tested.sort()
            notes.append("Features used by the models but not included in the univariate tests: " + ", ".join(not_tested) + ".")

    return table + "\n\n" + "\n".join(notes)


# ---------------------------------------------------------------------------
# Putting the tables into the report
# ---------------------------------------------------------------------------

def insert_table(text, placeholder, table):
    #normal case: the LLM wrote the placeholder, so we swap it for the table
    if placeholder in text:
        #blank lines around the table, otherwise markdown glues two tables (or text and a table) together
        return text.replace(placeholder, "\n" + table + "\n")

    #the LLM forgot the placeholder: put the table just before the Discussion section
    position = text.find("Discussion")
    if position != -1:
        #go back to the start of that line, so a heading like "**Discussion**" stays whole
        line_start = text.rfind("\n", 0, position) + 1
        return text[:line_start] + table + "\n\n" + text[line_start:]

    #no Discussion heading either: put the table at the end
    return text + "\n\n" + table


def render_report(text, stats_md, model_md):
    text = insert_table(text, "{{STATS_TABLE}}", stats_md)
    text = insert_table(text, "{{MODEL_TABLE}}", model_md)
    return text