import importlib.util

# De 18 modellene fra forskningsartikkelen.
# klasse = hvor modellen finnes i Python
# pakke = hvilken pakke som må være installert
# treg = er modellen treg på ca. 200 000 rader (trenes den på et utvalg)
MODELLER = {
    "AdaBoost": dict(klasse="sklearn.ensemble.AdaBoostClassifier", pakke="sklearn"),
    "Extra Trees": dict(klasse="sklearn.ensemble.ExtraTreesClassifier", pakke="sklearn"),
    "C5.0 (entropy decision tree as stand-in)": dict(klasse="sklearn.tree.DecisionTreeClassifier", pakke="sklearn",
                                                     standard={"criterion": "entropy"}),
    "CatBoost": dict(klasse="catboost.CatBoostClassifier", pakke="catboost",
                     standard={"verbose": 0, "thread_count": 2}),
    "Decision Tree": dict(klasse="sklearn.tree.DecisionTreeClassifier", pakke="sklearn"),
    "Gradient Boosting": dict(klasse="sklearn.ensemble.GradientBoostingClassifier", pakke="sklearn", treg=True),
    "Hist Gradient Boosting": dict(klasse="sklearn.ensemble.HistGradientBoostingClassifier", pakke="sklearn"),
    "Isolation Forest": dict(klasse="sklearn.ensemble.IsolationForest", pakke="sklearn", ikke_klassifikator=True),
    "K-Nearest Neighbors": dict(klasse="sklearn.neighbors.KNeighborsClassifier", pakke="sklearn", skala=True, treg=True),
    "LightGBM": dict(klasse="lightgbm.LGBMClassifier", pakke="lightgbm", standard={"verbose": -1}),
    "Linear Discriminant Analysis": dict(klasse="sklearn.discriminant_analysis.LinearDiscriminantAnalysis", pakke="sklearn"),
    "Logistic Regression": dict(klasse="sklearn.linear_model.LogisticRegression", pakke="sklearn", skala=True,
                                standard={"max_iter": 1000}),
    "Ridge Classifier": dict(klasse="sklearn.linear_model.RidgeClassifier", pakke="sklearn", skala=True),
    "Naive Bayes": dict(klasse="sklearn.naive_bayes.GaussianNB", pakke="sklearn"),
    "Nearest Centroid": dict(klasse="sklearn.neighbors.NearestCentroid", pakke="sklearn", skala=True),
    "Quadratic Discriminant Analysis": dict(klasse="sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis", pakke="sklearn"),
    "Random Forest": dict(klasse="sklearn.ensemble.RandomForestClassifier", pakke="sklearn"),
    "XGBoost": dict(klasse="xgboost.XGBClassifier", pakke="xgboost"),
}

# Bare disse tabellene kan brukes
TILLATTE_TABELLER = ["diabetes_binary", "diabetes_binary_corrupt", "diabetes_binary_bmi",
                     "diabetes_binary_missing", "diabetes_binary_swap"]


def er_installert(pakke):
    # Sjekker om en Python-pakke finnes, uten å importere den.
    return importlib.util.find_spec(pakke) is not None


def tilgjengelige_modeller():
    # Liste over modellene som kan brukes akkurat nå (pakken må være installert)
    ut = []
    for navn, m in MODELLER.items():
        if not er_installert(m["pakke"]):
            continue
        notat = []
        if m.get("ikke_klassifikator"):
            notat.append("anomaly detector, NOT a classifier, cannot be trained here")
        if m.get("treg"):
            notat.append("slow, trained on a sample of 20000 rows")
        ut.append({"model": navn, "note": "; ".join(notat)})
    return ut