import importlib.util

# De 18 modellene fra forskningsartikkelen (navn, hvor den finnes, pakke)
MODELLER = [
    ("AdaBoost", "sklearn.ensemble.AdaBoostClassifier", "sklearn"),
    ("Extra Trees", "sklearn.ensemble.ExtraTreesClassifier", "sklearn"),
    ("C5.0 (not in Python)", "use sklearn.tree.DecisionTreeClassifier(criterion='entropy') as a stand-in", "sklearn"),
    ("CatBoost", "catboost.CatBoostClassifier", "catboost"),
    ("Decision Tree", "sklearn.tree.DecisionTreeClassifier", "sklearn"),
    ("Gradient Boosting (slow on 200k rows)", "sklearn.ensemble.GradientBoostingClassifier", "sklearn"),
    ("Histogram Gradient Boosting (fast)", "sklearn.ensemble.HistGradientBoostingClassifier", "sklearn"),
    ("Isolation Forest (anomaly detector, NOT a classifier, do not use)", "sklearn.ensemble.IsolationForest", "sklearn"),
    ("K-Nearest Neighbors (very slow on 200k rows)", "sklearn.neighbors.KNeighborsClassifier", "sklearn"),
    ("LightGBM", "lightgbm.LGBMClassifier", "lightgbm"),
    ("Linear Discriminant Analysis", "sklearn.discriminant_analysis.LinearDiscriminantAnalysis", "sklearn"),
    ("Logistic Regression (needs StandardScaler)", "sklearn.linear_model.LogisticRegression", "sklearn"),
    ("Ridge Classifier (no predict_proba, use decision_function)", "sklearn.linear_model.RidgeClassifier", "sklearn"),
    ("Naive Bayes", "sklearn.naive_bayes.GaussianNB", "sklearn"),
    ("Nearest Centroid", "sklearn.neighbors.NearestCentroid", "sklearn"),
    ("Quadratic Discriminant Analysis", "sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis", "sklearn"),
    ("Random Forest", "sklearn.ensemble.RandomForestClassifier", "sklearn"),
    ("XGBoost", "xgboost.XGBClassifier", "xgboost"),
]


TILLATTE_TABELLER = ["diabetes_binary", "diabetes_binary_corrupt", "diabetes_binary_bmi",
                     "diabetes_binary_missing", "diabetes_binary_swap"]


def bibliotek_tekst():
    # Teksten som limes inn i prompten. Bare modeller med installert pakke tas med.
    return "\n".join(f"- {n}: {k}" for n, k, p in MODELLER if importlib.util.find_spec(p) is not None)