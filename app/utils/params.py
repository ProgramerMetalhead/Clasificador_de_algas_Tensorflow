# app/utils/params.py
# Parámetros por clasificador — combinación de clasificadores iniciales y nuevos sin duplicar

CLASSIFIER_PARAMS = {
    "Naive Bayes": {
        "var_smoothing": {"type": "float", "default": 1e-9},
    },
    "SVM": {
        "kernel": {"type": "combo", "values": ["linear", "poly", "rbf", "sigmoid"], "default": "linear"},
        "C": {"type": "float", "default": 1.0},
        "random_state": {"type": "int", "default": 42},
    },
    "KNN": {
        "n_neighbors": {"type": "int", "default": 5},
    },
    "Árbol de Decisión": {
        "random_state": {"type": "int", "default": 42},
        "max_depth": {"type": "int", "default": None},
    },
    "Random Forest": {
        "random_state": {"type": "int", "default": 42},
        "n_estimators": {"type": "int", "default": 100},
    },
    "MLP (Red Neuronal)": {
        "random_state": {"type": "int", "default": 42},
        "max_iter": {"type": "int", "default": 300},
    },
    "QDA": {
        "reg_param": {"type": "float", "default": 0.0},
    },
    "Gaussian Process": {
        "random_state": {"type": "int", "default": None},
    },
    "AdaBoost": {  # se mantiene como estaba, SAMME no se duplica
        "random_state": {"type": "int", "default": 42},
        "n_estimators": {"type": "int", "default": 50},
    },

    # ---- nuevos (no duplicados) ----
    "Logistic Regression": {
        "max_iter": {"type": "int", "default": 500},
    },
    "SGD Classifier": {
        "max_iter": {"type": "int", "default": 1000},
        "random_state": {"type": "int", "default": 42},
    },
    "Perceptron": {
        "max_iter": {"type": "int", "default": 1000},
    },
    "Linear SVC": {
        "C": {"type": "float", "default": 1.0},
    },
    "Extra Trees": {
        "n_estimators": {"type": "int", "default": 200},
        "random_state": {"type": "int", "default": 42},
    },
    "Gradient Boosting": {
        "n_estimators": {"type": "int", "default": 100},
    },
    "Histogram GB": {
        "max_depth": {"type": "int", "default": None},
    },
    "Bagging": {
        "n_estimators": {"type": "int", "default": 10},
    },
    "BernoulliNB": {
        "alpha": {"type": "float", "default": 1.0},
    },
    "Linear Discriminant": {
        "solver": {"type": "combo", "values": ["svd", "lsqr", "eigen"], "default": "svd"},
    },
    "Passive Aggressive": {
        "max_iter": {"type": "int", "default": 1000},
        "random_state": {"type": "int", "default": 42},
    },

    # ---- alias para CNN (mostrar pero sin params) ----
    "CNN‑Simple": {},
    "CNN‑VGGMini": {},
    "CNN‑ResMini": {},
    "CNN‑MobileMini": {},
}
