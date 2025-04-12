# Diccionario con la configuración de parámetros
CLASSIFIER_PARAMS = {
    "Naive Bayes": {
        "var_smoothing": {
            "type": "float",
            "default": 1e-9
        }
    },
    "SVM": {
        "kernel": {
            "type": "combo",
            "values": ["linear", "poly", "rbf", "sigmoid"],
            "default": "linear"
        },
        "random_state": {
            "type": "int",
            "default": 42
        },
        "C": {
            "type": "float",
            "default": 1.0
        }
    },
    "KNN": {
        "n_neighbors": {
            "type": "int",
            "default": 5
        }
    },
    "Árbol de Decisión": {
        "random_state": {
            "type": "int",
            "default": 42
        },
        "max_depth": {
            "type": "int",
            "default": None
        }
    },
    "Random Forest": {
        "random_state": {
            "type": "int",
            "default": 42
        },
        "n_estimators": {
            "type": "int",
            "default": 100
        }
    },
    "MLP (Red Neuronal)": {
        "random_state": {
            "type": "int",
            "default": 42
        },
        "max_iter": {
            "type": "int",
            "default": 300
        }
    },
    "QDA": {
        "reg_param": {
            "type": "float",
            "default": 0.0
        }
    },
    "Gaussian Process": {
        "random_state": {
            "type": "int",
            "default": None
        }
    },
    "AdaBoost": {
        "random_state": {
            "type": "int",
            "default": 42
        },
        "n_estimators": {
            "type": "int",
            "default": 50
        }
    }
}