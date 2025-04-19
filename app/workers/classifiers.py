# app/utils/classifier.py  (extendido con 12 clasificadores extra)
"""Entrena y evalúa clasificadores tradicionales.

Ahora soporta 20+ algoritmos, en sincronía con `CLASSIFIER_PARAMS`.
"""

from app.models.processingImage import ProcessImage
from PyQt6.QtCore import QThread, pyqtSignal

# ---- scikit‑learn imports ----
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    BaggingClassifier,
)
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import (
    QuadraticDiscriminantAnalysis,
    LinearDiscriminantAnalysis,
)
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import (
    LogisticRegression,
    SGDClassifier,
    Perceptron,
    PassiveAggressiveClassifier,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class DatasetWorker(QThread):
    """Hilo que procesa imágenes y entrena un clasificador tradicional."""

    log_signal = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, folder_path: str, classifier_choice: str):
        super().__init__()
        self.folder_path = folder_path
        self.classifier_choice = classifier_choice

    # -------------------------------------------------------------
    def run(self):
        try:
            features, labels = ProcessImage(self.folder_path).process()
            acc = self.train_and_evaluate(features, labels, self.classifier_choice)
            self.log_signal.emit("---------------------------")
            self.log_signal.emit(f"Precisión final: {acc * 100:.2f}%")
            self.log_signal.emit("---------------------------")
            self.finished.emit()
        except Exception as exc:
            self.log_signal.emit(f"Error: {exc}")

    # -------------------------------------------------------------
    def train_and_evaluate(self, X, y, choice: str):
        """Devuelve accuracy para el clasificador elegido."""

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Mapping extenso
        model: object
        if choice == "Naive Bayes":
            model = GaussianNB()
        elif choice == "BernoulliNB":
            model = BernoulliNB()
        elif choice == "Logistic Regression":
            model = LogisticRegression(max_iter=500)
        elif choice == "SGD Classifier":
            model = SGDClassifier(max_iter=1000, random_state=42)
        elif choice == "Perceptron":
            model = Perceptron(max_iter=1000)
        elif choice == "Passive Aggressive":
            model = PassiveAggressiveClassifier(max_iter=1000, random_state=42)
        elif choice == "SVM":
            model = SVC(kernel="linear", random_state=42)
        elif choice == "Linear SVC":
            model = LinearSVC(C=1.0)
        elif choice == "KNN":
            model = KNeighborsClassifier()
        elif choice == "Árbol de Decisión":
            model = DecisionTreeClassifier(random_state=42)
        elif choice == "Random Forest":
            model = RandomForestClassifier(random_state=42)
        elif choice == "Extra Trees":
            model = ExtraTreesClassifier(n_estimators=200, random_state=42)
        elif choice == "Gradient Boosting":
            model = GradientBoostingClassifier(n_estimators=100)
        elif choice == "Histogram GB":
            model = HistGradientBoostingClassifier()
        elif choice == "Bagging":
            model = BaggingClassifier(n_estimators=10)
        elif choice == "AdaBoost":
            model = AdaBoostClassifier(random_state=42, n_estimators=50)
        elif choice == "MLP (Red Neuronal)":
            model = MLPClassifier(random_state=42, max_iter=300)
        elif choice == "QDA":
            model = QuadraticDiscriminantAnalysis()
        elif choice == "Linear Discriminant":
            model = LinearDiscriminantAnalysis()
        elif choice == "Gaussian Process":
            model = GaussianProcessClassifier()
        else:
            self.log_signal.emit(f"Clasificador no soportado: {choice}")
            return 0.0

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Logs
        self.log_signal.emit(f"Clasificador: {choice}")
        self.log_signal.emit(f"- Muestras: {len(X)} (train: {len(X_train)}, test: {len(X_test)})")
        self.log_signal.emit(f"- Accuracy: {accuracy * 100:.2f}%")
        return accuracy
