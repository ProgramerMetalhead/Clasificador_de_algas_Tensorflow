''' app/utils/classifier.py '''

''' app/utils/classifier.py '''

# clasificador.py
from app.utils.processingImage import ProcessImage
from PyQt6.QtCore import QThread, pyqtSignal
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class DatasetWorker(QThread):
    # Señal para imprimir logs en la interfaz
    log_signal = pyqtSignal(str)
    # Señal que avisa que terminó el proceso
    finished = pyqtSignal()

    def __init__(self, folder_path, classifier_choice):
        super().__init__()
        self.folder_path = folder_path
        self.classifier_choice = classifier_choice

    # -------------------- Methodo principla del hilo --------------------

    def run(self):

        try:
        
            all_features, all_labels = ProcessImage(self.folder_path).process

            accuracy = self.train_and_evaluate(all_features, all_labels, self.classifier_choice)

            self.log_signal.emit("---------------------------")
            self.log_signal.emit(f"Precisión final: {accuracy * 100:.2f}%")
            self.log_signal.emit("---------------------------")
            self.finished.emit()

        except Exception as e:
            self.log_signal.emit(f"Error: {str(e)}")


    def train_and_evaluate(self, features, labels, classifier_choice):
        """
        Entrena y evalúa el clasificador elegido, muestra logs con la precisión.
        """
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42
        )

        if classifier_choice == "Naive Bayes":
            model = GaussianNB()
        elif classifier_choice == "SVM":
            model = SVC(kernel='linear', random_state=42)
        elif classifier_choice == "KNN":
            model = KNeighborsClassifier()
        elif classifier_choice == "Árbol de Decisión":
            model = DecisionTreeClassifier(random_state=42)
        elif classifier_choice == "Random Forest":
            model = RandomForestClassifier(random_state=42)
        elif classifier_choice == "MLP (Red Neuronal)":
            model = MLPClassifier(random_state=42, max_iter=300)
        elif classifier_choice == "QDA":
            model = QuadraticDiscriminantAnalysis()
        elif classifier_choice == "Gaussian Process":
            model = GaussianProcessClassifier()
        elif classifier_choice == "AdaBoost":
            model = AdaBoostClassifier(random_state=42)
        else:
            self.log_signal.emit(f"Clasificador no soportado: {classifier_choice}")
            return 0.0

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        self.log_signal.emit(f"Clasificador: {classifier_choice}")
        self.log_signal.emit(f"- Total de muestras: {len(features)}")
        self.log_signal.emit(f"- Muestras de entrenamiento: {len(X_train)}")
        self.log_signal.emit(f"- Muestras de prueba: {len(X_test)}")
        self.log_signal.emit(f"- Precisión (accuracy): {accuracy * 100:.2f}%")
        return accuracy

