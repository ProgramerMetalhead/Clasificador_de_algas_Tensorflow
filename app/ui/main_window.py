# app/ui/main_window.py  (orden corregido para inicializar topbar antes del menú)

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QTextEdit
)
from app.ui.widgets.menubar import MenuBar
from app.ui.widgets.toolbar import ToolBar
from app.ui.widgets.statusbar import StatusBar
from app.ui.widgets.treeview import TreeView
from app.ui.widgets.calssifiers_test_wiget import ClassifierTestWidget
from app.ui.widgets.cnns_test_widget import CNNTestWidget
from app.ui.actions.actions import appmethods
from app.workers.classifiers import DatasetWorker
from app.workers.cnn_worker import CNNClassifierWorker


class MainWindow(QMainWindow, appmethods):
    """Ventana principal: asegura que las toolbars existen antes de crear el menú."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Microalgae Classifier")
        self.setGeometry(100, 100, 1200, 800)

        central = QWidget(self)
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)

        # File explorer
        self.treeview = TreeView(self)
        layout.addWidget(self.treeview)

        # Widgets de clasificación
        self.classifier_widget = ClassifierTestWidget(); self.classifier_widget.setEnabled(False)
        self.cnn_widget = CNNTestWidget(); self.cnn_widget.setEnabled(False)
        layout.addWidget(self.classifier_widget)
        layout.addWidget(self.cnn_widget)

        # Primero crear toolbars → topbar ya existe cuando el menú lo necesita
        self._create_toolbars()
        self.setMenuBar(MenuBar(self))
        self.setStatusBar(StatusBar(self))

        # Señales
        self.classifier_widget.startClassification.connect(self._launch_dataset_worker)
        self.cnn_widget.startCnnTraining.connect(self._launch_cnn_worker)
        self.classifier_widget.combo_classifier.currentIndexChanged.connect(self._activate_classifier_mode)
        self.cnn_widget.combo_model.currentIndexChanged.connect(self._activate_cnn_mode)

    # Toolbars ----------------------------------------------------------------
    def _create_toolbars(self):
        self.topbar = ToolBar(self, Qt.Orientation.Horizontal, Qt.ToolButtonStyle.ToolButtonTextUnderIcon, (24, 24))
        self.topbar.add_button("Open", "resources/assets/icons/windows/imageres-10.ico", self.open_file)
        self.topbar.add_button("Save", "resources/assets/icons/windows/shell32-259.ico", self.save_file)
        self.topbar.add_separator()
        self.topbar.add_button("Exit", "resources/assets/icons/windows/shell32-220.ico", self.exit_app)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, self.topbar)

        self.rightbar = ToolBar(self, Qt.Orientation.Vertical, Qt.ToolButtonStyle.ToolButtonIconOnly, (24, 24))
        self.rightbar.add_separator()
        self.rightbar.add_button("Settings", "resources/assets/icons/windows/shell32-315.ico", self.settings_window)
        self.rightbar.add_button("Privacy", "resources/assets/icons/windows/shell32-167.ico", self.privacy_window)
        self.addToolBar(Qt.ToolBarArea.RightToolBarArea, self.rightbar)

    # Dataset selection -------------------------------------------------------
    def handle_dataset_folder_selected(self, folder_path: str):
        if not folder_path: return
        self.treeview.setRootIndex(self.treeview.file_system_model.index(folder_path))
        self.classifier_widget.setFolderPath(folder_path); self.classifier_widget.setEnabled(True)
        self.cnn_widget.setFolderPath(folder_path);       self.cnn_widget.setEnabled(False)
        self.statusBar().showMessage(f"Dataset cargado: {folder_path}")

    # Activación de modos -----------------------------------------------------
    def _activate_classifier_mode(self):
        self.classifier_widget.setEnabled(True); self.cnn_widget.setEnabled(False)

    def _activate_cnn_mode(self):
        self.cnn_widget.setEnabled(True); self.classifier_widget.setEnabled(False)

    # Lanzadores de hilos ------------------------------------------------------
    def _launch_dataset_worker(self, folder, clf, ts, params):
        self.log_editor.append(f"Entrenando {clf}...")
        worker = DatasetWorker(folder, clf); worker.log_signal.connect(self.log_editor.append)
        worker.finished.connect(lambda: self.statusBar().showMessage("Clásico listo")); worker.start()

    def _launch_cnn_worker(self, folder, model, inp, feat):
        self.log_editor.append(f"CNN {model}...")
        worker = CNNClassifierWorker(folder, model, inp, feat)
        worker.log_signal.connect(self.log_editor.append)
        worker.finished.connect(lambda: self.statusBar().showMessage("CNN listo")); worker.start()

    # Placeholders ------------------------------------------------------------
    def settings_window(self): pass
    def privacy_window(self): pass
