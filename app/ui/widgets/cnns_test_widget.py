# app/ui/widgets/cnn_test_widget.py
"""Widget para probar las 4 arquitecturas CNN.

Permite al usuario seleccionar:
  • Arquitectura (Simple, VGGMini, ResMini, MobileMini)
  • Tamaño de entrada (target_size) con QSpinBox (mismo valor para alto/ancho)
  • Dimensión de *feature‑vector* extra (por si se fusiona con features clásicos)

Emite una señal `startCnnTraining` con (folder_path, model_name, input_size, feat_size).
"""

from PyQt6.QtWidgets import (
    QWidget, QLabel, QComboBox, QSpinBox, QPushButton, QFormLayout, QVBoxLayout
)
from PyQt6.QtCore import pyqtSignal


class CNNTestWidget(QWidget):
    """Widget dedicado a CNNs — se activa cuando el usuario quiera entrenar una red."""

    startCnnTraining = pyqtSignal(str, str, int, int)
    # folder, model_name, input_size, feature_size

    def __init__(self, parent=None):
        super().__init__(parent)
        self.folder_path: str = ""
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        form = QFormLayout()
        self.combo_model = QComboBox()
        self.combo_model.addItems([
            "CNN‑Simple",
            "CNN‑VGGMini",
            "CNN‑ResMini",
            "CNN‑MobileMini",
        ])

        self.spin_input = QSpinBox();  self.spin_input.setRange(32, 512); self.spin_input.setValue(64)
        self.spin_feat  = QSpinBox();  self.spin_feat.setRange(16, 1024); self.spin_feat.setValue(128)

        form.addRow(QLabel("Modelo CNN:"), self.combo_model)
        form.addRow(QLabel("Input size (px):"), self.spin_input)
        form.addRow(QLabel("Feature size:"), self.spin_feat)
        layout.addLayout(form)

        self.btn_go = QPushButton("Entrenar CNN", self)
        self.btn_go.clicked.connect(self._on_start)
        layout.addWidget(self.btn_go)

        self.setLayout(layout)

    def _on_start(self):
        if not self.folder_path:
            return  # podrías mostrar mensaje de error
        self.startCnnTraining.emit(
            self.folder_path,
            self.combo_model.currentText(),
            self.spin_input.value(),
            self.spin_feat.value(),
        )

    # Se invoca externamente cuando el usuario eligió dataset
    def setFolderPath(self, folder: str):
        self.folder_path = folder
