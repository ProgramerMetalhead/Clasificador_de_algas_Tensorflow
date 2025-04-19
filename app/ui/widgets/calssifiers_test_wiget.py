# app/ui/widgets/classifier_test_widget.py

import os
from PyQt6.QtWidgets import (
    QWidget, QComboBox, QSpinBox, QDoubleSpinBox, 
    QLineEdit, QFormLayout, QHBoxLayout, QVBoxLayout,
    QPushButton, QLabel
)
from PyQt6.QtCore import pyqtSignal
from ...utils.params import CLASSIFIER_PARAMS


class ClassifierTestWidget(QWidget):
    """
    Widget que agrupa la lógica de:
      - Elegir clasificador (incluye +21 modelos)
      - Ajustar test_size
      - Configurar parámetros dinámicos
      - Botón 'Iniciar clasificación'

    Soporta los clasificadores originales y los 11 nuevos definidos en
    CLASSIFIER_PARAMS. Genera automáticamente los controles correspondientes
    a cada hiperparámetro.

    Señal:
      startClassification(folder_path: str,
                          classifier_choice: str,
                          test_size: float,
                          classifier_params: dict)
    """
    startClassification = pyqtSignal(str, str, float, dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.folder_path = ""
        self.classifier_params_dict = CLASSIFIER_PARAMS
        self.param_widgets = {}
        self._build_ui()

    def _build_ui(self):
        # Selector de clasificador
        self.combo_classifier = QComboBox(self)
        self.combo_classifier.addItems(list(self.classifier_params_dict.keys()))
        self.combo_classifier.currentTextChanged.connect(self._on_classifier_changed)

        # Selector de test_size
        self.spin_test_size = QDoubleSpinBox(self)
        self.spin_test_size.setRange(0.1, 0.9)
        self.spin_test_size.setSingleStep(0.05)
        self.spin_test_size.setValue(0.2)

        # Layout de parámetros dinámicos
        self.params_layout = QFormLayout()
        self.params_layout.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        # Botón de inicio
        self.btn_start = QPushButton("Iniciar clasificación", self)
        self.btn_start.clicked.connect(self._on_start_clicked)

        # Armar UI
        top_layout = QHBoxLayout()
        top_layout.addWidget(QLabel("Clasificador:"))
        top_layout.addWidget(self.combo_classifier)
        top_layout.addWidget(QLabel("Test size:"))
        top_layout.addWidget(self.spin_test_size)

        params_container = QWidget(self)
        params_container.setLayout(self.params_layout)

        main_layout = QVBoxLayout(self)
        main_layout.addLayout(top_layout)
        main_layout.addWidget(params_container)
        main_layout.addWidget(self.btn_start)
        self.setLayout(main_layout)

        # Construir parámetros del primer clasificador
        self._build_params_form(self.combo_classifier.currentText())

    def _on_classifier_changed(self, classifier_name: str):
        self._build_params_form(classifier_name)

    def _build_params_form(self, classifier_name: str):
        # Limpiar
        while self.params_layout.rowCount() > 0:
            self.params_layout.removeRow(0)
        self.param_widgets.clear()

        config = self.classifier_params_dict.get(classifier_name, {})
        for name, opts in config.items():
            ptype = opts.get("type", "text")
            default = opts.get("default")
            values = opts.get("values", [])
            pmin = opts.get("min")
            pmax = opts.get("max")

            if ptype == "combo":
                widget = QComboBox()
                widget.addItems([str(v) for v in values])
                if default in values:
                    widget.setCurrentText(str(default))
            elif ptype == "int":
                widget = QSpinBox()
                widget.setRange(pmin if pmin is not None else -999999,
                                pmax if pmax is not None else 999999)
                if isinstance(default, int):
                    widget.setValue(default)
            elif ptype == "float":
                widget = QDoubleSpinBox()
                widget.setRange(pmin if pmin is not None else -1e9,
                                pmax if pmax is not None else 1e9)
                widget.setDecimals(6)
                if isinstance(default, (int, float)):
                    widget.setValue(default)
            else:
                widget = QLineEdit()
                if default is not None:
                    widget.setText(str(default))

            self.params_layout.addRow(QLabel(name), widget)
            self.param_widgets[name] = widget

    def _on_start_clicked(self):
        if not self.folder_path or not os.path.isdir(self.folder_path):
            # No folder selected
            return
        clf = self.combo_classifier.currentText()
        ts = self.spin_test_size.value()
        params = {}
        for name, widget in self.param_widgets.items():
            if isinstance(widget, QSpinBox) or isinstance(widget, QDoubleSpinBox):
                params[name] = widget.value()
            elif isinstance(widget, QComboBox):
                params[name] = widget.currentText()
            else:
                text = widget.text().strip()
                params[name] = None if text.lower() == 'none' else text
        self.startClassification.emit(self.folder_path, clf, ts, params)

    def setFolderPath(self, folder: str):
        """Actualiza la carpeta de trabajo, habilita el widget."""
        self.folder_path = folder
        self.setEnabled(bool(folder))
