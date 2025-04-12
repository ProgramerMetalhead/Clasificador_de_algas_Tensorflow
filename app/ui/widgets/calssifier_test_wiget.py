# classifier_test_widget.py

import os
from PyQt6.QtWidgets import (
    QWidget, QComboBox, QSpinBox, QDoubleSpinBox, 
    QLineEdit, QFormLayout, QHBoxLayout, QVBoxLayout,
    QPushButton, QLabel, QComboBox as QParamCombo
)
from PyQt6.QtCore import pyqtSignal, Qt
from ...utils.params import CLASSIFIER_PARAMS

class ClassifierTestWidget(QWidget):
    """
    Widget que agrupa la lógica de:
     - Elegir clasificador
     - Ajustar test_size
     - Configurar parámetros
     - Botón 'Iniciar clasificación'

    Emite la señal `startClassification(...)` cuando 
    el usuario hace clic en el botón.
    """
    startClassification = pyqtSignal(str, str, float, dict)
    # Parámetros de la señal:
    # 1) folder_path: str
    # 2) classifier_choice: str
    # 3) test_size: float
    # 4) classifier_params: dict

    def __init__(self, classifier_params_dict, parent=None):
        """
        :param classifier_params_dict: Diccionario con la configuración
                                       de parámetros por clasificador, 
                                       similar a 'CLASSIFIER_PARAMS'
        """
        super().__init__(parent)
        self.folder_path = ""
        self.classifier_params_dict = classifier_params_dict
        self.param_widgets = {}

        self.initUI()

    def initUI(self):

        # Combo de clasificadores
        self.combo_classifier = QComboBox(self)
        self.combo_classifier.addItems(list(self.classifier_params_dict.keys()))
        self.combo_classifier.currentTextChanged.connect(self.on_classifier_changed)

        # SpinBox para test_size
        self.spin_test_size = QDoubleSpinBox(self)
        self.spin_test_size.setRange(0.1, 0.9)
        self.spin_test_size.setSingleStep(0.05)
        self.spin_test_size.setValue(0.2)

        # Layout para parámetros dinámicos
        self.params_layout = QFormLayout()
        self.params_layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        # Botón "Iniciar"
        self.btn_start = QPushButton("Iniciar clasificación", self)
        self.btn_start.clicked.connect(self.on_start_clicked)

        # Construcción layout principal
        layout_main = QVBoxLayout(self)

        row_top = QHBoxLayout()
        row_top.addWidget(QLabel("Clasificador:"))
        row_top.addWidget(self.combo_classifier)
        row_top.addWidget(QLabel("Test size:"))
        row_top.addWidget(self.spin_test_size)
        layout_main.addLayout(row_top)

        # Sección de parámetros
        params_container = QWidget()
        params_container.setLayout(self.params_layout)
        layout_main.addWidget(params_container)

        layout_main.addWidget(self.btn_start)

        self.setLayout(layout_main)

        # Construir el formulario de parámetros para el primer clasificador
        default_clf = self.combo_classifier.currentText()
        self.build_params_form(default_clf)

    def on_classifier_changed(self, new_classifier):
        self.build_params_form(new_classifier)

    def build_params_form(self, classifier_name):
        """
        Construye dinámicamente los widgets de parámetros para 
        'classifier_name' según self.classifier_params_dict.
        """
        # Limpiar layout anterior
        while self.params_layout.rowCount() > 0:
            self.params_layout.removeRow(0)

        self.param_widgets.clear()

        # Obtener configuración del diccionario
        params_config = self.classifier_params_dict.get(classifier_name, {})

        for param_name, config in params_config.items():
            ptype = config.get("type", "text")
            default_value = config.get("default", None)
            pmin = config.get("min", None)
            pmax = config.get("max", None)

            if ptype == "combo":
                combo = QParamCombo()
                values = config.get("values", [])
                combo.addItems([str(v) for v in values])
                if str(default_value) in values:
                    combo.setCurrentText(str(default_value))
                self.params_layout.addRow(QLabel(param_name), combo)
                self.param_widgets[param_name] = combo

            elif ptype == "int":
                spin = QSpinBox()
                spin.setRange(-999999, 999999)
                if pmin is not None:
                    spin.setMinimum(int(pmin))
                if pmax is not None:
                    spin.setMaximum(int(pmax))
                if isinstance(default_value, int):
                    spin.setValue(default_value)
                elif default_value is None:
                    spin.setValue(0)
                self.params_layout.addRow(QLabel(param_name), spin)
                self.param_widgets[param_name] = spin

            elif ptype == "float":
                dspin = QDoubleSpinBox()
                dspin.setRange(-1e9, 1e9)
                dspin.setDecimals(6)
                if pmin is not None:
                    dspin.setMinimum(float(pmin))
                if pmax is not None:
                    dspin.setMaximum(float(pmax))
                if isinstance(default_value, (int, float)):
                    dspin.setValue(float(default_value))
                else:
                    dspin.setValue(0.0)
                self.params_layout.addRow(QLabel(param_name), dspin)
                self.param_widgets[param_name] = dspin

            else:
                line = QLineEdit()
                if default_value is not None:
                    line.setText(str(default_value))
                self.params_layout.addRow(QLabel(param_name), line)
                self.param_widgets[param_name] = line

    def on_start_clicked(self):
        """
        Emitimos la señal startClassification con:
         - La carpeta (folder_path)
         - El clasificador elegido
         - El test_size
         - Un dict con los parámetros
        """
        if not self.folder_path or not os.path.isdir(self.folder_path):
            # Podrías mostrar un mensaje de error, o 
            # emitir la señal con carpeta vacía, etc.
            return

        chosen_classifier = self.combo_classifier.currentText()
        test_size = self.spin_test_size.value()

        # Recolectar los valores de parámetros
        classifier_params = {}
        from PyQt6.QtWidgets import QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox as QParamCombo
        for param_name, widget in self.param_widgets.items():
            if isinstance(widget, QSpinBox):
                classifier_params[param_name] = widget.value()
            elif isinstance(widget, QDoubleSpinBox):
                classifier_params[param_name] = widget.value()
            elif isinstance(widget, QParamCombo):
                val = widget.currentText()
                classifier_params[param_name] = val
            elif isinstance(widget, QLineEdit):
                text_val = widget.text().strip()
                if text_val.lower() == "none":
                    classifier_params[param_name] = None
                else:
                    try:
                        parsed = float(text_val)
                        if parsed.is_integer():
                            parsed = int(parsed)
                        classifier_params[param_name] = parsed
                    except:
                        classifier_params[param_name] = text_val

        # Emitir señal
        self.startClassification.emit(
            self.folder_path,
            chosen_classifier,
            test_size,
            classifier_params
        )

    def setFolderPath(self, folder):
        """
        Método para configurar externamente la carpeta donde
        se buscarán las imágenes.
        """
        self.folder_path = folder
