# app/ui/actions/actions.py
import os
from PyQt6.QtWidgets import QFileDialog, QMessageBox

class appmethods:

    def open_file(self):
        folder_path = QFileDialog.getExistingDirectory(
            self, "Seleccionar carpeta", os.getcwd())
        if folder_path:
            self.treeview.set_path(folder_path)
            self.classifier_widget.setEnabled(True)
            self.classifier_widget.setFolderPath(folder_path)
            self.statusBar().showMessage(f"Carpeta abierta: {folder_path}")

    def save_file(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Guardar archivo", "", "Todos los archivos (*)")
        if file_path:
            try:
                content = self.editbox.toPlainText()
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                self.statusBar().showMessage(f"Archivo guardado en: {file_path}")
            except Exception as e:
                self.show_message("Error", f"No se pudo guardar el archivo:\n{e}", QMessageBox.Icon.Critical)

    def exit_app(self):
        self.close()
