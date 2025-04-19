''' app/ui/widgets/menubar.py '''
from PyQt6.QtWidgets import QMenuBar, QFileDialog
from ..actions.actions import appmethods

class MenuBar(QMenuBar, appmethods):
    """
    Initialize the menu bar.

    Args:
        parent: The parent widget.
    """

    def __init__(self, parent=None) -> None:

        super().__init__(parent)
        self.file_menu = self.addMenu("File")
        self.edit_menu = self.addMenu("Edit")
        self.view_menu = self.addMenu("View")
        self.help_menu = self.addMenu("Help")

        open_dataset_action = self.file_menu.addAction("Cargar dataset")
        open_dataset_action.triggered.connect(self.open_file)

        # Add actions to the menus
        self.file_menu.addAction(self.parent().topbar.actions_call["Open"]) # type: ignore
        self.file_menu.addAction(self.parent().topbar.actions_call["Save"]) # type: ignore
        self.file_menu.addAction(self.parent().topbar.actions_call["Exit"]) # type: ignore