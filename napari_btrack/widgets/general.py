from magicgui.application import use_app
from magicgui.types import FileDialogMode


def save_path_dialogue_box():
    """Helper function to open a save configuration file dialog."""

    show_file_dialog = use_app().get_obj("show_file_dialog")
    filename = show_file_dialog(
        mode=FileDialogMode.OPTIONAL_FILE,
        caption="Specify file to save btrack configuration",
        start_path=None,
        filter="*.json",
    )

    return filename


def load_path_dialogue_box():
    """Helper function to open a load configuration file dialog."""

    show_file_dialog = use_app().get_obj("show_file_dialog")
    filename = show_file_dialog(
        mode=FileDialogMode.EXISTING_FILE,
        caption="Choose JSON file containing btrack configuration",
        start_path=None,
        filter="*.json",
    )

    return filename
