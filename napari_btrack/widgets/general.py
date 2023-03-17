from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from napari_btrack.config import UnscaledTackerConfig

import magicgui
import napari
from magicgui.types import FileDialogMode


def save_path_dialogue_box():
    """Helper function to open a save configuration file dialog."""

    app = magicgui.application.use_app()
    show_file_dialog = app.get_obj("show_file_dialog")
    filename = show_file_dialog(
        mode=FileDialogMode.OPTIONAL_FILE,
        caption="Specify file to save btrack configuration",
        start_path=None,
        filter="*.json",
    )

    return filename


def load_path_dialogue_box():
    """Helper function to open a load configuration file dialog."""

    app = magicgui.application.use_app()
    show_file_dialog = app.get_obj("show_file_dialog")
    filename = show_file_dialog(
        mode=FileDialogMode.EXISTING_FILE,
        caption="Choose JSON file containing btrack configuration",
        start_path=None,
        filter="*.json",
    )

    return filename


def create_input_widgets():
    """Create widgets for selecting labels layer and TrackerConfig"""

    tooltip = (
        "Select a 'Labels' layer to use for tracking.\n"
        "To use an 'Image' layer, first convert 'Labels' by right-clicking "
        "on it in the layers list, and clicking on 'Convert to Labels'"
    )
    segmentation_selector = magicgui.widgets.create_widget(
        annotation=napari.layers.Labels,
        name="segmentation_selector",
        label="segmentation",
        options={"tooltip": tooltip},
    )

    tooltip = "Select a loaded configuration.\nNote, this will update values set below."
    config_selector = magicgui.widgets.create_widget(
        value="cell",
        name="config_selector",
        label="base config",
        widget_type="ComboBox",
        options={
            "choices": ["cell", "particle"],
            "tooltip": tooltip,
        },
    )

    input_widgets = [segmentation_selector, config_selector]

    return input_widgets


def create_update_method_widgets(tracker_config: UnscaledTackerConfig):
    """Create widgets for selecting the update method"""

    tooltip = (
        "Select the update method.\n"
        "EXACT: exact calculation of Bayesian belief matrix.\n"
        "APPROXIMATE: approximate the Bayesian belief matrix. Useful for datasets with "
        "more than 1000 particles per frame."
    )
    update_method_selector = magicgui.widgets.create_widget(
        value="EXACT",
        name="update_method_selector",
        label="update method",
        widget_type="ComboBox",
        options={
            "choices": ["EXACT", "APPROXIMATE"],
            "tooltip": tooltip,
        },
    )

    # TODO: this widget should be hidden when the update method is set to EXACT
    tooltip = (
        "The local spatial search radius (isotropic, pixels) used when the update "
        "method is 'APPROXIMATE'"
    )
    max_search_radius = magicgui.widgets.create_widget(
        value=tracker_config.tracker_config.max_search_radius,
        name="max_search_radius",
        label="search radius",
        widget_type="SpinBox",
        options={"tooltip": tooltip},
    )

    update_method_widgets = [update_method_selector, max_search_radius]

    return update_method_widgets
