from __future__ import annotations

import magicgui
import napari


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


def create_update_method_widgets():
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
        value=100,
        name="max_search_radius",
        label="search radius",
        widget_type="SpinBox",
        options={"tooltip": tooltip},
    )

    update_method_widgets = [update_method_selector, max_search_radius]

    return update_method_widgets


def create_control_widgets():
    """Create widgets for running the analysis or handling I/O.

    This includes widgets for running the tracking, saving and loading
    configuration files, and resetting the widget values to those in
    the selected config."""

    names = [
        "load_config_button",
        "save_config_button",
        "reset_button",
        "call_button",
    ]
    labels = [
        "Load configuration",
        "Save configuration",
        "Reset defaults",
        "Run",
    ]
    tooltips = [
        "Load a TrackerConfig json file.",
        "Export the current configuration to a TrackerConfig json file.",
        "Reset the current configuration to the defaults of the base config.",
        "Run the tracking analysis with the current configuration.",
    ]

    control_buttons = []
    for name, label, tooltip in zip(names, labels, tooltips):
        widget = magicgui.widgets.create_widget(
            name=name,
            label=label,
            widget_type="PushButton",
            options={"tooltip": tooltip},
        )
        control_buttons.append(widget)

    return control_buttons
