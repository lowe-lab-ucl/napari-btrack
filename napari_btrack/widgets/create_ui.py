from napari_btrack.widgets._general import (
    create_control_widgets,
    create_input_widgets,
    create_update_method_widgets,
)
from napari_btrack.widgets._hypothesis import create_hypothesis_model_widgets
from napari_btrack.widgets._motion import create_motion_model_widgets


def create_widgets():
    """Create all the widgets for the plugin"""

    input_widgets = create_input_widgets()
    update_method_widgets = create_update_method_widgets()
    motion_model_widgets = create_motion_model_widgets()
    hypothesis_model_widgets = create_hypothesis_model_widgets()
    control_buttons = create_control_widgets()

    widgets = [
        *input_widgets,
        *update_method_widgets,
        *motion_model_widgets,
        *hypothesis_model_widgets,
        *control_buttons,
    ]

    return widgets
