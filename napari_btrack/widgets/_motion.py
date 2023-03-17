from __future__ import annotations

import magicgui


def _make_label_bold(label: str) -> str:
    """Generate html for a bold label"""

    bold_label = f"<b>{label}</b>"
    return bold_label


def _create_sigma_widgets():
    """Create widgets for setting the magnitudes of the MotionModel matrices"""

    tooltip = "Magnitude of error in initial estimates.\n Used to scale the matrix P."
    P_sigma = magicgui.widgets.create_widget(
        value=150.0,
        name="P_sigma",
        label=f"max({_make_label_bold('P')})",
        widget_type="FloatSpinBox",
        options={"tooltip": tooltip},
    )

    tooltip = "Magnitude of error in process.\n Used to scale the matrix G."
    G_sigma = magicgui.widgets.create_widget(
        value=15.0,
        name="G_sigma",
        label=f"max({_make_label_bold('G')})",
        widget_type="FloatSpinBox",
        options={"tooltip": tooltip},
    )

    tooltip = "Magnitude of error in measurements.\n Used to scale the matrix R."
    R_sigma = magicgui.widgets.create_widget(
        value=5.0,
        name="R_sigma",
        label=f"max({_make_label_bold('R')})",
        widget_type="FloatSpinBox",
        options={"tooltip": tooltip},
    )

    sigma_widgets = [
        P_sigma,
        G_sigma,
        R_sigma,
    ]

    return sigma_widgets


def create_motion_model_widgets():
    """Create widgets for setting parameters of the MotionModel"""

    motion_model_label = magicgui.widgets.create_widget(
        label=_make_label_bold("Motion model"),
        widget_type="Label",
        gui_only=True,
    )

    sigma_widgets = _create_sigma_widgets()

    tooltip = "Integration limits for calculating probabilities"
    accuracy = magicgui.widgets.create_widget(
        value=7.5,
        name="accuracy",
        label="accuracy",
        widget_type="FloatSpinBox",
        options={"tooltip": tooltip},
    )

    tooltip = "Number of frames without observation before marking as lost"
    max_lost_frames = magicgui.widgets.create_widget(
        value=5,
        name="max_lost",
        label="max lost",
        widget_type="SpinBox",
        options={"tooltip": tooltip},
    )

    motion_model_widgets = [
        motion_model_label,
        *sigma_widgets,
        accuracy,
        max_lost_frames,
    ]

    return motion_model_widgets
