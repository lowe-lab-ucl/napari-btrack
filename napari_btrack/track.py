from __future__ import annotations

import btrack
import napari
import numpy.typing as npt
from btrack.config import (
    TrackerConfig,
    save_config,
)
from btrack.utils import segmentation_to_objects
from magicgui.widgets import Container, create_widget
from qtpy.QtWidgets import QScrollArea

import napari_btrack.widgets
from napari_btrack.config import (
    Sigmas,
    TrackerConfigs,
    UnscaledTackerConfig,
)

__all__ = [
    "track",
]


def run_tracker(
    segmentation: napari.layers.Image | napari.layers.Labels,
    tracker_config: TrackerConfig,
) -> tuple[npt.NDArray, dict, dict]:
    """
    Runs BayesianTracker with given segmentation and configuration.
    """
    with btrack.BayesianTracker() as tracker:
        tracker.configure(tracker_config)

        # append the objects to be tracked
        segmented_objects = segmentation_to_objects(segmentation.data)
        tracker.append(segmented_objects)

        # set the volume
        segmentation_size = segmentation.level_shapes[0]
        # btrack order of dimensions is XY(Z)
        # napari order of dimensions is T(Z)XY
        # so we ignore the first entry and then iterate backwards
        tracker.volume = tuple((0, s) for s in segmentation_size[1:][::-1])

        # track them (in interactive mode)
        tracker.track_interactive(step_size=100)

        # generate hypotheses and run the global optimizer
        tracker.optimize()

        # get the tracks in a format for napari visualization
        data, properties, graph = tracker.to_napari(ndim=2)
        return data, properties, graph


def _make_label_bold(label: str) -> str:
    """Generate html for a bold label"""

    bold_label = f"<b>{label}</b>"
    return bold_label


def _create_motion_model_sigma_widgets(sigmas: Sigmas):
    """Create widgest for setting the magnitudes of the MotionModel matrices"""

    tooltip = "Magnitude of error in initial estimates.\n Used to scale the matrix P."
    P_sigma = create_widget(
        value=sigmas.P,
        name="P_sigma",
        label=f"max({_make_label_bold('P')})",
        widget_type="FloatSpinBox",
        options={"tooltip": tooltip},
    )

    tooltip = "Magnitude of error in process.\n Used to scale the matrix G."
    G_sigma = create_widget(
        value=sigmas.G,
        name="G_sigma",
        label=f"max({_make_label_bold('G')})",
        widget_type="FloatSpinBox",
        options={"tooltip": tooltip},
    )

    tooltip = "Magnitude of error in measurements.\n Used to scale the matrix R."
    R_sigma = create_widget(
        value=sigmas.R,
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


def _create_motion_model_widgets(tracker_config: UnscaledTackerConfig):
    """Create widgets for setting parameters of the MotionModel"""

    motion_model_label = create_widget(
        label=_make_label_bold("Motion model"),
        widget_type="Label",
        gui_only=True,
    )

    sigma_widgets = _create_motion_model_sigma_widgets(
        sigmas=tracker_config.sigmas,
    )

    tooltip = "Integration limits for calculating probabilities"
    accuracy = create_widget(
        value=tracker_config.tracker_config.motion_model.accuracy,
        name="accuracy",
        label="accuracy",
        widget_type="FloatSpinBox",
        options={"tooltip": tooltip},
    )

    tooltip = "Number of frames without observation before marking as lost"
    max_lost_frames = create_widget(
        value=tracker_config.tracker_config.motion_model.max_lost,
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


def update_config_from_widgets(
    unscaled_config: UnscaledTackerConfig,
    container: Container,
):
    """Update an UnscaledTrackerConfig with the current widget values."""

    sigmas = unscaled_config.sigmas
    sigmas.P = container.P_sigma.value
    sigmas.G = container.G_sigma.value
    sigmas.R = container.R_sigma.value

    config = unscaled_config.tracker_config
    config.update_method = (
        container.update_method_selector._widget._qwidget.currentIndex()
    )
    config.max_search_radius = container.max_search_radius.value

    motion_model = config.motion_model
    motion_model.accuracy = container.accuracy.value
    motion_model.max_lost = container.max_lost.value

    hypothesis_model = config.hypothesis_model
    hypotheses = []
    for hypothesis in [
        "P_FP",
        "P_init",
        "P_term",
        "P_link",
        "P_branch",
        "P_dead",
        "P_merge",
    ]:
        if container[hypothesis].value:
            hypotheses.append(hypothesis)
    hypothesis_model.hypotheses = hypotheses

    hypothesis_model.lambda_time = container.lambda_time.value
    hypothesis_model.lambda_dist = container.lambda_dist.value
    hypothesis_model.lambda_link = container.lambda_link.value
    hypothesis_model.lambda_branch = container.lambda_branch.value

    hypothesis_model.theta_dist = container.theta_dist.value
    hypothesis_model.theta_time = container.theta_time.value
    hypothesis_model.dist_thresh = container.dist_thresh.value
    hypothesis_model.time_thresh = container.time_thresh.value
    hypothesis_model.apop_thresh = container.apop_thresh.value

    hypothesis_model.segmentation_miss_rate = container.segmentation_miss_rate.value


def update_widgets_from_config(
    unscaled_config: UnscaledTackerConfig,
    container: Container,
):
    """
    Update the widgets in a container with the values in an
    UnscaledTrackerConfig.
    """

    sigmas = unscaled_config.sigmas
    container.P_sigma.value = sigmas.P
    container.G_sigma.value = sigmas.G
    container.R_sigma.value = sigmas.R

    config = unscaled_config.tracker_config
    container.update_method_selector.value = config.update_method.name
    container.max_search_radius.value = config.max_search_radius

    motion_model = config.motion_model
    container.accuracy.value = motion_model.accuracy
    container.max_lost.value = motion_model.max_lost

    hypothesis_model = config.hypothesis_model
    for hypothesis in [
        "P_FP",
        "P_init",
        "P_term",
        "P_link",
        "P_branch",
        "P_dead",
        "P_merge",
    ]:
        is_checked = hypothesis in hypothesis_model.hypotheses
        container[hypothesis].value = is_checked

    container.lambda_time.value = hypothesis_model.lambda_time
    container.lambda_dist.value = hypothesis_model.lambda_dist
    container.lambda_link.value = hypothesis_model.lambda_link
    container.lambda_branch.value = hypothesis_model.lambda_branch

    container.theta_dist.value = hypothesis_model.theta_dist
    container.theta_time.value = hypothesis_model.theta_time
    container.dist_thresh.value = hypothesis_model.dist_thresh
    container.time_thresh.value = hypothesis_model.time_thresh
    container.apop_thresh.value = hypothesis_model.apop_thresh

    container.segmentation_miss_rate.value = hypothesis_model.segmentation_miss_rate

    return container


def track() -> Container:  # noqa: PLR0915
    """Create widgets for the btrack plugin."""

    # TrackerConfigs automatically loads default cell and particle configs
    all_configs = TrackerConfigs()
    current_config = all_configs["cell"]

    input_widgets = napari_btrack.widgets.create_input_widgets()
    update_method_widgets = napari_btrack.widgets.create_update_method_widgets(
        tracker_config=current_config,
    )
    motion_model_widgets = _create_motion_model_widgets(
        tracker_config=current_config,
    )
    hypothesis_model_widgets = napari_btrack.widgets.create_hypothesis_model_widgets()
    control_buttons = napari_btrack.widgets.create_control_widgets()

    widgets: list = [
        *input_widgets,
        *update_method_widgets,
        *motion_model_widgets,
        *hypothesis_model_widgets,
        *control_buttons,
    ]

    btrack_widget = Container(widgets=widgets, scrollable=True)
    btrack_widget.viewer = napari.current_viewer()
    btrack_widget.unscaled_configs = all_configs

    @btrack_widget.config_selector.changed.connect
    def select_config() -> None:
        """Set widget values from a newly-selected base config"""

        # first update the previous config with the current widget values
        previous_config_name = all_configs.current_config
        update_config_from_widgets(
            unscaled_config=all_configs[previous_config_name],
            container=btrack_widget,
        )
        # now load the newly-selected config and set widget values
        new_config_name = btrack_widget.config_selector.value
        all_configs.current_config = new_config_name
        update_widgets_from_config(
            unscaled_config=all_configs[new_config_name],
            container=btrack_widget,
        )

    @btrack_widget.call_button.changed.connect
    def run() -> None:
        """
        Update the TrackerConfig from widget values, run tracking,
        and add tracks to the viewer.
        """

        unscaled_config = all_configs[btrack_widget.config_selector.current_choice]
        update_config_from_widgets(
            unscaled_config=unscaled_config,
            container=btrack_widget,
        )

        config = unscaled_config.scale_config()
        segmentation = btrack_widget.segmentation_selector.value
        data, properties, graph = run_tracker(segmentation, config)

        btrack_widget.viewer.add_tracks(
            data=data,
            properties=properties,
            graph=graph,
            name=f"{segmentation}_btrack",
        )

    @btrack_widget.reset_button.changed.connect
    def restore_defaults() -> None:
        """ "Reload the config file and set widgets to default values."""

        config_name = all_configs.current_config
        filename = all_configs[config_name].filename
        all_configs.add_config(
            filename=filename,
            overwrite=True,
        )

        update_widgets_from_config(
            unscaled_config=all_configs[config_name],
            container=btrack_widget,
        )

    @btrack_widget.save_config_button.changed.connect
    def save_config_to_json() -> None:
        """Save widget values to file"""

        save_path = napari_btrack.widgets.save_path_dialogue_box()
        if save_path is None:
            # user has cancelled
            return

        unscaled_config = all_configs[btrack_widget.config_selector.current_choice]
        update_config_from_widgets(
            unscaled_config=unscaled_config,
            container=btrack_widget,
        )
        config = unscaled_config.scale_config()

        save_config(save_path, config)

    @btrack_widget.load_config_button.changed.connect
    def load_config_from_json() -> None:
        """Load a config from file and set it as the selected base config"""

        load_path = napari_btrack.widgets.load_path_dialogue_box()
        if load_path is None:
            # user has cancelled
            return

        config_name = all_configs.add_config(filename=load_path, overwrite=False)
        btrack_widget.config_selector.options["choices"].append(config_name)
        btrack_widget.config_selector.reset_choices()
        btrack_widget.config_selector.value = config_name

    scroll = QScrollArea()
    scroll.setWidget(btrack_widget._widget._qwidget)
    btrack_widget._widget._qwidget = scroll

    return btrack_widget
