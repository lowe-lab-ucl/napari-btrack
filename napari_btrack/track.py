from __future__ import annotations

import copy
import logging
import os
from dataclasses import dataclass, field
from typing import Any

import btrack
import napari
import numpy as np
import numpy.typing as npt
from btrack import datasets
from btrack.config import (
    HypothesisModel,
    MotionModel,
    TrackerConfig,
    load_config,
    save_config,
)
from btrack.utils import segmentation_to_objects
from magicgui.application import use_app
from magicgui.types import FileDialogMode
from magicgui.widgets import Container, PushButton, Widget, create_widget
from qtpy.QtWidgets import QScrollArea

default_cell_config = load_config(datasets.cell_config())

# widgets for which the default widget type is incorrect
HIDDEN_VARIABLE_NAMES = [
    "name",
    "measurements",
    "states",
    "dt",
    "apoptosis_rate",
    "prob_not_assign",
    "eta",
]
ALL_HYPOTHESES = ["P_FP", "P_init", "P_term", "P_link", "P_branch", "P_dead"]


@dataclass
class Sigmas:
    """Values to scale TrackerConfig MotionModel matrices by"""

    P: float
    G: float
    R: float


@dataclass
class UnscaledTackerConfig:
    """Convert TrackerConfig MotionModel matrices from scaled to unscaled.

    This is needed because TrackerConfig stores "scaled" matrices, i.e. it
    doesn't store sigma and the "unscaled" MotionModel matrices separately.
    """

    filename: os.PathLike
    unscaled_config: TrackerConfig = field(init=False)
    sigmas: Sigmas = field(init=False)

    def __post_init__(self):
        """Create the TrackerConfig and un-scale the MotionModel indices"""

        config = load_config(self.filename)
        self.unscaled_config, self.sigmas = self._unscale_config(config)

    def _unscale_config(self, config: TrackerConfig) -> tuple[TrackerConfig, Sigmas]:
        """Convert the matrices of a scaled TrackerConfig MotionModel to unscaled."""

        P_sigma = np.max(config.motion_model.P)
        config.motion_model.P /= P_sigma

        R_sigma = np.max(config.motion_model.R)
        config.motion_model.R /= R_sigma

        # Use only G, not Q. If we use both G and Q, then Q_sigma must be updated
        # when G_sigma is, and vice-versa
        # Instead, use G if it exists. If not, determine G from Q, which we can
        # do because Q = G.T @ G
        if config.motion_model.G is None:
            config.motion_model.G = config.motion_model.Q.diagonal() ** 0.5
        G_sigma = np.max(config.motion_model.G)
        config.motion_model.G /= G_sigma

        sigmas = Sigmas(
            P=P_sigma,
            G=G_sigma,
            R=R_sigma,
        )

        return config, sigmas

    def scale_config(self):
        """Create a new TrackerConfig with scaled MotionModel matrices"""

        # Create a copy so that config values stay in sync with widget values
        scaled_config = copy.deepcopy(self.unscaled_config)
        scaled_config.motion_model.P *= self.sigmas.P
        scaled_config.motion_model.R *= self.sigmas.R
        scaled_config.motion_model.G *= self.sigmas.G
        scaled_config.motion_model.Q = (
            scaled_config.motion_model.G.T @ scaled_config.motion_model.G
        )

        return scaled_config


@dataclass
class TrackerConfigs:
    configs: dict[str, UnscaledTackerConfig] = field(default_factory=dict)

    def __post_init__(self):
        """Add the default cell and particle configs."""

        self.add_config(
            filename=datasets.cell_config(),
            name="cell",
        )
        self.add_config(
            filename=datasets.particle_config(),
            name="particle",
        )

    def __getitem__(self, config_name):
        return self.configs[config_name]

    def add_config(self, filename, name=None):
        """Load a TrackerConfig and add it to the store"""

        config = UnscaledTackerConfig(filename)
        config_name = name if name is not None else config.unscaled_config.name

        # TODO: Make the combobox editable so config names can be changed within the GUI
        if config_name in self.configs:
            _msg = (
                f"Config '{config_name}' already exists - config names must be unique."
            )
            raise ValueError(_msg)

        self.configs[config_name] = config


@dataclass
class Matrices:
    """A helper dataclass to adapt matrix representation to and from pydantic.
    This is needed because TrackerConfig stores "scaled" matrices, i.e.
    doesn't store sigma and the "unscaled" matrix separately.
    """

    names: list[str] = field(default_factory=lambda: ["A", "H", "P", "G", "R", "Q"])
    widget_labels: list[str] = field(
        default_factory=lambda: [
            "A_sigma",
            "H_sigma",
            "P_sigma",
            "G_sigma",
            "R_sigma",
            "Q_sigma",
        ]
    )
    default_sigmas: list[float] = field(
        default_factory=lambda: [1.0, 1.0, 150.0, 15.0, 5.0]
    )
    unscaled_matrices: dict[str, npt.NDArray[np.float64]] = field(
        default_factory=lambda: {
            "A_cell": np.array(
                [
                    [1, 0, 0, 1, 0, 0],
                    [0, 1, 0, 0, 1, 0],
                    [0, 0, 1, 0, 0, 1],
                    [0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 1],
                ]
            ),
            "A_particle": np.array(
                [
                    [1, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 1],
                ]
            ),
            "H": np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]]),
            "P": np.array(
                [
                    [0.1, 0, 0, 0, 0, 0],
                    [0, 0.1, 0, 0, 0, 0],
                    [0, 0, 0.1, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 1],
                ]
            ),
            "G": np.array([[0.5, 0.5, 0.5, 1, 1, 1]]),
            "R": np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            "Q": np.array(
                [
                    [56.25, 56.25, 56.25, 112.5, 112.5, 112.5],
                    [56.25, 56.25, 56.25, 112.5, 112.5, 112.5],
                    [56.25, 56.25, 56.25, 112.5, 112.5, 112.5],
                    [112.5, 112.5, 112.5, 225.0, 225.0, 225.0],
                    [112.5, 112.5, 112.5, 225.0, 225.0, 225.0],
                    [112.5, 112.5, 112.5, 225.0, 225.0, 225.0],
                ]
            ),
        }
    )

    @classmethod
    def get_scaled_matrix(
        cls, name: str, *, sigma: float, use_cell_config: bool = True
    ) -> list[float]:
        """Returns the scaled version (i.e. the unscaled matrix multiplied by sigma)
        of the matrix.

        Keyword arguments:
        name -- the matrix name (can be one of A, H, P, G, R)
        sigma -- the factor to scale the matrix entries with
        cell -- whether to use cell config matrices or not (default true)
        """
        if name == "A":
            name = "A_cell" if use_cell_config else "A_particle"
        return (np.asarray(cls().unscaled_matrices[name]) * sigma).tolist()

    @classmethod
    def get_sigma(cls, name: str, scaled_matrix: npt.NDArray[np.float64]) -> float:
        """Returns the factor sigma which is the multiplier between the given scaled
        matrix and the unscaled matrix of the given name.

        Note: The calculation is done with the top-left entry of the matrix,
        and all other entries are ignored.

        Keyword arguments:
        name -- the matrix name (can be one of A, H, P, G, R)
        scaled_matrix -- the scaled matrix to find sigma from.
        """
        if name == "A":
            name = "A_cell"  # doesn't matter which A we use here, as [0][0] is the same
        return scaled_matrix[0][0] / cls().unscaled_matrices[name][0][0]


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


def get_save_path():
    """Helper function to open a save configuration file dialog."""
    show_file_dialog = use_app().get_obj("show_file_dialog")
    return show_file_dialog(
        mode=FileDialogMode.OPTIONAL_FILE,
        caption="Specify file to save btrack configuration",
        start_path=None,
        filter="*.json",
    )


def get_load_path():
    """Helper function to open a load configuration file dialog."""
    show_file_dialog = use_app().get_obj("show_file_dialog")
    return show_file_dialog(
        mode=FileDialogMode.EXISTING_FILE,
        caption="Choose JSON file containing btrack configuration",
        start_path=None,
        filter="*.json",
    )


def html_label_widget(label: str, tag: str = "b") -> dict:
    """
    Create a HMTL label widget.
    """
    return {
        "widget_type": "Label",
        "label": f"<{tag}>{label}</{tag}>",
    }


def _widgets_to_tracker_config(container: Container) -> TrackerConfig:
    """Helper function to convert from the widgets to a tracker configuration."""
    motion_model_dict: dict[str, Any] = {}
    hypothesis_model_dict = {}

    motion_model_keys = default_cell_config.motion_model.dict().keys()
    hypothesis_model_keys = default_cell_config.hypothesis_model.dict().keys()
    hypotheses = []
    for widget in container:
        # setup motion model
        # matrices need special treatment
        if widget.name in Matrices().widget_labels:
            sigma = getattr(container, widget.name).value
            matrix_name = widget.name.rstrip("_sigma")
            matrix = Matrices.get_scaled_matrix(
                matrix_name,
                sigma=sigma,
                use_cell_config=(container.config_selector.value == "cell"),
            )
            motion_model_dict[matrix_name] = matrix
        elif widget.name in motion_model_keys:
            motion_model_dict[widget.name] = widget.value
        # setup hypothesis model
        if widget.name in hypothesis_model_keys:
            hypothesis_model_dict[widget.name] = widget.value
        # hypotheses need special treatment
        if widget.name in ALL_HYPOTHESES and getattr(container, widget.name).value:
            hypotheses.append(widget.name)

    # add some non-exposed default values to the motion model
    mode = container.config_selctor.value
    for default_name, default_value in zip(
        ["measurements", "states", "dt", "prob_not_assign", "name"],
        [3, 6, 1.0, 0.001, f"{mode}_motion"],
    ):
        motion_model_dict[default_name] = default_value

    # add some non-exposed default value to the hypothesis model
    for default_name, default_value in zip(
        ["apoptosis_rate", "eta", "name"],
        [0.001, 1.0e-10, f"{mode}_hypothesis"],
    ):
        hypothesis_model_dict[default_name] = default_value

    # add hypotheses to hypothesis model
    hypothesis_model_dict["hypotheses"] = hypotheses
    motion_model = MotionModel(**motion_model_dict)
    hypothesis_model = HypothesisModel(**hypothesis_model_dict)

    # add parameters outside the internal models
    max_search_radius = container.max_search_radius.value
    return TrackerConfig(
        max_search_radius=max_search_radius,
        motion_model=motion_model,
        hypothesis_model=hypothesis_model,
    )


def _update_widgets_from_config(container: Container, config: TrackerConfig) -> None:
    """Helper function to update a container's widgets
    with the values in a given tracker config.
    """
    container.max_search_radius.value = config.max_search_radius
    for model in ["motion_model", "hypothesis_model", "object_model"]:
        if model_config := getattr(config, model):
            for parameter, value in model_config:
                if parameter in HIDDEN_VARIABLE_NAMES:
                    continue
                if parameter in Matrices().names:
                    sigma = Matrices.get_sigma(parameter, value)
                    getattr(container, f"{parameter}_sigma").value = sigma
                elif parameter == "hypotheses":
                    for hypothesis in ALL_HYPOTHESES:
                        getattr(container, hypothesis).value = hypothesis in value
                else:
                    getattr(container, parameter).value = value
    # we can determine whether we are in particle or cell mode
    # by checking whether the 4th entry of the first row of the
    # A matrix is 1 or 0 (1 for cell mode)
    mode_is_cell = config.motion_model.A[0, 3] == 1
    logging.info(f"mode is cell: {mode_is_cell}")
    container.config_selector.value = "cell" if mode_is_cell else "particle"


def _create_button_widgets(widgets: list[Widget]) -> None:
    """Create the set of button widgets needed:
    run, save/load configuration and reset."""
    widget_names = [
        "load_config_button",
        "save_config_button",
        "reset_button",
        "call_button",
    ]
    widget_labels = [
        "Load configuration",
        "Save configuration",
        "Reset defaults",
        "Run",
    ]
    widget = create_widget(**html_label_widget("Control buttons"))
    widgets.append(widget)
    print(widget)
    for widget_name, widget_label in zip(widget_names, widget_labels):
        widget = create_widget(
            name=widget_name, label=widget_label, widget_type=PushButton
        )
        print("Control buttons", widget)
        widgets.append(widget)


def _create_input_widgets():
    """Create widgets for selecting labels layer and TrackerConfig"""

    tooltip = (
        "Select a 'Labels' layer to use for tracking.\n"
        "To use an 'Image' layer, first convert 'Labels' by right-clicking "
        "on it in the layers list, and clicking on 'Convert to Labels'"
    )
    segmentation_selector = create_widget(
        annotation=napari.layers.Labels,
        name="segmentation_selector",
        label="segmentation",
        options={"tooltip": tooltip},
    )

    tooltip = "Select a loaded configuration.\nNote, this will update values set below."
    config_selector = create_widget(
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


def _create_update_method_widgets(tracker_config: UnscaledTackerConfig):
    """Create widgets for selecting the update method"""

    tooltip = (
        "Select the update method.\n"
        "EXACT: exact calculation of Bayesian belief matrix.\n"
        "APPROXIMATE: approximate the Bayesian belief matrix. Useful for datasets with "
        "more than 1000 particles per frame."
    )
    update_method = create_widget(
        value="EXACT",
        name="update_method",
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
    max_search_radius = create_widget(
        value=tracker_config.unscaled_config.max_search_radius,
        name="max_search_radius",
        label="search radius",
        widget_type="SpinBox",
        options={"tooltip": tooltip},
    )

    update_method_widgets = [update_method, max_search_radius]

    return update_method_widgets


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
        name="G_sigma",
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
        value=tracker_config.unscaled_config.motion_model.accuracy,
        name="accuracy",
        label="accuracy",
        widget_type="FloatSpinBox",
        options={"tooltip": tooltip},
    )

    tooltip = "Number of frames without observation before marking as lost"
    max_lost_frames = create_widget(
        value=tracker_config.unscaled_config.motion_model.max_lost,
        name="max_lost",
        label="max lost",
        widget_type="FloatSpinBox",
        options={"tooltip": tooltip},
    )

    motion_model_widgets = [
        motion_model_label,
        *sigma_widgets,
        accuracy,
        max_lost_frames,
    ]

    return motion_model_widgets


def _create_hypothesis_model_hypotheses_widgets():
    """Create widgets for selecting which hypotheses to generate."""

    hypotheses = [
        "P_FP",
        "P_init",
        "P_term",
        "P_link",
        "P_branch",
        "P_dead",
        "P_merge",
    ]
    tooltips = [
        "Hypothesis that a tracklet is a false positive detection. Always required.",
        "Hypothesis that a tracklet starts at the beginning of the movie or edge of the FOV.",  # noqa: E501
        "Hypothesis that a tracklet ends at the end of the movie or edge of the FOV.",
        "Hypothesis that two tracklets should be linked together.",
        "Hypothesis that a tracklet can split onto two daughter tracklets.",
        "Hypothesis that a tracklet terminates without leaving the FOV.",
        "Hypothesis that two tracklets merge into one tracklet.",
    ]

    hypotheses_widgets = []
    for hypothesis, tooltip in zip(hypotheses, tooltips):
        widget = create_widget(
            value=True,
            name=hypothesis,
            label=hypothesis,
            widget_type="CheckBox",
            options={"tooltip": tooltip},
        )
        hypotheses_widgets.append(widget)

    # P_FP is always required
    P_FP_hypothesis = hypotheses_widgets[0]
    P_FP_hypothesis.enabled = False

    return hypotheses_widgets


def _create_hypothesis_model_scaling_factor_widgets():
    """Create widgets for setting the scaling factors of the HypothesisModel"""

    values = [5.0, 3.0, 10.0, 50.0]
    names = [
        "lambda_time",
        "lambda_dist",
        "lambda_link",
        "lambda_branch",
    ]
    labels = [
        "λ time",
        "λ distance",
        "λ linking",
        "λ branching",
    ]
    tooltips = [
        "Scaling factor for the influence of time when determining initialization or termination hypotheses.",  # noqa: E501
        "Scaling factor for the influence of distance at the border when determining initialization or termination hypotheses.",  # noqa: E501
        "Scaling factor for the influence of track-to-track distance on linking probability.",  # noqa: E501
        "Scaling factor for the influence of cell state and position on division (mitosis/branching) probability.",  # noqa: E501
    ]

    scaling_factor_widgets = []
    for value, name, label, tooltip in zip(values, names, labels, tooltips):
        widget = create_widget(
            value=value,
            name=name,
            label=label,
            widget_type="FloatSpinBox",
            options={"tooltip": tooltip},
        )
        scaling_factor_widgets.append(widget)

    return scaling_factor_widgets


def _create_hypothesis_model_threshold_widgets():
    """Create widgets for setting thresholds for the HypothesisModel"""

    tooltip = (
        "A threshold distance from the edge of the FOV to add an "
        "initialization or termination hypothesis."
    )
    distance_threshold = create_widget(
        value=20.0,
        name="theta_dist",
        label="distance threshold",
        widget_type="FloatSpinBox",
        options={"tooltip": tooltip},
    )

    tooltip = (
        "A threshold time from the beginning or end of movie to add "
        "an initialization or termination hypothesis."
    )
    time_threshold = create_widget(
        value=5.0,
        name="theta_time",
        label="time threshold",
        widget_type="FloatSpinBox",
        options={"tooltip": tooltip},
    )

    tooltip = (
        "Number of apoptotic detections to be considered a genuine event.\n"
        "Detections are counted consecutively from the back of the track"
    )
    apoptosis_threshold = create_widget(
        value=5,
        name="apop_thresh",
        label="apoptosis threshold",
        widget_type="SpinBox",
        options={"tooltip": tooltip},
    )

    threshold_widgets = [
        distance_threshold,
        time_threshold,
        apoptosis_threshold,
    ]

    return threshold_widgets


def _create_hypothesis_model_bin_size_widgets():
    """Create widget for setting bin sizes for the HypothesisModel"""

    tooltip = (
        "Isotropic spatial bin size for considering hypotheses.\n"
        "Larger bin sizes generate more hypothesese for each tracklet."
    )
    distance_bin_size = create_widget(
        value=40.0,
        name="dist_thresh",
        label="distance bin size",
        widget_type="FloatSpinBox",
        options={"tooltip": tooltip},
    )

    tooltip = (
        "Temporal bin size for considering hypotheses.\n"
        "Larger bin sizes generate more hypothesese for each tracklet."
    )
    time_bin_size = create_widget(
        value=2.0,
        name="time_thresh",
        label="time bin size",
        widget_type="FloatSpinBox",
        options={"tooltip": tooltip},
    )

    bin_size_widgets = [
        distance_bin_size,
        time_bin_size,
    ]

    return bin_size_widgets


def _create_hypothesis_model_widgets(tracker_config: UnscaledTackerConfig):
    """Create widgets for setting parameters of the MotionModel"""

    hypothesis_model_label = create_widget(
        label=_make_label_bold("Hypothesis model"),
        widget_type="Label",
        gui_only=True,
    )

    hypotheses_widgets = _create_hypothesis_model_hypotheses_widgets()
    scaling_factor_widgets = _create_hypothesis_model_scaling_factor_widgets()
    threshold_widgets = _create_hypothesis_model_threshold_widgets()
    bin_size_widgets = _create_hypothesis_model_bin_size_widgets()

    tooltip = (
        "Miss rate for the segmentation.\n"
        "e.g. 1/100 segmentations incorrect gives a segmentation miss rate of 0.01."
    )
    segmentation_miss_rate_widget = create_widget(
        value=0.1,
        name="segmentation_miss_rate",
        label="miss rate",
        widget_type="FloatSpinBox",
        options={"tooltip": tooltip},
    )

    tooltip = (
        "Disable the time and distance thresholds.\n"
        "This means that tracks can initialize or terminate anywhere and"
        "at any time in the dataset."
    )
    relax = create_widget(
        value=True,
        name="relax",
        label="relax thresholds",
        widget_type="CheckBox",
        options={"tooltip": tooltip},
    )

    hypothesis_model_widgets = [
        hypothesis_model_label,
        *hypotheses_widgets,
        *scaling_factor_widgets,
        *threshold_widgets,
        *bin_size_widgets,
        segmentation_miss_rate_widget,
        relax,
    ]

    return hypothesis_model_widgets


def track() -> Container:
    """
    Create a series of widgets programatically
    """

    # TrackerConfigs automatically loads default cell and particle configs
    all_configs = TrackerConfigs()
    current_config = all_configs["cell"]

    input_widgets = _create_input_widgets()
    update_method_widgets = _create_update_method_widgets(
        tracker_config=current_config,
    )
    motion_model_widgets = _create_motion_model_widgets(
        tracker_config=current_config,
    )
    hypothesis_model_widgets = _create_hypothesis_model_widgets(
        tracker_config=current_config,
    )

    widgets: list = [
        *input_widgets,
        *update_method_widgets,
        *motion_model_widgets,
        *hypothesis_model_widgets,
    ]

    _create_button_widgets(widgets)

    btrack_widget = Container(widgets=widgets, scrollable=True)
    btrack_widget.viewer = napari.current_viewer()

    @btrack_widget.reset_button.changed.connect
    def restore_defaults() -> None:
        _update_widgets_from_config(btrack_widget, default_cell_config)

    @btrack_widget.call_button.changed.connect
    def run() -> None:
        config = _widgets_to_tracker_config(btrack_widget)
        segmentation = btrack_widget.segmentation_selector.value
        data, properties, graph = run_tracker(segmentation, config)
        btrack_widget.viewer.add_tracks(
            data=data, properties=properties, graph=graph, name=f"{segmentation}_btrack"
        )

    @btrack_widget.config_selector.changed.connect
    def select_config() -> None:
        pass
        # TODO: add function to update widgets from current UnscaledTackerConfig
        # update_widgets_from_config(config=all_configs[new_config_name])

    @btrack_widget.save_config_button.changed.connect
    def save_config_to_json() -> None:
        save_path = get_save_path()
        if save_path:  # save path is None if user cancels
            save_config(save_path, _widgets_to_tracker_config(btrack_widget))

    @btrack_widget.load_config_button.changed.connect
    def load_config_from_json() -> None:
        load_path = get_load_path()
        if load_path:  # load path is None if user cancels
            config = load_config(load_path)
            _update_widgets_from_config(btrack_widget, config)

    scroll = QScrollArea()
    scroll.setWidget(btrack_widget._widget._qwidget)
    btrack_widget._widget._qwidget = scroll

    return btrack_widget
