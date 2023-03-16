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
from pydantic import BaseModel
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
    """Values to scale unscaled TrackerConfig matrices by"""

    A: float
    H: float
    P: float
    G: float
    R: float


@dataclass
class UnscaledTackerConfig:
    """A helper dataclass to convert TrackerConfig matrices from scaled to unscaled.

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

        A_sigma = np.max(config.motion_model.A)
        config.motion_model.A /= A_sigma

        H_sigma = np.max(config.motion_model.H)
        config.motion_model.H /= H_sigma

        P_sigma = np.max(config.motion_model.P)
        config.motion_model.P /= P_sigma

        R_sigma = np.max(config.motion_model.R)
        config.motion_model.R /= R_sigma

        # Use only G, not Q
        # If we use both G and Q, then Q_sigma must be updated when G_sigma is,
        # and vice-versa
        # Instead, use G if it exists. If not, determine G from Q, which we can
        # do because Q is symmetric
        if config.motion_model.G is not None:
            G_sigma = np.max(config.motion_model.G)
            config.motion_model.G /= G_sigma
        elif config.motion_model.Q is not None:
            Q_sigma = np.max(config.motion_model.Q)
            G_sigma = Q_sigma**0.5
            config.motion_model.Q /= Q_sigma
            config.motion_model.G = config.motion_model.Q[0] / np.max(
                config.motion_model.Q[0]
            )
        else:
            _msg = "Either a `G` or `Q` matrix is required for the MotionModel."
            raise ValueError(_msg)

        sigmas = Sigmas(
            A=A_sigma,
            H=H_sigma,
            P=P_sigma,
            G=G_sigma,
            R=R_sigma,
        )

        return config, sigmas

    def scale_config(self):
        """Create a new TrackerConfig with scaled MotionModel matrices"""

        # Create a copy so that config values stay in sync with widget values
        scaled_config = copy.deepcopy(self.unscaled_config)

        scaled_config.motion_model.A *= self.sigmas.A
        scaled_config.motion_model.H *= self.sigmas.H
        scaled_config.motion_model.P *= self.sigmas.P
        scaled_config.motion_model.R *= self.sigmas.R
        scaled_config.motion_model.G *= self.sigmas.G
        scaled_config.motion_model.Q = (
            scaled_config.motion_model.G.T @ scaled_config.motion_model.G
        )

        return scaled_config


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


def _create_per_model_widgets(model: BaseModel) -> list[Widget]:
    """
    For a given model create the required list of widgets.
    The items "hypotheses" and the various matrices need customisation,
    otherwise we can use the napari default.
    """
    widgets: list[Widget] = []
    widget = create_widget(**html_label_widget(type(model).__name__))
    widgets.append(widget)
    print(type(model), widget)
    for parameter, default_value in model:
        if parameter in HIDDEN_VARIABLE_NAMES:
            print(f'{parameter} skipped')
            continue
        if parameter in Matrices().names:
            # only expose the scalar sigma to user
            sigma = Matrices.get_sigma(parameter, default_value)
            widget = create_widget(value=sigma, name=f"{parameter}_sigma", annotation=float)
            widgets.append(widget)
            print(type(model), widget)
        elif parameter == "hypotheses":
            # the hypothesis list should be represented as a series of checkboxes
            for choice in ALL_HYPOTHESES:
                widget = create_widget(value=(choice in default_value), name=choice, annotation=bool)
                widgets.append(widget)
                print(type(model), widget)
        else:  # use napari default
            widget = create_widget(value=default_value, name=parameter, annotation=type(default_value))
            widgets.append(widget)
            print(type(model), widget)

    return widgets


def _create_napari_specific_widgets(widgets: list[Widget]) -> None:
    """
    Add the widgets which interact with napari itself
    """
    widget = create_widget(**html_label_widget("Segmentation"))
    widgets.append(widget)
    print(widget)
    segmentation_widget = create_widget(
        name="segmentation",
        annotation=napari.layers.Labels,
        options={
            "tooltip": (
                "Should be a Labels layer. Convert an Image to Labels by right-clicking"
                "on it in the layers list, and clicking on 'Convert to Labels'"
            ),
        },
    )
    widgets.append(segmentation_widget)
    print(segmentation_widget)


def _create_pydantic_default_widgets(
    widgets: list[Widget], config: TrackerConfig
) -> None:
    """
    Create the widgets which have a tracker config equivalent.
    """
    widget = create_widget(name="max_search_radius", value=config.max_search_radius)
    widgets.append(widget)
    print(widget)
    model_configs = [config.motion_model, config.hypothesis_model]
    model_widgets = [_create_per_model_widgets(model) for model in model_configs]
    widgets.extend([item for sublist in model_widgets for item in sublist])


def _create_cell_or_particle_widget(widgets: list[Widget]) -> None:
    """Create a dropdown menu to choose between cell or particle mode."""
    widget = create_widget(**html_label_widget("Mode"))
    widgets.append(widget)
    print('mode', widget)
    widget = create_widget(name="mode", value="cell", options={"choices": ["cell", "particle"]})
    widgets.append(widget)
    print(widget)


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
                use_cell_config=(container.mode.value == "cell"),
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
    mode = container.mode.value
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
    container.mode.value = "cell" if mode_is_cell else "particle"


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
        widget = create_widget(name=widget_name, label=widget_label, widget_type=PushButton)
        print('Control buttons', widget)
        widgets.append(widget)


def track() -> Container:
    """
    Create a series of widgets programatically
    """
    # initialise a list for all widgets
    widgets: list = []

    # create all the widgets
    _create_napari_specific_widgets(widgets)
    _create_cell_or_particle_widget(widgets)
    _create_pydantic_default_widgets(widgets, default_cell_config)
    _create_button_widgets(widgets)

    btrack_widget = Container(widgets=widgets, scrollable=True)
    btrack_widget.viewer = napari.current_viewer()

    @btrack_widget.reset_button.changed.connect
    def restore_defaults() -> None:
        _update_widgets_from_config(btrack_widget, default_cell_config)

    @btrack_widget.call_button.changed.connect
    def run() -> None:
        config = _widgets_to_tracker_config(btrack_widget)
        segmentation = btrack_widget.segmentation.value
        data, properties, graph = run_tracker(segmentation, config)
        btrack_widget.viewer.add_tracks(
            data=data, properties=properties, graph=graph, name=f"{segmentation}_btrack"
        )

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
