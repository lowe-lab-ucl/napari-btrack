from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from magicgui.widgets import Container

import json
from unittest.mock import patch

import btrack
import napari
import numpy as np
import numpy.typing as npt
import pytest
from btrack import datasets
from btrack.datasets import cell_config, particle_config

import napari_btrack
import napari_btrack.main

OLD_WIDGET_LAYERS = 1
NEW_WIDGET_LAYERS = 2


def test_add_widget(make_napari_viewer):
    """Checks that the track widget can be added inside a dock widget."""

    viewer = make_napari_viewer()
    num_dw = len(list(viewer.window._dock_widgets))
    viewer.window.add_plugin_dock_widget(
        plugin_name="napari-btrack",
        widget_name="Track",
    )

    assert len(list(viewer.window._dock_widgets)) == num_dw + 1  # noqa: S101


@pytest.fixture
def track_widget(make_napari_viewer) -> Container:
    """Provides an instance of the track widget to test"""
    make_napari_viewer()  # make sure there is a viewer available
    return napari_btrack.main.create_btrack_widget()


@pytest.mark.parametrize("config", [cell_config(), particle_config()])
def test_config_to_widgets_round_trip(track_widget, config):
    """Tests that going back and forth between
    config objects and widgets works as expected.
    """

    expected_config = btrack.config.load_config(config).json()

    unscaled_config = napari_btrack.config.UnscaledTackerConfig(config)
    napari_btrack.sync.update_widgets_from_config(unscaled_config, track_widget)
    napari_btrack.sync.update_config_from_widgets(unscaled_config, track_widget)

    actual_config = unscaled_config.scale_config().json()

    # use json.loads to avoid failure in string comparison because e.g "100.0" != "100"
    assert json.loads(actual_config) == json.loads(expected_config)  # noqa: S101


def test_save_button(track_widget):
    """Tests that clicking the save configuration button
    triggers a call to btrack.config.save_config with expected arguments.
    """

    current_config_name = track_widget.unscaled_configs.current_config
    expected_config = (
        track_widget.unscaled_configs[current_config_name].scale_config().json()
    )

    with patch(
        "napari_btrack.widgets.save_path_dialogue_box"
    ) as save_path_dialogue_box:
        save_path_dialogue_box.return_value = "user_config.json"
        track_widget.save_config_button.clicked()

    actual_config = btrack.config.load_config("user_config.json").json()

    # use json.loads to avoid failure in string comparison because e.g "100.0" != "100"
    assert json.loads(expected_config) == json.loads(actual_config)  # noqa: S101


def test_load_config(track_widget):
    """Tests that another TrackerConfig can be loaded."""

    all_configs = track_widget.unscaled_configs  # 2 configs loaded by default
    n_original_configs = len(all_configs.configs)
    original_config_name = all_configs.current_config

    with patch(
        "napari_btrack.widgets.load_path_dialogue_box"
    ) as load_path_dialogue_box:
        load_path_dialogue_box.return_value = cell_config()
        track_widget.load_config_button.clicked()

    n_expected_configs = n_original_configs + 1
    n_actual_configs = len(all_configs.configs)
    new_config_name = all_configs.current_config

    assert n_expected_configs == n_actual_configs  # noqa: S101
    assert track_widget.config_selector.value == "Default"  # noqa: S101
    assert new_config_name != original_config_name  # noqa: S101


def test_reset_button(track_widget):
    """Tests that clicking the reset button restores the default config values"""

    current_config_name = track_widget.unscaled_configs.current_config
    expected_config = (
        track_widget.unscaled_configs[current_config_name].scale_config().json()
    )

    # change some widget values
    track_widget.max_search_radius.value += 10
    track_widget.relax.value = not track_widget.relax

    # click reset button - restores defaults of the currently-selected base config
    track_widget.reset_button.clicked()
    actual_config = (
        track_widget.unscaled_configs[current_config_name].scale_config().json()
    )

    # use json.loads to avoid failure in string comparison because e.g "100.0" != "100"
    assert json.loads(actual_config) == json.loads(expected_config)  # noqa: S101


@pytest.fixture
def simplistic_tracker_outputs() -> (
    tuple[npt.NDArray, dict[str, npt.NDArray], dict[int, list]]
):
    """Provides simplistic return values of a btrack run.

    They have the correct types and dimensions, but contain zeros.
    Useful for mocking the tracker.
    """
    n, d = 10, 3
    data = np.zeros((n, d + 1))
    properties = {"some_property": np.zeros(n)}
    graph = {0: [0]}
    return data, properties, graph


def test_run_button(track_widget, simplistic_tracker_outputs):
    """Tests that clicking the run button calls run_tracker,
    and that the napari viewer has an additional tracks layer after running.
    """
    with patch("napari_btrack.main._run_tracker") as run_tracker:
        run_tracker.return_value = simplistic_tracker_outputs
        segmentation = datasets.example_segmentation()
        track_widget.viewer.add_labels(segmentation)
        assert len(track_widget.viewer.layers) == OLD_WIDGET_LAYERS  # noqa: S101
        track_widget.call_button.clicked()
    assert run_tracker.called  # noqa: S101
    assert len(track_widget.viewer.layers) == NEW_WIDGET_LAYERS  # noqa: S101
    assert isinstance(  # noqa: S101
        track_widget.viewer.layers[-1], napari.layers.Tracks
    )
