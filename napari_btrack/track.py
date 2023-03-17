from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy.typing as npt
    from btrack.config import TrackerConfig
    from magicgui.widgets import Container

import btrack
import magicgui.widgets
import napari
import qtpy.QtWidgets
from btrack.utils import segmentation_to_objects

import napari_btrack.config
import napari_btrack.sync
import napari_btrack.widgets

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


def track() -> Container:
    """Create widgets for the btrack plugin."""

    # First create our UI along with some default configs for the widgets
    widgets = napari_btrack.widgets.create_widgets()
    all_configs = napari_btrack.config.create_default_configs()

    btrack_widget = magicgui.widgets.Container(widgets=widgets, scrollable=True)
    btrack_widget.viewer = napari.current_viewer()
    btrack_widget.unscaled_configs = all_configs

    # Now set the callbacks
    @btrack_widget.config_selector.changed.connect
    def select_config():
        """Set widget values from a newly-selected base config"""

        # first update the previous config with the current widget values
        previous_config_name = all_configs.current_config
        previous_config = all_configs[previous_config_name]
        previous_config = napari_btrack.sync.update_config_from_widgets(
            unscaled_config=previous_config,
            container=btrack_widget,
        )

        # now load the newly-selected config and set widget values
        new_config_name = btrack_widget.config_selector.value
        all_configs.current_config = new_config_name
        new_config = all_configs[new_config_name]
        new_config = napari_btrack.sync.update_widgets_from_config(
            unscaled_config=new_config,
            container=btrack_widget,
        )

    @btrack_widget.call_button.changed.connect
    def run() -> None:
        """
        Update the TrackerConfig from widget values, run tracking,
        and add tracks to the viewer.
        """

        unscaled_config = all_configs[btrack_widget.config_selector.current_choice]
        unscaled_config = napari_btrack.sync.update_config_from_widgets(
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

        config = all_configs[config_name]
        config = napari_btrack.sync.update_widgets_from_config(
            unscaled_config=config,
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
        napari_btrack.sync.update_config_from_widgets(
            unscaled_config=unscaled_config,
            container=btrack_widget,
        )
        config = unscaled_config.scale_config()

        btrack.config.save_config(save_path, config)

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

    scroll = qtpy.QtWidgets.QScrollArea()
    scroll.setWidget(btrack_widget._widget._qwidget)
    btrack_widget._widget._qwidget = scroll

    return btrack_widget
