from pathlib import Path
from typing import Optional

import btrack
import napari
from btrack.utils import segmentation_to_objects
from magicgui import magicgui
from magicgui.widgets import FunctionGui


def run_tracker(objects, config_file_path):
    with btrack.BayesianTracker() as tracker:
        # configure the tracker using a config file
        tracker.configure_from_file(config_file_path)
        tracker.max_search_radius = 50

        # append the objects to be tracked
        tracker.append(objects)

        # set the volume
        tracker.volume = ((0, 1600), (0, 1200), (-1e5, 64.0))

        # track them (in interactive mode)
        tracker.track_interactive(step_size=100)

        # generate hypotheses and run the global optimizer
        tracker.optimize()

        # get the tracks in a format for napari visualization
        data, properties, graph = tracker.to_napari(ndim=2)
        return data, properties, graph


def track() -> FunctionGui:
    @magicgui(
        call_button=True,
        persist=True,
        config_file_path=dict(value=Path.home()),
        tracking_lower_limit=dict(
            value=0,
            label="tracking first layer",
            min=0,
            max=10_000,
        ),
        tracking_upper_limit=dict(
            value=-1,
            label="tracking last layer",
            min=-100,
            max=10_000,
        ),
        reset_button=dict(widget_type="PushButton", text="Reset defaults"),
    )
    def widget(
        viewer: napari.Viewer,
        segmentation: napari.layers.Image,
        config_file_path: Optional[Path],
        tracking_lower_limit: int,
        tracking_upper_limit: int,
        reset_button,
    ):
        """
        Parameters
        ----------

        tracking_upper_limit : int
            Last element of the segmented layer to analyse. -1 is the last one.
        """

        if tracking_upper_limit < 0:
            tracking_upper_limit += segmentation.data.shape[0]
        if not (
            tracking_lower_limit < tracking_upper_limit
            and 0 <= tracking_lower_limit < segmentation.data.shape[0]
            and tracking_upper_limit < segmentation.data.shape[0]
        ):
            raise ValueError(
                "The time range used for the tracking "
                f"{tracking_lower_limit}:{tracking_upper_limit} is not valid"
            )
        segmented_objects = segmentation_to_objects(
            segmentation.data[tracking_lower_limit:tracking_upper_limit, ...]
        )
        data, properties, graph = run_tracker(segmented_objects, config_file_path)

        properties["t"] += tracking_lower_limit
        data[:, 1] += tracking_lower_limit

        viewer.add_tracks(
            data=data, properties=properties, graph=graph, name=f"{segmentation}_btrack"
        )

    return widget
