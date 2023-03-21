"""
This module contains functions for syncing widget values with TrackerConfig
values.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from btrack.config import TrackerConfig
    from magicgui.widgets import Container

    from napari_btrack.config import Sigmas, UnscaledTrackerConfig


def update_config_from_widgets(
    unscaled_config: UnscaledTrackerConfig,
    container: Container,
) -> TrackerConfig:
    """Update an UnscaledTrackerConfig with the current widget values."""

    sigmas: Sigmas = unscaled_config.sigmas
    sigmas.P = container.P_sigma.value
    sigmas.G = container.G_sigma.value
    sigmas.R = container.R_sigma.value

    config = unscaled_config.tracker_config
    config.update_method = container.update_method._widget._qwidget.currentIndex()
    config.max_search_radius = container.max_search_radius.value

    motion_model = config.motion_model
    motion_model.accuracy = container.accuracy.value
    motion_model.max_lost = container.max_lost.value

    hypothesis_model = config.hypothesis_model
    hypotheses = [
        hypothesis
        for hypothesis in [
            "P_FP",
            "P_init",
            "P_term",
            "P_link",
            "P_branch",
            "P_dead",
            "P_merge",
        ]
        if container[hypothesis].value
    ]
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
    hypothesis_model.relax = container.relax.value

    hypothesis_model.segmentation_miss_rate = container.segmentation_miss_rate.value

    return unscaled_config


def update_widgets_from_config(
    unscaled_config: UnscaledTrackerConfig,
    container: Container,
) -> Container:
    """
    Update the widgets in a container with the values in an
    UnscaledTrackerConfig.
    """

    sigmas: Sigmas = unscaled_config.sigmas
    container.P_sigma.value = sigmas.P
    container.G_sigma.value = sigmas.G
    container.R_sigma.value = sigmas.R

    config = unscaled_config.tracker_config
    container.update_method.value = config.update_method.name
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
    container.relax.value = hypothesis_model.relax

    container.segmentation_miss_rate.value = hypothesis_model.segmentation_miss_rate

    return container
