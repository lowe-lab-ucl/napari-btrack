from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from btrack.config import TrackerConfig

import copy
import os
from dataclasses import dataclass, field

import btrack
import numpy as np
from btrack import datasets

__all__ = [
    "create_default_configs",
]


@dataclass
class Sigmas:
    """Values to scale TrackerConfig MotionModel matrices by"""

    P: float
    G: float
    R: float


@dataclass
class UnscaledTrackerConfig:
    """Convert TrackerConfig MotionModel matrices from scaled to unscaled.

    This is needed because TrackerConfig stores "scaled" matrices, i.e. it
    doesn't store sigma and the "unscaled" MotionModel matrices separately.
    """

    filename: os.PathLike
    tracker_config: TrackerConfig = field(init=False)
    sigmas: Sigmas = field(init=False)

    def __post_init__(self):
        """Create the TrackerConfig and un-scale the MotionModel indices"""

        config = btrack.config.load_config(self.filename)
        self.tracker_config, self.sigmas = self._unscale_config(config)

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
        scaled_config = copy.deepcopy(self.tracker_config)
        scaled_config.motion_model.P *= self.sigmas.P
        scaled_config.motion_model.R *= self.sigmas.R
        scaled_config.motion_model.G *= self.sigmas.G
        scaled_config.motion_model.Q = (
            scaled_config.motion_model.G.T @ scaled_config.motion_model.G
        )

        return scaled_config


@dataclass
class TrackerConfigs:
    configs: dict[str, UnscaledTrackerConfig] = field(default_factory=dict)
    current_config: str = field(init=False)

    def __post_init__(self):
        """Add the default cell and particle configs."""

        _ = self.add_config(
            filename=datasets.cell_config(),
            name="cell",
            overwrite=False,
        )
        _ = self.add_config(
            filename=datasets.particle_config(),
            name="particle",
            overwrite=False,
        )

        self.current_config = "cell"

    def __getitem__(self, config_name):
        return self.configs[config_name]

    def add_config(
        self,
        filename,
        overwrite,
        name=None,
    ) -> str:
        """Load a TrackerConfig and add it to the store."""

        config = UnscaledTrackerConfig(filename)
        config_name = config.tracker_config.name if name is None else name
        config.tracker_config.name = config_name

        # TODO: Make the combobox editable so config names can be changed within the GUI
        if config_name in self.configs and not overwrite:
            _msg = (
                f"Config '{config_name}' already exists - config names must be unique."
            )
            raise ValueError(_msg)

        self.configs[config_name] = config

        return config_name


def create_default_configs():
    """Create a set of default configurations"""

    # TrackerConfigs automatically loads default cell and particle configs
    return TrackerConfigs()
