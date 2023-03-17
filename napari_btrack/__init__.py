try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

from napari_btrack import track

__all__ = [
    "track",
]
