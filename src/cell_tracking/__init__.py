"""CSE488 Cell Segmentation & Tracking utilities."""

from importlib.metadata import version, PackageNotFoundError

try:  # pragma: no cover - metadata only available after install
    __version__ = version("cse488-cell-tracking")
except PackageNotFoundError:  # pragma: no cover - local editable install
    __version__ = "0.0.0"
