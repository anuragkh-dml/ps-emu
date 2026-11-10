"""Access portable example configuration shipped with the distribution."""

from importlib.resources import files
from pathlib import Path


def example_profile(name: str) -> Path:
    """Return an installed profile path by profile-relative POSIX name."""
    if not name or name.startswith("/") or ".." in Path(name).parts:
        raise ValueError("profile name must be a nonempty relative path")
    resource = files("ps_emu").joinpath("profiles", *Path(name).parts)
    path = Path(str(resource))
    if not path.is_file():
        raise FileNotFoundError(f"unknown example profile: {name}")
    return path
