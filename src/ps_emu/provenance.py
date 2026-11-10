"""Report verifiable package and source identity without guessing revisions."""

from __future__ import annotations

import json
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote, urlsplit


def source_identity() -> dict[str, str | bool | None]:
    """Return package version and any independently verifiable source commit.

    Editable installs are inspected in their checkout. PEP 610 VCS metadata is
    used for immutable VCS installs. Ordinary wheels correctly report no source
    SHA because the wheel alone cannot prove which source tree produced it.
    """
    try:
        dist = distribution("ps-emu")
        version = dist.version
        direct = json.loads(dist.read_text("direct_url.json") or "{}")
    except (PackageNotFoundError, ValueError, json.JSONDecodeError):
        return {
            "package_name": "ps-emu",
            "package_version": None,
            "source_commit": None,
            "source_dirty": None,
            "source_kind": "unversioned",
        }

    vcs = direct.get("vcs_info", {})
    commit = vcs.get("commit_id")
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        commit = None
    url = urlsplit(direct.get("url", ""))
    dir_info = direct.get("dir_info", {})
    root: Path | None = None
    if url.scheme == "file" and dir_info.get("editable") is True:
        root = Path(unquote(url.path))
    if root is not None:
        package = Path(__file__).resolve().parent
        dirty: bool | None = None
        try:
            found = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=root, check=True,
                capture_output=True, text=True,
            ).stdout.strip()
            status = subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=normal"],
                cwd=root, check=True, capture_output=True, text=True,
            ).stdout
            dirty = bool(status)
            if re.fullmatch(r"[0-9a-f]{40}", found):
                commit = found
            for installed in package.rglob("*.py"):
                source = root / "src" / "ps_emu" / installed.relative_to(package)
                if not source.is_file() or source.read_bytes() != installed.read_bytes():
                    dirty = True
                    break
        except (OSError, subprocess.CalledProcessError):
            dirty = None
        return {
            "package_name": "ps-emu",
            "package_version": version,
            "source_commit": commit,
            "source_dirty": dirty,
            "source_kind": "editable",
        }
    if commit:
        kind = "vcs_build"
        dirty = False
    else:
        kind = "wheel"
        dirty = None
    return {
        "package_name": "ps-emu",
        "package_version": version,
        "source_commit": commit,
        "source_dirty": dirty,
        "source_kind": kind,
    }
