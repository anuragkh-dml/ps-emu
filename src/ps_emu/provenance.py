"""Report verifiable package and source identity without guessing revisions."""

from __future__ import annotations

import json
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
import re
import subprocess


def source_identity() -> dict[str, str | bool | None]:
    """Return package version and any independently verifiable source commit.

    Editable installs are inspected in their loaded ``src/ps_emu`` checkout,
    including editable installs whose PEP 610 URL has a VCS scheme. PEP 610 VCS
    metadata is used for immutable VCS installs. Ordinary wheels correctly
    report no source SHA because the wheel alone cannot prove its source tree.
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
    dir_info = direct.get("dir_info", {})
    if dir_info.get("editable") is True:
        package = Path(__file__).resolve().parent
        root = package.parent.parent if package.parent.name == "src" else None
        dirty: bool | None = None
        verified = root is not None and (root / "src" / "ps_emu").resolve() == package
        if verified:
            try:
                top_level = subprocess.run(
                    ["git", "rev-parse", "--show-toplevel"], cwd=root, check=True,
                    capture_output=True, text=True,
                ).stdout.strip()
                if Path(top_level).resolve() != root.resolve():
                    verified = False
            except (OSError, subprocess.CalledProcessError):
                verified = False
        if verified:
            try:
                assert root is not None
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
                else:
                    commit = None
                    dirty = None
                for installed in package.rglob("*.py"):
                    source = root / "src" / "ps_emu" / installed.relative_to(package)
                    if not source.is_file() or source.read_bytes() != installed.read_bytes():
                        dirty = True
                        break
            except (OSError, subprocess.CalledProcessError):
                commit = None
                dirty = None
        else:
            commit = None
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
