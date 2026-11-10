import json
from pathlib import Path
import re
from types import SimpleNamespace

import ps_emu.provenance as provenance


SHA = "a" * 40


def _metadata(monkeypatch, direct_url):
    dist = SimpleNamespace(version="0.1.0", read_text=lambda name: json.dumps(direct_url))
    monkeypatch.setattr(provenance, "distribution", lambda _name: dist)


def _completed(stdout):
    return SimpleNamespace(stdout=stdout)


def test_source_identity_never_invents_a_commit():
    identity = provenance.source_identity()
    assert identity["package_name"] == "ps-emu"
    assert identity["package_version"]
    assert identity["source_kind"] in {"editable", "vcs_build", "wheel", "unversioned"}
    assert identity["source_commit"] is None or re.fullmatch(r"[0-9a-f]{40}", identity["source_commit"])
    assert identity["source_dirty"] is None or isinstance(identity["source_dirty"], bool)
    if identity["source_kind"] == "wheel":
        assert identity["source_commit"] is None
        assert identity["source_dirty"] is None


def test_editable_vcs_checkout_reports_git_head_and_dirty_state(tmp_path, monkeypatch):
    root = tmp_path / "checkout"
    package = root / "src" / "ps_emu"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("# package\n")
    (package / "provenance.py").write_text("# loaded code\n")
    monkeypatch.setattr(provenance, "__file__", str(package / "provenance.py"))
    _metadata(monkeypatch, {
        "url": "git+https://example.invalid/ps-emu.git",
        "vcs_info": {"vcs": "git", "commit_id": "b" * 40},
        "dir_info": {"editable": True},
    })

    def run(command, *, cwd, **_kwargs):
        assert Path(cwd) == root
        if command[-1] == "--show-toplevel":
            return _completed(str(root))
        if command[-1] == "HEAD":
            return _completed(SHA)
        if command[-1] == "--untracked-files=normal":
            return _completed(" M src/ps_emu/provenance.py\n")
        raise AssertionError(command)

    monkeypatch.setattr(provenance.subprocess, "run", run)
    identity = provenance.source_identity()
    assert identity["source_kind"] == "editable"
    assert identity["source_commit"] == SHA
    assert identity["source_dirty"] is True


def test_editable_without_verifiable_checkout_reports_unknown(tmp_path, monkeypatch):
    package = tmp_path / "site-packages" / "ps_emu"
    package.mkdir(parents=True)
    loaded = package / "provenance.py"
    loaded.touch()
    monkeypatch.setattr(provenance, "__file__", str(loaded))
    _metadata(monkeypatch, {
        "url": "git+https://example.invalid/ps-emu.git",
        "vcs_info": {"vcs": "git", "commit_id": "b" * 40},
        "dir_info": {"editable": True},
    })
    identity = provenance.source_identity()
    assert identity["source_kind"] == "editable"
    assert identity["source_commit"] is None
    assert identity["source_dirty"] is None


def test_editable_invalid_git_head_clears_commit_and_dirty(tmp_path, monkeypatch):
    root = tmp_path / "checkout"
    package = root / "src" / "ps_emu"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("# package\n")
    (package / "provenance.py").write_text("# loaded code\n")
    monkeypatch.setattr(provenance, "__file__", str(package / "provenance.py"))
    _metadata(monkeypatch, {
        "url": "git+https://example.invalid/ps-emu.git",
        "vcs_info": {"vcs": "git", "commit_id": "b" * 40},
        "dir_info": {"editable": True},
    })

    def run(command, *, cwd, **_kwargs):
        if command[-1] == "--show-toplevel":
            return _completed(str(root))
        if command[-1] == "HEAD":
            return _completed("not-a-sha")
        if command[-1] == "--untracked-files=normal":
            return _completed("")
        raise AssertionError(command)

    monkeypatch.setattr(provenance.subprocess, "run", run)
    identity = provenance.source_identity()
    assert identity["source_commit"] is None
    assert identity["source_dirty"] is None


def test_vcs_install_uses_pep610_commit_and_clean_state(monkeypatch):
    _metadata(monkeypatch, {
        "url": "git+https://example.invalid/ps-emu.git",
        "vcs_info": {"vcs": "git", "commit_id": SHA},
    })
    identity = provenance.source_identity()
    assert identity["source_kind"] == "vcs_build"
    assert identity["source_commit"] == SHA
    assert identity["source_dirty"] is False


def test_ordinary_wheel_does_not_claim_commit_or_clean_state(monkeypatch):
    _metadata(monkeypatch, {"url": "https://files.invalid/ps_emu.whl"})
    identity = provenance.source_identity()
    assert identity["source_kind"] == "wheel"
    assert identity["source_commit"] is None
    assert identity["source_dirty"] is None
