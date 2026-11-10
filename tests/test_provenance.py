import re

from ps_emu.provenance import source_identity


def test_source_identity_never_invents_a_commit():
    identity = source_identity()
    assert identity["package_name"] == "ps-emu"
    assert identity["package_version"]
    assert identity["source_kind"] in {"editable", "vcs_build", "wheel", "unversioned"}
    assert identity["source_commit"] is None or re.fullmatch(r"[0-9a-f]{40}", identity["source_commit"])
    assert identity["source_dirty"] is None or isinstance(identity["source_dirty"], bool)
    if identity["source_kind"] == "wheel":
        assert identity["source_commit"] is None
        assert identity["source_dirty"] is None
