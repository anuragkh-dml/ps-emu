"""Check fixed wire and registry references using the standalone implementation."""

import json
from pathlib import Path

import numpy as np
import pytest

from ps_emu.cohort import CohortRegistry, membership_hash
from ps_emu.protocol import unpack

FIXTURE = Path(__file__).parent / "fixtures/policy_server_emulator/protocol.json"


def test_fixed_wire_fixture_decodes_with_emulator_codec():
    fixture = json.loads(FIXTURE.read_text())
    assert len(fixture["reference_revision"]) == 40
    for case in fixture["frames"]:
        expected = np.array(case["values"], dtype=case["dtype"])
        actual = unpack(bytes.fromhex(case["hex"]))
        assert actual.shape == tuple(case["shape"])
        assert actual.dtype == expected.dtype
        np.testing.assert_array_equal(actual, expected)


def test_membership_hash_matches_fixed_reference_for_reordered_members():
    for case in json.loads(FIXTURE.read_text())["membership"]:
        assert membership_hash(case["member_ids"]) == case["sha256"]


def test_registry_sequences_match_pinned_production_contract():
    for case in json.loads(FIXTURE.read_text())["registry_sequences"]:
        registry = CohortRegistry()
        for step, accepted in zip(case["steps"], case["accepted"], strict=True):
            def execute():
                if step.get("uncoordinated"):
                    registry.register_uncoordinated()
                else:
                    registry.register({
                        "epoch_id": step["epoch"], "member_id": step["member"],
                        "member_ids": ["a", "b"], "membership_hash": "fixed",
                        "deadline_ns": 1000, "policy_id": "policy",
                    })
            if accepted:
                execute()
            else:
                with pytest.raises(ValueError):
                    execute()
