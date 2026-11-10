from __future__ import annotations

import pytest

from ps_emu.cohort import CohortRegistry, COHORT_SCHEMA, membership_hash, validate_cohort


def _payload(member: str = "a", request: str = "r") -> dict[str, object]:
    members = ["a", "b"]
    return {
        "schema": COHORT_SCHEMA,
        "epoch_id": "epoch-1",
        "member_ids": members,
        "membership_hash": membership_hash(members),
        "deadline_ns": 1000,
        "policy_id": "policy",
        "member_id": member,
        "request_id": request,
    }


def test_cohort_validates_membership_and_request_identity() -> None:
    parsed = validate_cohort(_payload(), request_id="r", policy_id="policy", max_batch_size=4)
    assert parsed["member_id"] == "a"
    invalid = _payload()
    invalid["membership_hash"] = "bad"
    with pytest.raises(ValueError, match="membership_hash"):
        validate_cohort(invalid, request_id="r", policy_id="policy", max_batch_size=4)
    with pytest.raises(ValueError, match="request_id"):
        validate_cohort(_payload(), request_id="other", policy_id="policy", max_batch_size=4)


def test_registry_rejects_duplicate_conflict_and_reopen() -> None:
    cohort = validate_cohort(_payload(), request_id="r", policy_id="policy", max_batch_size=4)
    registry = CohortRegistry()
    registry.register(cohort)
    with pytest.raises(ValueError, match="duplicate"):
        registry.register(cohort)
    changed = dict(cohort, deadline_ns=2000, member_id="b", request_id="r2")
    with pytest.raises(ValueError, match="conflicting"):
        registry.register(changed)
    registry.close("epoch-1")
    with pytest.raises(ValueError, match="closed"):
        registry.register(changed)


def test_registry_caps_retained_epoch_history_without_eviction() -> None:
    registry = CohortRegistry(max_registered_epochs=1)
    first = validate_cohort(_payload(), request_id="r", policy_id="policy", max_batch_size=4)
    registry.register(first)
    registry.close("epoch-1")
    second = dict(first, epoch_id="epoch-2", member_id="a", request_id="r2")
    with pytest.raises(ValueError, match="capacity"):
        registry.register(second)


def test_registry_tracks_production_epoch_continuity_and_uncoordinated_barrier() -> None:
    registry = CohortRegistry()
    epoch_one_a = validate_cohort(
        _payload("a", "r1"), request_id="r1", policy_id="policy", max_batch_size=4
    )
    epoch_one_b = validate_cohort(
        _payload("b", "r2"), request_id="r2", policy_id="policy", max_batch_size=4
    )
    epoch_two_a = dict(epoch_one_a, epoch_id="epoch-2", request_id="r3")
    epoch_two_b = dict(epoch_one_b, epoch_id="epoch-2", request_id="r4")
    registry.register(epoch_one_a)
    registry.register(epoch_two_a)
    with pytest.raises(ValueError, match="noncontiguous"):
        registry.register(epoch_one_b)
    registry.register(epoch_two_b)
    registry.close("epoch-2")

    fresh = CohortRegistry()
    fresh.register(epoch_one_a)
    fresh.register_uncoordinated()
    with pytest.raises(ValueError, match="noncontiguous"):
        fresh.register(epoch_one_b)
