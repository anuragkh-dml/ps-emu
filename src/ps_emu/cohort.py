"""Validation and lifecycle tracking for vLLM-Omni cohort wire metadata."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

COHORT_FIELD = "vllm_omni_cohort"
COHORT_SCHEMA = "vllm_omni.cohort.v1"


def membership_hash(member_ids: list[str]) -> str:
    payload = json.dumps(
        sorted(member_ids), ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 256:
        raise ValueError(f"cohort {name} must be a nonempty string of at most 256 characters")
    return value


def validate_cohort(
    payload: Any, *, request_id: str, policy_id: str, max_batch_size: int
) -> dict[str, Any]:
    if not isinstance(payload, dict) or payload.get("schema") != COHORT_SCHEMA:
        raise ValueError("unsupported cohort schema")
    epoch = _identifier(payload.get("epoch_id"), "epoch_id")
    policy = _identifier(payload.get("policy_id"), "policy_id")
    if policy != policy_id:
        raise ValueError("cohort policy_id does not match configured policy")
    members = payload.get("member_ids")
    if (
        not isinstance(members, list)
        or not members
        or len(members) > max_batch_size
        or any(not isinstance(item, str) or not item for item in members)
        or members != sorted(set(members))
    ):
        raise ValueError("cohort member_ids must be sorted unique identifiers within capacity")
    digest = payload.get("membership_hash")
    if digest != membership_hash(members):
        raise ValueError("cohort membership_hash mismatch")
    member = _identifier(payload.get("member_id"), "member_id")
    if member not in members:
        raise ValueError("cohort member_id is not in membership")
    if payload.get("request_id") != request_id:
        raise ValueError("cohort request_id does not match request")
    deadline = payload.get("deadline_ns")
    if not isinstance(deadline, int) or isinstance(deadline, bool) or deadline <= 0:
        raise ValueError("cohort deadline_ns must be a positive integer")
    return {
        "epoch_id": epoch,
        "member_ids": list(members),
        "membership_hash": digest,
        "deadline_ns": deadline,
        "policy_id": policy,
        "member_id": member,
        "request_id": request_id,
    }


@dataclass
class CohortRegistry:
    """Reject conflicting/reopened epochs and track observed members."""

    contracts: dict[str, tuple[Any, ...]] = field(default_factory=dict)
    seen: dict[str, set[str]] = field(default_factory=dict)
    closed: set[str] = field(default_factory=set)
    max_registered_epochs: int = 65536
    last_epoch: str | None = None
    epoch_order: list[str] = field(default_factory=list)
    predecessor: dict[str, str | None] = field(default_factory=dict)

    def register(self, cohort: dict[str, Any]) -> None:
        epoch = cohort["epoch_id"]
        if epoch in self.closed:
            raise ValueError("cohort epoch is already closed")
        is_new = epoch not in self.contracts
        if not is_new and epoch != self.last_epoch:
            raise ValueError("cohort epoch continuation is noncontiguous")
        if is_new and len(self.contracts) >= self.max_registered_epochs:
            raise ValueError("cohort registered-epoch capacity exceeded")
        contract = (
            tuple(cohort["member_ids"]),
            cohort["membership_hash"],
            cohort["deadline_ns"],
            cohort["policy_id"],
        )
        if not is_new and self.contracts[epoch] != contract:
            raise ValueError("conflicting cohort contract for epoch")
        members = self.seen.setdefault(epoch, set())
        if cohort["member_id"] in members:
            raise ValueError("duplicate cohort member")
        if is_new:
            self.contracts[epoch] = contract
            self.predecessor[epoch] = self.last_epoch
            self.epoch_order.append(epoch)
        self.last_epoch = epoch
        members.add(cohort["member_id"])

    def unregister(self, cohort: dict[str, Any]) -> None:
        epoch = cohort["epoch_id"]
        members = self.seen.get(epoch)
        if members is not None and epoch not in self.closed:
            members.discard(cohort["member_id"])
            if not members:
                self.seen.pop(epoch, None)
                self.contracts.pop(epoch, None)
                previous = self.predecessor.pop(epoch, None)
                if epoch in self.epoch_order:
                    self.epoch_order.remove(epoch)
                if self.last_epoch == epoch:
                    self.last_epoch = previous

    def register_uncoordinated(self) -> None:
        # The uncoordinated request is an ordering barrier. An epoch may not
        # resume after it, even if some declared members have not arrived.
        self.last_epoch = None

    def close(self, epoch_id: str) -> None:
        self.closed.add(epoch_id)
