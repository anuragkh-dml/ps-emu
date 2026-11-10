"""External, coarse residual overhead budgets; service and queue remain separate."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from statistics import median
from typing import Any

from .profiles import _keys, _positive_int, _read


SCHEMA = "policy_server_emulator.overhead_profile.v1"


@dataclass(frozen=True)
class OverheadProfile:
    data: dict[str, Any]
    sha256: str
    raw_sha256: str
    path: Path


def residual_budget(source: list[dict], local: list[dict], field: str) -> int:
    return max(0, round(median(s[field] for s in source) - median(s[field] for s in local)))


def load_overhead_profile(path: Path | str) -> OverheadProfile:
    data, digest, raw, file = _read(path, SCHEMA)
    _keys(
        data,
        {
            "schema",
            "method",
            "timing_basis",
            "service_profile_sha256",
            "model_profile_sha256",
            "hardware_profile_sha256",
            "source_run_ids",
            "local_run_ids",
            "input_inventory",
            "samples_by_size",
            "before_queue_ns_by_connection_count",
            "after_service_ns_by_batch_size",
        },
    )
    if (
        data["method"] != "nonnegative_median_difference_v1"
        or data["timing_basis"] != "pre_active_excluding_queue_and_post_active_completion"
    ):
        raise ValueError("unsupported coarse overhead method/basis")
    for key in ("service_profile_sha256", "model_profile_sha256", "hardware_profile_sha256"):
        if not isinstance(data[key], str) or not re.fullmatch(r"[0-9a-f]{64}", data[key]):
            raise ValueError(f"invalid overhead {key}")
    for key in ("source_run_ids", "local_run_ids"):
        if (
            not isinstance(data[key], list)
            or not data[key]
            or any(not isinstance(s, str) or not s for s in data[key])
            or len(set(data[key])) != len(data[key])
        ):
            raise ValueError(f"overhead {key} requires distinct run IDs")
    if set(data["source_run_ids"]) & set(data["local_run_ids"]):
        raise ValueError("overhead source and local run IDs must be disjoint")
    if (
        not isinstance(data["input_inventory"], dict)
        or not data["input_inventory"]
        or any(
            not isinstance(p, str) or not isinstance(h, str) or not re.fullmatch(r"[0-9a-f]{64}", h)
            for p, h in data["input_inventory"].items()
        )
    ):
        raise ValueError("overhead input inventory requires file digests")
    sizes = data["samples_by_size"]
    if (
        not isinstance(sizes, dict)
        or not sizes
        or any(not k.isdigit() or int(k) < 1 for k in sizes)
    ):
        raise ValueError("overhead samples require positive sizes")
    for key in ["before_queue_ns_by_connection_count", "after_service_ns_by_batch_size"]:
        if not isinstance(data[key], dict) or set(data[key]) != set(sizes):
            raise ValueError("overhead budgets must cover every sampled size")
        for n in data[key].values():
            _positive_int(n, key, allow_zero=True)
    for size, groups in sizes.items():
        if not isinstance(groups, dict):
            raise ValueError("overhead sample groups must be objects")
        _keys(groups, {"source", "local"})
        for kind, samples in groups.items():
            if not isinstance(samples, list) or len(samples) < 4:
                raise ValueError(
                    "coarse overhead requires at least four complete request samples per group"
                )
            identities = []
            for sample in samples:
                if not isinstance(sample, dict):
                    raise ValueError("overhead samples must be objects")
                _keys(
                    sample, {"run_id", "request_id", "batch_id", "pre_queue_ns", "post_service_ns"}
                )
                if (
                    sample["run_id"]
                    not in data["source_run_ids" if kind == "source" else "local_run_ids"]
                ):
                    raise ValueError("overhead sample run identity differs")
                if (
                    not isinstance(sample["request_id"], str)
                    or not sample["request_id"]
                    or not isinstance(sample["batch_id"], str)
                    or not sample["batch_id"]
                ):
                    raise ValueError("overhead sample request/batch identity invalid")
                for field in ["pre_queue_ns", "post_service_ns"]:
                    _positive_int(sample[field], field, allow_zero=True)
                identities.append((sample["run_id"], sample["request_id"]))
            if len(set(identities)) != len(identities):
                raise ValueError("duplicate overhead request sample")
        for field, budget in [
            ("pre_queue_ns", "before_queue_ns_by_connection_count"),
            ("post_service_ns", "after_service_ns_by_batch_size"),
        ]:
            if data[budget][size] != residual_budget(groups["source"], groups["local"], field):
                raise ValueError("overhead budget differs from source/local median derivation")
    return OverheadProfile(data, digest, raw, file)
