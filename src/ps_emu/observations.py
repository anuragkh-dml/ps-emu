"""Validate and deterministically generate bounded model-shaped observations."""

from __future__ import annotations

from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import validate_observation
from .profiles import ModelProfile, require_concrete_input_geometry

SCHEMA = "policy_server_emulator.observations.v1"
GENERATOR = "numpy_uniform_pool_v1"


def _digest(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return sha256(raw).hexdigest()


def load_observation_profile(path: Path | str) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {"schema", "generator", "seed", "source_rate_hz", "duration_s", "pool_size", "prompt"}
    if not isinstance(data, dict) or set(data) != required:
        raise ValueError("generated observations have unknown or missing fields")
    if data["schema"] != SCHEMA or data["generator"] != GENERATOR:
        raise ValueError("unsupported generated observation schema/generator")
    for key, minimum in (("seed", 0), ("pool_size", 2)):
        if type(data[key]) is not int or data[key] < minimum:
            raise ValueError(f"generated observations {key} must be an integer >= {minimum}")
    for key in ("source_rate_hz", "duration_s"):
        if (isinstance(data[key], bool) or not isinstance(data[key], (int, float))
                or not math.isfinite(data[key]) or data[key] <= 0):
            raise ValueError(f"generated observations {key} must be finite and positive")
    count = math.ceil(data["duration_s"] * data["source_rate_hz"])
    if count < 2 or count > 1_000_000 or data["pool_size"] > count:
        raise ValueError("generated observations require 2..1000000 frames and pool_size <= frames")
    if not isinstance(data["prompt"], str) or not data["prompt"].strip():
        raise ValueError("generated observation prompt must be nonempty")
    return data


def observation_pool_bytes(model: ModelProfile, pool_size: int) -> int:
    """Compute decoded storage and require a finite numeric model input shape."""
    if type(pool_size) is not int or pool_size < 1:
        raise ValueError("pool_size must be a positive integer")
    require_concrete_input_geometry(model)
    arrays = [f for f in model.data["observation"]["fields"] if f["kind"] == "ndarray"]
    if any(np.dtype(f["dtype"]).kind not in "biuf" for f in arrays):
        raise ValueError("generated observations support bool/integer/float arrays")
    return sum(math.prod(f["shape"]) * np.dtype(f["dtype"]).itemsize for f in arrays) * pool_size


def generate_observation_pool(
    model: ModelProfile,
    profile: dict[str, Any],
    client_id: str,
    *,
    max_decoded_bytes: int,
) -> tuple[tuple[dict[str, Any], ...], int, str]:
    """Generate a read-only deterministic pool for one logical client."""
    if not isinstance(client_id, str) or not client_id:
        raise ValueError("client_id must be nonempty")
    pool_bytes = observation_pool_bytes(model, profile["pool_size"])
    if type(max_decoded_bytes) is not int or max_decoded_bytes <= 0:
        raise ValueError("max_decoded_bytes must be positive")
    if pool_bytes > max_decoded_bytes:
        raise ValueError("generated observation pool exceeds decoded-memory budget")
    fields = model.data["observation"]["fields"]
    descriptor = {
        "schema": "policy_server_emulator.generated_stream.v1",
        "input_kind": "generated",
        "client_id": client_id,
        "observation_profile": profile,
        "observation_profile_sha256": _digest(profile),
        "model_profile_sha256": model.sha256,
        "input_signature": model.data["observation"],
        "decoded_pool_bytes": pool_bytes,
    }
    identity = _digest(descriptor)
    rng = np.random.default_rng(int(identity, 16))
    pool: list[dict[str, Any]] = []
    for _ in range(profile["pool_size"]):
        observation: dict[str, Any] = {}
        for field in fields:
            cursor = observation
            for part in field["path"][:-1]:
                cursor = cursor.setdefault(part, {})
            if field["kind"] == "string":
                value = profile["prompt"]
            else:
                dtype, shape = np.dtype(field["dtype"]), field["shape"]
                if dtype.kind in "bui":
                    maximum = 1 if dtype.kind == "b" else np.iinfo(dtype).max
                    value = rng.integers(0, maximum, size=shape, endpoint=True, dtype=dtype)
                else:
                    value = rng.uniform(-1, 1, size=shape).astype(dtype)
                value.flags.writeable = False
            cursor[field["path"][-1]] = value
        validate_observation(model, observation)
        pool.append(observation)
    return tuple(pool), pool_bytes, identity
