"""Generic interpreter for the declarative OpenPI model contract vocabulary."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from typing import Any

import numpy as np

from .profiles import ModelProfile


def _profile(model: ModelProfile | Mapping[str, Any]) -> Mapping[str, Any]:
    return model.data if isinstance(model, ModelProfile) else model


def _at(root: Any, path: Sequence[str]) -> Any:
    value = root
    for segment in path:
        if not isinstance(value, Mapping) or segment not in value:
            raise ValueError(f"observation is missing path {list(path)!r}")
        value = value[segment]
    return value


def _dtype_name(value: Any) -> str:
    try:
        return np.dtype(value.dtype).name
    except (AttributeError, TypeError):
        return ""


def _validate_observation_keys(
    profile_fields: Sequence[Mapping[str, Any]], observation: Mapping[str, Any]
) -> None:
    tree: dict[str, Any] = {}
    for field in profile_fields:
        cursor = tree
        for part in field["path"]:
            cursor = cursor.setdefault(part, {})
        cursor["__leaf__"] = True

    def descend(expected: Mapping[str, Any], value: Mapping[str, Any], path: str) -> None:
        allowed = set(expected) - {"__leaf__"}
        unknown = value.keys() - allowed
        if unknown:
            raise ValueError(f"unknown observation fields at {path or '<root>'}: {sorted(unknown)}")
        for key, child in expected.items():
            if key == "__leaf__" or key not in value:
                continue  # required field lookup below supplies the precise error
            if child == {"__leaf__": True}:
                continue
            if not isinstance(value[key], Mapping):
                raise ValueError(f"observation {path + key!r} must be a mapping")
            descend(child, value[key], path + key + ".")

    root = dict(tree)
    root.update(
        {
            key: {"__leaf__": True}
            for key in (
                "endpoint",
                "session_id",
                "request_id",
                "vllm_omni_cohort",
                "sampling_params",
            )
        }
    )
    descend(root, observation, "")


def validate_observation(
    model: ModelProfile | Mapping[str, Any],
    observation: Any,
    features: dict[str, Any] | None = None,
) -> None:
    data = _profile(model)
    if not isinstance(observation, Mapping):
        raise ValueError("OpenPI observation must be a mapping")
    _validate_observation_keys(data["observation"]["fields"], observation)
    for field in data["observation"]["fields"]:
        path = field["path"]
        value = _at(observation, path)
        if field["kind"] == "string":
            if not isinstance(value, str):
                raise ValueError(f"{path!r} must be a string")
            if field.get("nonempty_after_strip", False) and not value.strip():
                raise ValueError(f"{path!r} must be nonempty")
            continue
        if not isinstance(value, np.ndarray):
            raise ValueError(f"{path!r} must be a NumPy array")
        if _dtype_name(value) != field["dtype"]:
            raise ValueError(f"{path!r} must have dtype {field['dtype']}; got {_dtype_name(value)}")
        shape = field["shape"]
        if value.ndim != len(shape) or any(
            expected is not None and expected != got
            for expected, got in zip(shape, value.shape, strict=True)
        ):
            raise ValueError(f"{path!r} has invalid shape {value.shape}; expected {shape}")
        if any(dim <= 0 for dim in value.shape):
            raise ValueError(f"{path!r} dimensions must be nonempty")
        if field.get("finite", False) and not np.isfinite(value).all():
            raise ValueError(f"{path!r} contains nonfinite values")
    params = observation.get("sampling_params", {})
    if not isinstance(params, Mapping):
        raise ValueError("sampling_params must be a mapping")
    rtc_enabled = bool((features or {}).get("rtc_enabled", False))
    allowed = {item["name"] for item in data["sampling_parameters"]["fields"]}
    if rtc_enabled and "guided_rtc_v1" in data.get("features", {}):
        allowed.add("rtc")
    unknown = params.keys() - allowed
    if unknown:
        raise ValueError(f"unsupported sampling parameters: {sorted(unknown)}")
    for declaration in data["sampling_parameters"]["fields"]:
        name = declaration["name"]
        if name not in params:
            if declaration.get("required", False):
                raise ValueError(f"required sampling parameter {name!r} is missing")
            continue
        value, kind = params[name], declaration["kind"]
        valid = {
            "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
            "number": lambda v: (
                isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
            ),
            "boolean": lambda v: isinstance(v, bool),
            "string": lambda v: isinstance(v, str),
        }[kind](value)
        if not valid:
            raise ValueError(f"sampling parameter {name!r} must be {kind}")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if (
                "minimum" in declaration
                and value < declaration["minimum"]
                or "maximum" in declaration
                and value > declaration["maximum"]
            ):
                raise ValueError(f"sampling parameter {name!r} is outside its declared range")
        if "values" in declaration and value not in declaration["values"]:
            raise ValueError(f"sampling parameter {name!r} is outside its declared values")


def _overlay(target: dict[str, Any], overlay: Mapping[str, Any]) -> None:
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(target.get(key), dict):
            _overlay(target[key], value)
        else:
            target[key] = value


def handshake(
    model: ModelProfile | Mapping[str, Any], features: dict[str, Any] | None = None
) -> dict[str, Any]:
    data = _profile(model)
    result = json.loads(json.dumps(data["handshake"]))
    rtc = data.get("features", {}).get("guided_rtc_v1")
    enabled = bool((features or {}).get("rtc_enabled", False))
    if enabled:
        if rtc is None:
            raise ValueError("guided RTC requested but profile does not declare guided_rtc_v1")
        _overlay(result, rtc["handshake"])
    # Reserved emulator identity cannot be overridden by model profile content.
    marker = (features or {}).get("emulator_marker")
    if marker is not None:
        result["policy_server_emulator"] = marker
    return result


def _canonical(value: Any) -> bytes:
    if isinstance(value, np.ndarray):
        return value.dtype.str.encode() + repr(value.shape).encode() + value.tobytes(order="C")
    if isinstance(value, Mapping):
        return (
            b"{"
            + b"".join(str(k).encode() + b":" + _canonical(value[k]) for k in sorted(value))
            + b"}"
        )
    if isinstance(value, (list, tuple)):
        return b"[" + b";".join(_canonical(x) for x in value) + b"]"
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False, default=str
    ).encode()


def _seed_material(
    data: Mapping[str, Any], observation: Mapping[str, Any], identity: Mapping[str, Any]
) -> bytes:
    generator = data["generator"]
    parts: list[bytes] = []
    for key in generator["identity_inputs"]:
        parts.append(key.encode() + b"=" + _canonical(identity.get(key)))
    for path in generator["state_paths"]:
        parts.append(b"state=" + _canonical(_at(observation, path)))
    params = observation.get("sampling_params", {})
    for declaration in data["sampling_parameters"]["fields"]:
        name = declaration["name"]
        if declaration["affects_generation"] and name in params:
            parts.append(b"sampling=" + name.encode() + b":" + _canonical(params[name]))
    return hashlib.sha256(b"\0".join(parts)).digest()


def _array(
    shape: Sequence[int],
    dtype: str,
    *,
    seed: bytes,
    generator: Mapping[str, Any],
    tweak: Mapping[str, Any] | None = None,
) -> np.ndarray:
    params = dict(generator)
    if tweak:
        params.update(tweak)
    amplitude = float(params.get("amplitude", 0.0))
    offset = float(params.get("offset", 0.0))
    row_stride = float(params.get("row_stride", 0.0))
    dim_stride = float(params.get("dimension_stride", 0.0))
    entropy = int.from_bytes(seed[:8], "big") / (2**64 - 1) - 0.5
    arr = np.empty(tuple(shape), dtype=np.dtype(dtype))
    if arr.ndim == 1:
        for d in range(arr.shape[0]):
            arr[d] = offset + amplitude * entropy + dim_stride * d
    else:
        time_axis = int(params.get("time_axis", 0))
        moved = np.moveaxis(arr, time_axis, 0)
        flat_width = math.prod(moved.shape[1:])
        values = np.empty(moved.shape, dtype=np.float64)
        for row in range(moved.shape[0]):
            values[row] = (
                offset
                + amplitude * entropy
                + row_stride * row
                + dim_stride * np.arange(flat_width).reshape(moved.shape[1:])
            )
        arr = np.moveaxis(values.astype(dtype), 0, time_axis)
    if dtype.startswith("float") and not np.isfinite(arr).all():
        raise ValueError("generated action contains nonfinite values")
    return arr


def _build(
    node: Mapping[str, Any],
    seed: bytes,
    generator: Mapping[str, Any],
    path: str = "",
    generator_override: Mapping[str, Any] | None = None,
) -> Any:
    kind = node["kind"]
    if kind == "ndarray":
        tweak = {**(generator_override or {}), **node.get("generator", {})}
        return _array(
            node["shape"],
            node["dtype"],
            seed=hashlib.sha256(seed + path.encode()).digest(),
            generator={**generator, "time_axis": node["time_axis"]},
            tweak=tweak,
        )
    if kind == "mapping":
        return {
            key: _build(child, seed, generator, f"{path}/{key}", generator_override)
            for key, child in node["fields"].items()
        }
    raise ValueError(f"unsupported generated response node {kind!r}")


def _set_path(root: dict[str, Any], path: Sequence[str], value: Any) -> None:
    cursor = root
    for part in path[:-1]:
        cursor = cursor.setdefault(part, {})
    cursor[path[-1]] = value


def _get_path(root: Mapping[str, Any], path: Sequence[str]) -> Any:
    value: Any = root
    for part in path:
        value = value[part]
    return value


def _rtc_request(
    observation: Mapping[str, Any], descriptor: Mapping[str, Any]
) -> Mapping[str, Any] | None:
    raw = _get_path(observation, descriptor["request"]["path"])
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ValueError("RTC sampling_params.rtc must be a mapping")
    return raw


def _validate_rtc_request(raw: Mapping[str, Any], descriptor: Mapping[str, Any]) -> None:
    request = descriptor["request"]
    allowed = set(request["fields"])
    if set(raw) != allowed:
        raise ValueError(f"RTC request fields mismatch: expected {sorted(allowed)}")
    if raw["schema_version"] != 1 or raw["mode"] != "guided":
        raise ValueError("RTC schema_version/mode is unsupported")
    _positive_or_zero(raw["inference_delay"], "RTC inference_delay")
    _positive_or_zero(raw["execution_horizon"], "RTC execution_horizon")
    delay, horizon = raw["inference_delay"], raw["execution_horizon"]
    if delay > horizon or horizon > descriptor["action_horizon"]:
        raise ValueError("RTC timing must satisfy 0 <= delay <= horizon <= action_horizon")
    previous = raw["previous_actions"]
    expected = request["previous_actions_shape"]
    if (
        not isinstance(previous, np.ndarray)
        or previous.dtype.name != request["previous_actions_dtype"]
        or list(previous.shape) != expected
    ):
        raise ValueError("RTC previous_actions dtype/shape mismatch")
    if not np.isfinite(previous).all():
        raise ValueError("RTC previous_actions must be finite")
    if raw["prefix_attention_schedule"] not in descriptor["schedules"]:
        raise ValueError("RTC prefix_attention_schedule is not advertised")
    weight = raw["max_guidance_weight"]
    if (
        isinstance(weight, bool)
        or not isinstance(weight, (int, float))
        or not math.isfinite(weight)
        or weight < 0
    ):
        raise ValueError("RTC max_guidance_weight must be finite and nonnegative")
    correlation = raw["correlation"]
    if not isinstance(correlation, Mapping) or not correlation:
        raise ValueError("RTC correlation must be a nonempty mapping")
    if any(
        not isinstance(k, str) or not isinstance(v, (str, int, float, bool, type(None)))
        for k, v in correlation.items()
    ):
        raise ValueError("RTC correlation must contain scalar values")
    if any(isinstance(v, float) and not math.isfinite(v) for v in correlation.values()):
        raise ValueError("RTC correlation floats must be finite")


def _positive_or_zero(value: Any, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def make_actions(
    model: ModelProfile | Mapping[str, Any],
    observation: Mapping[str, Any],
    identity: Mapping[str, Any],
    features: dict[str, Any] | None = None,
) -> Any:
    data = _profile(model)
    validate_observation(model, observation, features)
    base_seed = _seed_material(data, observation, identity)
    result = _build(data["response"], base_seed, data["generator"])
    rtc = data.get("features", {}).get("guided_rtc_v1")
    enabled = bool((features or {}).get("rtc_enabled", False))
    if not enabled:
        return result
    if rtc is None:
        raise ValueError("guided RTC requested but profile has no guided_rtc_v1 descriptor")
    request = _rtc_request(observation, rtc)
    if request is None:
        raise ValueError("guided RTC enabled but request has no RTC parameters")
    _validate_rtc_request(request, rtc)
    canonical = result
    model_actions = _build(
        data["response"],
        hashlib.sha256(base_seed + b"model-space").digest(),
        data["generator"],
        "model-space",
        rtc["response"].get("model_space_generator"),
    )
    response: dict[str, Any] = {}
    _set_path(response, rtc["response"]["actions_path"], canonical)
    _set_path(response, rtc["response"]["model_actions_path"], model_actions)
    metadata = {
        "schema_version": rtc["response"]["schema_version"],
        "mode": rtc["response"]["mode"],
        "correlation": dict(request["correlation"]),
    }
    _set_path(response, rtc["response"]["metadata_path"], metadata)
    return response
