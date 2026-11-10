"""Strict loaders for portable model, hardware, and service profile documents."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate JSON key: {key}")
        out[key] = value
    return out


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _read(path: str | Path, schema: str) -> tuple[dict[str, Any], str, str, Path]:
    file = Path(path).resolve()
    raw = file.read_bytes()
    try:
        data = json.loads(
            raw,
            object_pairs_hook=_pairs,
            parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s)),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON profile {file}: {exc}") from exc
    if not isinstance(data, dict) or data.get("schema") != schema:
        raise ValueError(f"{file} must be an object with schema {schema!r}")
    return data, hashlib.sha256(_canonical(data)).hexdigest(), hashlib.sha256(raw).hexdigest(), file


def _keys(data: dict[str, Any], required: set[str], optional: set[str] = frozenset()) -> None:
    missing, extra = required - data.keys(), data.keys() - required - optional
    if missing or extra:
        raise ValueError(f"profile fields missing={sorted(missing)} unknown={sorted(extra)}")


def _positive_int(value: Any, name: str, *, allow_zero: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < (0 if allow_zero else 1):
        raise ValueError(f"{name} must be an integer >= {0 if allow_zero else 1}")


@dataclass(frozen=True)
class ModelProfile:
    data: dict[str, Any]
    sha256: str
    raw_sha256: str
    path: Path

    @property
    def model_id(self) -> str:
        return self.data["id"]


@dataclass(frozen=True)
class HardwareProfile:
    data: dict[str, Any]
    sha256: str
    raw_sha256: str
    path: Path


@dataclass(frozen=True)
class ServiceProfile:
    data: dict[str, Any]
    sha256: str
    raw_sha256: str
    path: Path


def _validate_shape(shape: Any, name: str, *, output: bool = False) -> None:
    if not isinstance(shape, list) or not shape:
        raise ValueError(f"{name}.shape must be a nonempty list")
    for dim in shape:
        if dim is None and not output:
            continue
        _positive_int(dim, f"{name}.shape dimension")


def _validate_response(node: Any, name: str = "response") -> None:
    if not isinstance(node, dict) or node.get("kind") not in {
        "mapping",
        "ndarray",
        "string",
        "scalar",
    }:
        raise ValueError(f"{name} has an unsupported response node")
    kind = node["kind"]
    if kind == "mapping":
        _keys(node, {"kind", "fields", "exact_keys"})
        if (
            not isinstance(node["fields"], dict)
            or not node["fields"]
            or not isinstance(node["exact_keys"], bool)
        ):
            raise ValueError(f"{name} mapping requires fields and exact_keys")
        for key, child in node["fields"].items():
            _validate_response(child, f"{name}.{key}")
    elif kind == "ndarray":
        _keys(node, {"kind", "dtype", "shape", "time_axis", "finite"}, {"generator"})
        _validate_shape(node["shape"], name, output=True)
        if node["dtype"] not in {"float32", "float64", "uint8", "int32", "int64"}:
            raise ValueError(f"{name} has unsupported dtype")
        axis = node["time_axis"]
        if (
            isinstance(axis, bool)
            or not isinstance(axis, int)
            or not 0 <= axis < len(node["shape"])
        ):
            raise ValueError(f"{name}.time_axis is invalid")
        if not isinstance(node["finite"], bool):
            raise ValueError(f"{name}.finite must be boolean")
    else:
        raise ValueError(f"{name}: only ndarray and mapping responses are implemented")


def load_model_profile(path: str | Path) -> ModelProfile:
    data, digest, raw_digest, file = _read(path, "policy_server_emulator.model.v1")
    _keys(
        data,
        {
            "schema",
            "id",
            "handshake",
            "observation",
            "response",
            "generator",
            "sampling_parameters",
            "features",
        },
        {"checkpoint_revision"},
    )
    if not isinstance(data["id"], str) or not data["id"]:
        raise ValueError("model id must be a nonempty string")
    handshake = data["handshake"]
    if not isinstance(handshake, dict) or "action_horizon" not in handshake:
        raise ValueError("handshake must declare action_horizon")
    if "policy_server_emulator" in handshake:
        raise ValueError("model profile cannot define the reserved emulator handshake marker")
    _positive_int(handshake["action_horizon"], "handshake.action_horizon")
    observation = data["observation"]
    if (
        not isinstance(observation, dict)
        or set(observation) != {"fields"}
        or not isinstance(observation["fields"], list)
    ):
        raise ValueError("observation must contain fields list")
    seen_paths: set[tuple[str, ...]] = set()
    for field in observation["fields"]:
        if not isinstance(field, dict):
            raise ValueError("observation fields must be objects")
        _keys(field, {"path", "kind"}, {"dtype", "shape", "finite", "nonempty_after_strip"})
        path_parts = field["path"]
        if (
            not isinstance(path_parts, list)
            or not path_parts
            or any(not isinstance(part, str) or not part for part in path_parts)
        ):
            raise ValueError("observation path must be nonempty literal string segments")
        normalized = tuple(path_parts)
        if normalized in seen_paths:
            raise ValueError(f"duplicate observation path: {path_parts!r}")
        seen_paths.add(normalized)
        if field["kind"] == "ndarray":
            _validate_shape(field.get("shape"), f"observation {path_parts}")
            if field.get("dtype") not in {"float32", "float64", "uint8", "int32", "int64"}:
                raise ValueError(f"unsupported observation dtype at {path_parts}")
        elif field["kind"] == "string":
            if set(field) - {"path", "kind", "nonempty_after_strip"}:
                raise ValueError("string observation field has invalid options")
        else:
            raise ValueError(f"unsupported observation kind {field['kind']!r}")
    _validate_response(data["response"])

    def check_horizon(node: dict[str, Any], name: str = "response") -> None:
        if node["kind"] == "mapping":
            for key, child in node["fields"].items():
                check_horizon(child, f"{name}.{key}")
        elif node["kind"] == "ndarray":
            if node["shape"][node["time_axis"]] != handshake["action_horizon"]:
                raise ValueError(f"{name} time-axis dimension must match action_horizon")

    check_horizon(data["response"])
    generator = data["generator"]
    if not isinstance(generator, dict) or generator.get("kind") != "deterministic_pattern_v1":
        raise ValueError("unsupported generator kind")
    _keys(
        generator,
        {
            "kind",
            "identity_inputs",
            "state_paths",
            "amplitude",
            "offset",
            "row_stride",
            "dimension_stride",
        },
    )
    if not isinstance(generator.get("identity_inputs"), list) or not isinstance(
        generator.get("state_paths"), list
    ):
        raise ValueError("generator identity_inputs and state_paths must be lists")
    for key in ("amplitude", "offset", "row_stride", "dimension_stride"):
        value = generator[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(f"generator.{key} must be finite numeric")
    sampling = data["sampling_parameters"]
    if not isinstance(sampling, dict):
        raise ValueError("sampling_parameters must be an object")
    _keys(sampling, {"fields", "unknown_fields"})
    if sampling["unknown_fields"] != "reject" or not isinstance(sampling["fields"], list):
        raise ValueError("sampling_parameters unknown_fields must be reject and fields a list")
    names: set[str] = set()
    for param in sampling["fields"]:
        if not isinstance(param, dict):
            raise ValueError("sampling parameter declarations must be objects")
        _keys(
            param,
            {"name", "kind", "affects_generation"},
            {"minimum", "maximum", "values", "required"},
        )
        if not isinstance(param["name"], str) or not param["name"] or param["name"] in names:
            raise ValueError("sampling parameter names must be unique nonempty strings")
        names.add(param["name"])
        if param["kind"] not in {"integer", "number", "boolean", "string"} or not isinstance(
            param["affects_generation"], bool
        ):
            raise ValueError(f"sampling parameter {param['name']} has invalid type/effect")
        if "required" in param and not isinstance(param["required"], bool):
            raise ValueError("sampling parameter required must be boolean")
        if "minimum" in param and (
            isinstance(param["minimum"], bool) or not isinstance(param["minimum"], (int, float))
        ):
            raise ValueError("sampling parameter minimum must be numeric")
        if "maximum" in param and (
            isinstance(param["maximum"], bool) or not isinstance(param["maximum"], (int, float))
        ):
            raise ValueError("sampling parameter maximum must be numeric")
        if "values" in param and (not isinstance(param["values"], list) or not param["values"]):
            raise ValueError("sampling parameter values must be a nonempty list")
    if not isinstance(data["features"], dict):
        raise ValueError("features must be an object")
    rtc = data["features"].get("guided_rtc_v1")
    if rtc is not None:
        _validate_rtc(data, rtc)
    return ModelProfile(data, digest, raw_digest, file)


def _validate_rtc(model: dict[str, Any], rtc: Any) -> None:
    if not isinstance(rtc, dict):
        raise ValueError("guided_rtc_v1 must be an object")
    required = {"handshake", "request", "response", "schedules", "action_horizon", "action_dim"}
    _keys(rtc, required)
    if not isinstance(rtc["schedules"], list) or not rtc["schedules"]:
        raise ValueError("RTC schedules must be a nonempty list")
    _positive_int(rtc["action_horizon"], "RTC action_horizon")
    _positive_int(rtc["action_dim"], "RTC action_dim")
    if rtc["action_horizon"] != model["handshake"]["action_horizon"]:
        raise ValueError("RTC action horizon must agree with handshake")
    if rtc["action_dim"] != model["handshake"].get("action_dim"):
        raise ValueError("RTC action dimension must agree with handshake")
    request = rtc["request"]
    if not isinstance(request, dict) or request.get("path") != ["sampling_params", "rtc"]:
        raise ValueError("RTC request path must be sampling_params.rtc")
    if (
        request.get("previous_actions_shape") != [rtc["action_horizon"], rtc["action_dim"]]
        or request.get("previous_actions_dtype") != "float32"
    ):
        raise ValueError("RTC previous action contract must match the action dimensions")
    if set(request.get("fields", [])) != {
        "schema_version",
        "mode",
        "previous_actions",
        "inference_delay",
        "execution_horizon",
        "prefix_attention_schedule",
        "max_guidance_weight",
        "correlation",
    }:
        raise ValueError("RTC request fields are unsupported")
    response = rtc["response"]
    _keys(
        response,
        {
            "actions_path",
            "model_actions_path",
            "metadata_path",
            "schema_version",
            "mode",
            "model_space_generator",
        },
    )
    if (
        response["schema_version"] != 1
        or response["mode"] != "guided"
        or any(
            not isinstance(response[k], list) or not response[k]
            for k in ("actions_path", "model_actions_path", "metadata_path")
        )
    ):
        raise ValueError("RTC response contract is unsupported")
    if not isinstance(response["model_space_generator"], dict):
        raise ValueError("RTC model_space_generator must be an object")


def load_hardware_profile(path: str | Path) -> HardwareProfile:
    data, digest, raw_digest, file = _read(path, "policy_server_emulator.hardware.v1")
    _keys(data, {"schema", "id", "device_kind", "device_count"}, {"device_name", "attributes"})
    if (
        not isinstance(data["id"], str)
        or not data["id"]
        or not isinstance(data["device_kind"], str)
    ):
        raise ValueError("hardware id and device_kind must be strings")
    _positive_int(data["device_count"], "device_count")
    if "attributes" in data and not isinstance(data["attributes"], dict):
        raise ValueError("hardware attributes must be an object")
    return HardwareProfile(data, digest, raw_digest, file)


def require_concrete_input_geometry(model: ModelProfile) -> None:
    """Reject wildcard input shapes when binding a measured service profile.

    Service durations depend on the actual input payload geometry. Wildcards are
    useful for fixed synthetic correctness profiles, but cannot demonstrate
    that measured timings cover every shape admitted by the contract.
    """
    for field in model.data["observation"]["fields"]:
        if field["kind"] == "ndarray" and any(dim is None for dim in field["shape"]):
            raise ValueError(
                "empirical service profiles require concrete ndarray input geometry; "
                f"model field {field['path']!r} contains wildcard dimensions"
            )


def load_service_profile(path: str | Path) -> ServiceProfile:
    data, digest, raw_digest, file = _read(path, "policy_server_emulator.service_profile.v1")
    kind = data.get("kind")
    common = {"schema", "kind", "source_kind", "binding", "timing_basis", "units"}
    if kind == "fixed":
        _keys(data, common | {"active_ns_by_batch_size"})
        if data["source_kind"] != "synthetic_fixture" or data["binding"] not in {
            "synthetic_unbound",
            "profile_digests",
        }:
            raise ValueError("fixed service profiles must be explicitly synthetic")
        if data["timing_basis"] != "scheduler_active_interval" or data["units"] != "ns":
            raise ValueError("fixed timings must use scheduler_active_interval nanoseconds")
        timings = data["active_ns_by_batch_size"]
        if not isinstance(timings, dict) or not timings:
            raise ValueError("fixed profile needs batch-size timings")
        for size, active in timings.items():
            if not re.fullmatch(r"[1-9][0-9]*", size):
                raise ValueError("batch size keys must be positive decimal strings")
            _positive_int(active, f"active_ns_by_batch_size[{size}]")
    elif kind == "empirical":
        required = common | {
            "model",
            "model_profile_sha256",
            "checkpoint_revision",
            "source_server_revision",
            "hardware_profile_sha256",
            "hardware_id",
            "source_configuration_sha256",
            "source_route_configuration_sha256",
            "source_execution_projection",
            "sources",
            "samples_by_batch_size",
        }
        _keys(data, required)
        if (
            data["source_kind"] != "measured"
            or data["binding"] != "measured"
            or data["timing_basis"] != "scheduler_active_interval"
            or data["units"] != "ns"
        ):
            raise ValueError(
                "empirical profiles must be measured scheduler_active_interval nanoseconds"
            )
        for key in (
            "model_profile_sha256",
            "hardware_profile_sha256",
            "source_configuration_sha256",
            "source_route_configuration_sha256",
        ):
            if not isinstance(data[key], str) or not re.fullmatch(r"[0-9a-f]{64}", data[key]):
                raise ValueError(f"{key} must be a SHA-256 hex digest")
        if not re.fullmatch(r"[0-9a-f]{40}", data["source_server_revision"]):
            raise ValueError("source_server_revision must be a full commit SHA")
        projection = data["source_execution_projection"]
        if not isinstance(projection, dict):
            raise ValueError("source_execution_projection must be an object")
        projection_keys = {
            "model",
            "checkpoint_revision",
            "server_revision",
            "deploy_config_sha256",
            "configured_max_batch_size",
            "configured_batch_wait_ms",
            "rtc_enabled",
            "sampling_parameters",
            "input_signature",
        }
        _keys(projection, projection_keys)
        if (
            not isinstance(projection["model"], str)
            or not projection["model"]
            or not isinstance(projection["checkpoint_revision"], str)
            or not projection["checkpoint_revision"]
        ):
            raise ValueError(
                "source execution projection model/checkpoint must be nonempty strings"
            )
        if not re.fullmatch(r"[0-9a-f]{40}", projection["server_revision"]):
            raise ValueError("source execution server_revision must be a full commit SHA")
        if not re.fullmatch(r"[0-9a-f]{64}", projection["deploy_config_sha256"]):
            raise ValueError("source execution deploy_config_sha256 must be SHA-256")
        _positive_int(
            projection["configured_max_batch_size"], "projection.configured_max_batch_size"
        )
        _positive_int(
            projection["configured_batch_wait_ms"],
            "projection.configured_batch_wait_ms",
            allow_zero=True,
        )
        if (
            not isinstance(projection["rtc_enabled"], bool)
            or not isinstance(projection["sampling_parameters"], dict)
            or not isinstance(projection["input_signature"], dict)
        ):
            raise ValueError("source execution projection has invalid types")
        if (
            hashlib.sha256(_canonical(projection)).hexdigest()
            != data["source_configuration_sha256"]
        ):
            raise ValueError(
                "source_configuration_sha256 does not match canonical execution projection"
            )
        if not isinstance(data["sources"], list) or not data["sources"]:
            raise ValueError("empirical profile requires source receipts")
        for src in data["sources"]:
            _keys(
                src,
                {"run_id", "receipt_sha256", "joined_requests_sha256"},
                {"excluded_rows", "incomplete_batches", "hardware_evidence_sha256"},
            )
            if not isinstance(src["run_id"], str) or not src["run_id"]:
                raise ValueError("source receipt fields must be nonempty strings")
            for key in ("receipt_sha256", "joined_requests_sha256", "hardware_evidence_sha256"):
                if key in src and (
                    not isinstance(src[key], str) or not re.fullmatch(r"[0-9a-f]{64}", src[key])
                ):
                    raise ValueError(f"source {key} must be SHA-256")
            if "excluded_rows" in src:
                if not isinstance(src["excluded_rows"], dict):
                    raise ValueError("excluded_rows must be a count mapping")
                for count in src["excluded_rows"].values():
                    _positive_int(count, "excluded row count", allow_zero=True)
            if "incomplete_batches" in src:
                _positive_int(src["incomplete_batches"], "incomplete_batches", allow_zero=True)
        batches = data["samples_by_batch_size"]
        if not isinstance(batches, dict) or not batches:
            raise ValueError("empirical profile requires samples")
        for size, samples in batches.items():
            if (
                not re.fullmatch(r"[1-9][0-9]*", size)
                or not isinstance(samples, list)
                or not samples
            ):
                raise ValueError("empirical batch entries must be nonempty lists")
            for sample in samples:
                _keys(sample, {"sample_id", "active_ns"})
                if not isinstance(sample["sample_id"], str) or not sample["sample_id"]:
                    raise ValueError("sample_id must be a nonempty string")
                _positive_int(sample["active_ns"], "sample.active_ns")
    else:
        raise ValueError("service profile kind must be fixed or empirical")
    return ServiceProfile(data, digest, raw_digest, file)


@dataclass(frozen=True)
class ServiceSample:
    sample_id: str
    active_ns: int


def choose_sample(
    profile: ServiceProfile,
    *,
    batch_size: int,
    per_size_sequence: int,
    seed: int,
    sampling: str | None = None,
) -> ServiceSample:
    _positive_int(batch_size, "batch_size")
    _positive_int(per_size_sequence, "per_size_sequence", allow_zero=True)
    _positive_int(seed, "seed", allow_zero=True)
    key = str(batch_size)
    if profile.data["kind"] == "fixed":
        if sampling not in (None, "fixed"):
            raise ValueError("fixed service profiles require sampling='fixed'")
        try:
            return ServiceSample(
                f"fixed-b{batch_size}", profile.data["active_ns_by_batch_size"][key]
            )
        except KeyError as exc:
            raise ValueError(f"service profile has no batch size {batch_size}") from exc
    try:
        samples = profile.data["samples_by_batch_size"][key]
    except KeyError as exc:
        raise ValueError(f"service profile has no batch size {batch_size}") from exc
    if sampling not in (None, "independent_v1"):
        raise ValueError(f"unsupported service sampling mode {sampling!r}")
    material = f"{seed}:{batch_size}:{per_size_sequence}".encode()
    index = int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % len(samples)
    selected = samples[index]
    return ServiceSample(selected["sample_id"], selected["active_ns"])
