"""Execution-profile composition and whole-configuration validation.

Fault rules selecting the same request may combine reply/service/status delays;
delay application is ``max_once_v1``. Duplicate selectors involving a
disconnect, protocol error, or invalid action shape are rejected because their
outcomes depend on which incompatible fault wins.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import tomllib
from typing import Any

from .profiles import (
    HardwareProfile,
    ModelProfile,
    ServiceProfile,
    load_hardware_profile,
    load_model_profile,
    load_service_profile,
    require_concrete_input_geometry,
)
from .overheads import OverheadProfile, load_overhead_profile


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


@dataclass(frozen=True)
class ResolvedEmulatorConfig:
    data: dict[str, Any]
    model: ModelProfile
    hardware: HardwareProfile
    service_profile: ServiceProfile
    fault_profile: dict[str, Any] | None
    sha256: str
    fault_raw_sha256: str | None = None
    overhead_profile: OverheadProfile | None = None

    def snapshot(self) -> dict[str, Any]:
        result = {
            "schema": "policy_server_emulator.snapshot.v1",
            "configuration": self.data,
            "configuration_sha256": self.sha256,
            "fault_profile": self.fault_profile,
            "profiles": {
                "model": {
                    "path": str(self.model.path),
                    "sha256": self.model.sha256,
                    "raw_sha256": self.model.raw_sha256,
                    "data": self.model.data,
                },
                "hardware": {
                    "path": str(self.hardware.path),
                    "sha256": self.hardware.sha256,
                    "raw_sha256": self.hardware.raw_sha256,
                    "data": self.hardware.data,
                },
                "service": {
                    "path": str(self.service_profile.path),
                    "sha256": self.service_profile.sha256,
                    "raw_sha256": self.service_profile.raw_sha256,
                    "data": self.service_profile.data,
                },
            },
        }
        if self.fault_profile is not None:
            result["profiles"]["fault"] = {
                "path": self.data.get("fault_profile"),
                "sha256": hashlib.sha256(_canonical(self.fault_profile)).hexdigest(),
                "raw_sha256": self.fault_raw_sha256,
                "data": self.fault_profile,
            }
        if self.overhead_profile is not None:
            profile = self.overhead_profile
            result["profiles"]["overhead"] = {
                "path": str(profile.path),
                "sha256": profile.sha256,
                "raw_sha256": profile.raw_sha256,
                "data": profile.data,
            }
        return result


def _require_table(data: Any, name: str) -> dict[str, Any]:
    if not isinstance(data, dict) or any(not isinstance(k, str) for k in data):
        raise ValueError(f"{name} must be a table")
    return data


def _path(base: Path, raw: Any, name: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"{name} must be a nonempty path string")
    candidate = Path(raw)
    return (candidate if candidate.is_absolute() else base / candidate).resolve()


def _check_fields(
    table: dict[str, Any], allowed: set[str], name: str, required: set[str] | None = None
) -> None:
    required = allowed if required is None else required
    missing, unknown = required - table.keys(), table.keys() - allowed
    if missing or unknown:
        raise ValueError(f"{name} missing={sorted(missing)} unknown={sorted(unknown)}")


def _integer(value: Any, name: str, minimum: int = 1) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _validate_execution(data: dict[str, Any]) -> None:
    _check_fields(
        data,
        {
            "schema",
            "model_profile",
            "hardware_profile",
            "service_profile",
            "fault_profile",
            "overhead_profile",
            "transport",
            "scheduler",
            "service",
            "features",
            "cohort",
            "lifecycle",
            "evidence",
        },
        "execution",
        {
            "schema",
            "model_profile",
            "hardware_profile",
            "service_profile",
            "transport",
            "scheduler",
            "service",
            "features",
            "cohort",
            "lifecycle",
            "evidence",
        },
    )
    if data["schema"] != "policy_server_emulator.execution.v1":
        raise ValueError("unsupported execution schema")
    for name, allowed, required in [
        (
            "transport",
            {
                "host",
                "port",
                "path",
                "compression",
                "ping_interval_s",
                "max_connections",
                "max_frame_bytes",
                "receive_queue_capacity",
                "idle_timeout_s",
                "max_sessions_per_connection",
            },
            {
                "host",
                "port",
                "path",
                "compression",
                "ping_interval_s",
                "max_connections",
                "max_frame_bytes",
                "receive_queue_capacity",
                "idle_timeout_s",
            },
        ),
        (
            "scheduler",
            {
                "policy",
                "max_pending_requests",
                "max_pending_bytes",
                "max_batch_size",
                "batch_wait_ms",
                "active_batches",
                "cancellation_policy",
            },
            {
                "policy",
                "max_pending_requests",
                "max_pending_bytes",
                "max_batch_size",
                "batch_wait_ms",
                "active_batches",
                "cancellation_policy",
            },
        ),
        ("service", {"sampling", "seed", "scale"}, {"sampling", "seed", "scale"}),
        ("features", {"rtc_enabled", "cohort_enabled"}, {"rtc_enabled", "cohort_enabled"}),
        (
            "cohort",
            {"wire_schema", "policy_id", "max_registered_epochs"},
            {"wire_schema", "policy_id"},
        ),
        (
            "lifecycle",
            {"readiness_timeout_s", "shutdown_grace_s"},
            {"readiness_timeout_s", "shutdown_grace_s"},
        ),
        (
            "evidence",
            {
                "status_interval_ms",
                "recent_event_capacity",
                "writer_queue_capacity",
                "flush_interval_ms",
            },
            {
                "status_interval_ms",
                "recent_event_capacity",
                "writer_queue_capacity",
                "flush_interval_ms",
            },
        ),
    ]:
        table = _require_table(data.get(name), name)
        _check_fields(table, allowed, name, required)
    data["transport"].setdefault("max_sessions_per_connection", 1024)
    data["cohort"].setdefault("max_registered_epochs", 65536)
    transport, scheduler = data["transport"], data["scheduler"]
    if (
        not isinstance(transport["host"], str)
        or not transport["host"]
        or not isinstance(transport["path"], str)
        or not transport["path"].startswith("/")
    ):
        raise ValueError("transport host/path invalid")
    _integer(transport["port"], "transport.port", 0)
    if transport["port"] > 65535:
        raise ValueError("transport.port must be at most 65535")
    if transport["compression"] != "none":
        raise ValueError("only uncompressed OpenPI WebSocket frames are supported")
    for key in ("ping_interval_s", "idle_timeout_s"):
        value = transport[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"transport.{key} must be positive finite seconds")
    _integer(transport["max_connections"], "transport.max_connections")
    _integer(transport["max_frame_bytes"], "transport.max_frame_bytes")
    _integer(transport["receive_queue_capacity"], "transport.receive_queue_capacity")
    _integer(transport["max_sessions_per_connection"], "transport.max_sessions_per_connection")
    for key in ("max_pending_requests", "max_pending_bytes", "max_batch_size", "active_batches"):
        _integer(scheduler[key], f"scheduler.{key}")
    _integer(scheduler["batch_wait_ms"], "scheduler.batch_wait_ms", 0)
    if (
        scheduler["policy"] != "fifo_collect_v1"
        or scheduler["cancellation_policy"] != "finish_active_discard_reply_v1"
        or scheduler["active_batches"] != 1
    ):
        raise ValueError("unsupported v1 scheduler or cancellation behavior")
    if scheduler["max_batch_size"] > scheduler["max_pending_requests"]:
        raise ValueError("max_batch_size cannot exceed max_pending_requests")
    if data["features"]["rtc_enabled"] not in (True, False) or data["features"][
        "cohort_enabled"
    ] not in (True, False):
        raise ValueError("feature flags must be booleans")
    if not isinstance(data["features"]["rtc_enabled"], bool) or not isinstance(
        data["features"]["cohort_enabled"], bool
    ):
        raise ValueError("feature flags must be booleans")
    cohort = data["cohort"]
    _integer(cohort["max_registered_epochs"], "cohort.max_registered_epochs")
    if cohort["wire_schema"] != "vllm_omni.cohort.v1" or not isinstance(cohort["policy_id"], str):
        raise ValueError("unsupported cohort protocol configuration")
    if data["features"]["cohort_enabled"] and not cohort["policy_id"]:
        raise ValueError("cohort policy_id is required when cohort support is enabled")
    _integer(data["service"]["seed"], "service.seed", 0)
    if data["service"]["sampling"] not in {"independent_v1", "fixed"}:
        raise ValueError("unsupported service sampling policy")
    scale = data["service"]["scale"]
    if (
        isinstance(scale, bool)
        or not isinstance(scale, (int, float))
        or not math.isfinite(scale)
        or scale <= 0
    ):
        raise ValueError("service.scale must be positive finite numeric")
    for name, key in (("lifecycle", "readiness_timeout_s"), ("lifecycle", "shutdown_grace_s")):
        value = data[name][key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"{name}.{key} must be positive finite seconds")
    for name, key in (
        ("evidence", "status_interval_ms"),
        ("evidence", "recent_event_capacity"),
        ("evidence", "writer_queue_capacity"),
        ("evidence", "flush_interval_ms"),
    ):
        _integer(data[name][key], f"{name}.{key}")


def resolve_config(
    path: str | Path, overrides: dict[str, Any] | None = None
) -> ResolvedEmulatorConfig:
    file = Path(path).resolve()
    try:
        data = tomllib.loads(file.read_text())
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ValueError(f"cannot load execution config {file}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("execution config must be a table")
    data = json.loads(json.dumps(data))
    if overrides:
        for key, value in overrides.items():
            if key in {
                "model_profile",
                "hardware_profile",
                "service_profile",
                "fault_profile",
                "overhead_profile",
            }:
                data[key] = str(Path(value).resolve())
            elif key in {
                "transport",
                "scheduler",
                "service",
                "features",
                "cohort",
                "lifecycle",
                "evidence",
            } and isinstance(value, dict):
                data.setdefault(key, {}).update(value)
            else:
                raise ValueError(f"unsupported execution override {key!r}")
    _validate_execution(data)
    model = load_model_profile(_path(file.parent, data["model_profile"], "model_profile"))
    hardware = load_hardware_profile(
        _path(file.parent, data["hardware_profile"], "hardware_profile")
    )
    service = load_service_profile(_path(file.parent, data["service_profile"], "service_profile"))
    expected_sampling = "fixed" if service.data["kind"] == "fixed" else "independent_v1"
    if data["service"]["sampling"] != expected_sampling:
        raise ValueError(
            f"service profile kind={service.data['kind']} requires sampling={expected_sampling}"
        )
    fault: dict[str, Any] | None = None
    fault_raw_hash: str | None = None
    if data.get("fault_profile"):
        fault_path = _path(file.parent, data["fault_profile"], "fault_profile")
        try:
            raw_fault = fault_path.read_bytes()
            fault = tomllib.loads(raw_fault.decode())
            fault_raw_hash = hashlib.sha256(raw_fault).hexdigest()
        except (OSError, tomllib.TOMLDecodeError) as exc:
            raise ValueError(f"cannot load fault profile {fault_path}: {exc}") from exc
        if not isinstance(fault, dict) or fault.get("schema") != "policy_server_emulator.faults.v1":
            raise ValueError("invalid fault profile schema")
        _check_fields(fault, {"schema", "service_delay_combination", "rules"}, "fault profile")
        if fault["service_delay_combination"] != "max_once_v1" or not isinstance(
            fault["rules"], list
        ):
            raise ValueError("unsupported fault profile semantics")
        for index, rule in enumerate(fault["rules"]):
            if not isinstance(rule, dict):
                raise ValueError(f"fault rule {index} must be a table")
            allowed = {"session_id", "request_id", "request_sequence", "kind", "delay_ms"}
            _check_fields(rule, allowed, f"fault rule {index}", {"kind"})
            if rule["kind"] not in {
                "reply_delay",
                "service_delay",
                "status_freeze",
                "disconnect",
                "protocol_error",
                "invalid_action_shape",
            }:
                raise ValueError(f"fault rule {index} has unsupported kind")
            for key in ("session_id", "request_id"):
                if key in rule and (not isinstance(rule[key], str) or not rule[key]):
                    raise ValueError(f"fault rule {index} {key} must be nonempty")
            if "request_sequence" in rule:
                _integer(rule["request_sequence"], f"fault rule {index}.request_sequence", 0)
            if rule["kind"] in {"reply_delay", "service_delay", "status_freeze"}:
                _integer(
                    rule.get("delay_ms"),
                    f"fault rule {index}.delay_ms",
                    1 if rule["kind"] == "status_freeze" else 0,
                )
            elif "delay_ms" in rule:
                raise ValueError(f"fault rule {index} does not accept delay_ms")
        exclusive = {"disconnect", "protocol_error", "invalid_action_shape"}
        seen_selectors: dict[tuple[Any, ...], str] = {}
        for index, rule in enumerate(fault["rules"]):
            selector = (
                rule.get("session_id", "*"),
                rule.get("request_id", "*"),
                rule.get("request_sequence", "*"),
            )
            previous = seen_selectors.get(selector)
            if previous in exclusive or rule["kind"] in exclusive:
                if previous is not None:
                    raise ValueError(
                        f"fault rules with selector {selector!r} conflict: {previous} and {rule['kind']}"
                    )
            seen_selectors[selector] = rule["kind"]
    max_batch = data["scheduler"]["max_batch_size"]
    overhead = None
    if data.get("overhead_profile"):
        overhead = load_overhead_profile(
            _path(file.parent, data["overhead_profile"], "overhead_profile")
        )
        for key, expected in [
            ("model_profile_sha256", model.sha256),
            ("hardware_profile_sha256", hardware.sha256),
            ("service_profile_sha256", service.sha256),
        ]:
            if overhead.data[key] != expected:
                raise ValueError(f"overhead {key} binding differs")
        if not {str(n) for n in range(1, max_batch + 1)} <= set(
            overhead.data["after_service_ns_by_batch_size"]
        ):
            raise ValueError("overhead batch coverage differs from configured capacity")
        if any(
            value > data["transport"]["idle_timeout_s"] * 1e9
            for key in ["before_queue_ns_by_connection_count", "after_service_ns_by_batch_size"]
            for value in overhead.data[key].values()
        ):
            raise ValueError("overhead delay exceeds transport idle timeout")
    timings = service.data.get(
        "active_ns_by_batch_size", service.data.get("samples_by_batch_size", {})
    )
    missing = [str(n) for n in range(1, max_batch + 1) if str(n) not in timings]
    if missing:
        raise ValueError(f"service profile lacks configured batch sizes: {missing}")
    if service.data["kind"] == "empirical":
        if hardware.data["device_kind"] == "synthetic":
            raise ValueError("empirical service requires observed hardware identity")
        require_concrete_input_geometry(model)
        if (
            service.data["model"] != model.model_id
            or service.data["model_profile_sha256"] != model.sha256
        ):
            raise ValueError("measured profile model identity/digest does not match")
        if (
            service.data["hardware_id"] != hardware.data["id"]
            or service.data["hardware_profile_sha256"] != hardware.sha256
        ):
            raise ValueError("measured profile hardware identity/digest does not match")
        projection = service.data["source_execution_projection"]
        if (
            projection["model"] != model.model_id
            or projection["server_revision"] != service.data["source_server_revision"]
        ):
            raise ValueError("measured profile source execution model/server identity mismatch")
        if projection["checkpoint_revision"] != service.data["checkpoint_revision"]:
            raise ValueError("measured profile checkpoint identity mismatch")
        if model.data.get("checkpoint_revision") != projection["checkpoint_revision"]:
            raise ValueError("source checkpoint does not match the selected model profile")
        expected_input = {
            "observation": model.data["observation"],
            "response": model.data["response"],
        }
        if (
            projection["input_signature"] != expected_input
            or projection["sampling_parameters"] != model.data["sampling_parameters"]
        ):
            raise ValueError(
                "measured profile source input/sampling signature does not match the model contract"
            )
        if projection["rtc_enabled"] != data["features"]["rtc_enabled"]:
            raise ValueError("measured profile RTC mode does not match execution features")
    if (
        data["features"]["cohort_enabled"]
        and data["cohort"]["wire_schema"] != "vllm_omni.cohort.v1"
    ):
        raise ValueError("unsupported cohort wire schema")
    if data["features"]["rtc_enabled"]:
        rtc = model.data["features"]["guided_rtc_v1"]
        if model.data["handshake"].get("action_dim") != rtc["action_dim"]:
            raise ValueError("RTC action dimension does not match model handshake")

    def output_bytes(node: dict[str, Any]) -> int:
        if node["kind"] == "ndarray":
            import math as _math
            import numpy as np

            return _math.prod(node["shape"]) * np.dtype(node["dtype"]).itemsize
        if node["kind"] == "mapping":
            return sum(
                len(key.encode()) + output_bytes(child) for key, child in node["fields"].items()
            )
        return 0

    maximum_reply = output_bytes(model.data["response"])
    if data["features"]["rtc_enabled"]:
        maximum_reply = maximum_reply * 2 + 1024
    if maximum_reply > data["transport"]["max_frame_bytes"]:
        raise ValueError("model action response exceeds transport.max_frame_bytes")
    if data["features"]["rtc_enabled"] and "guided_rtc_v1" not in model.data["features"]:
        raise ValueError("RTC enabled but model profile does not declare guided_rtc_v1")
    normalized = {
        **data,
        "model_profile": str(model.path),
        "hardware_profile": str(hardware.path),
        "service_profile": str(service.path),
        "execution_file": str(file),
    }
    if fault is not None:
        normalized["fault_profile"] = str(
            _path(file.parent, data["fault_profile"], "fault_profile")
        )
    if overhead is not None:
        normalized["overhead_profile"] = str(overhead.path)
    digest = hashlib.sha256(
        _canonical(
            {
                "execution": normalized,
                "model": model.sha256,
                "hardware": hardware.sha256,
                "service": service.sha256,
                "fault": fault,
                **({"overhead": overhead.sha256} if overhead is not None else {}),
            }
        )
    ).hexdigest()
    return ResolvedEmulatorConfig(
        normalized, model, hardware, service, fault, digest, fault_raw_hash, overhead
    )
