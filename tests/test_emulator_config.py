from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ps_emu.config import resolve_config
from ps_emu.profiles import load_hardware_profile, load_model_profile


from ps_emu.resources import example_profile

CONFIG = example_profile("server.toml")
ROOT = CONFIG.parent


def test_resolve_config_paths_relative_to_execution_file_and_snapshot() -> None:
    resolved = resolve_config(CONFIG)
    assert resolved.model.model_id == "pi05"
    assert resolved.hardware.data["id"] == "synthetic"
    assert resolved.service_profile.data["active_ns_by_batch_size"]["4"] > 0
    assert len(resolved.sha256) == 64
    snap = resolved.snapshot()
    assert snap["configuration_sha256"] == resolved.sha256
    assert snap["profiles"]["model"]["sha256"] == resolved.model.sha256


def test_overrides_are_applied_before_digest_and_validation() -> None:
    resolved = resolve_config(
        CONFIG, {"scheduler": {"max_batch_size": 4}, "transport": {"port": 0}}
    )
    assert resolved.data["scheduler"]["max_batch_size"] == 4
    assert resolved.data["transport"]["port"] == 0
    with pytest.raises(ValueError, match="lacks configured batch"):
        resolve_config(CONFIG, {"scheduler": {"max_batch_size": 5}})


def test_unknown_execution_keys_and_unsupported_scheduler_are_rejected(tmp_path: Path) -> None:
    text = CONFIG.read_text().replace("max_batch_size = 1", "max_batch_size = 1\nparallelism = 2")
    bad = tmp_path / "bad.toml"
    # Make referenced configs absolute because the copied execution file has a different directory.
    text = (
        text.replace(
            '"models/pi05.json"', f'"{example_profile("models/pi05.json")}"'
        )
        .replace(
            '"hardware/synthetic.json"',
            f'"{example_profile("hardware/synthetic.json")}"',
        )
        .replace(
            '"service/fixed.json"',
            f'"{example_profile("service/fixed.json")}"',
        )
    )
    bad.write_text(text)
    with pytest.raises(ValueError, match="unknown"):
        resolve_config(bad)
    with pytest.raises(ValueError, match="unsupported v1 scheduler"):
        resolve_config(CONFIG, {"scheduler": {"active_batches": 2}})


@pytest.mark.parametrize(
    ("section", "name"),
    [("transport", "max_sessions_per_connection"), ("cohort", "max_registered_epochs")],
)
def test_resource_caps_must_be_positive(section: str, name: str) -> None:
    assert resolve_config(CONFIG).data[section][name] > 0
    with pytest.raises(ValueError, match=f"{section}.{name}"):
        resolve_config(CONFIG, {section: {name: 0}})


def test_resource_cap_defaults_keep_older_execution_profiles_readable(tmp_path: Path) -> None:
    original = CONFIG.read_text()
    stripped = "\n".join(
        line
        for line in original.splitlines()
        if not line.startswith(("max_sessions_per_connection =", "max_registered_epochs ="))
    )
    for rel, absolute in (
        ('"models/pi05.json"', f'"{example_profile("models/pi05.json")}"'),
        (
            '"hardware/synthetic.json"',
            f'"{example_profile("hardware/synthetic.json")}"',
        ),
        ('"service/fixed.json"', f'"{example_profile("service/fixed.json")}"'),
    ):
        stripped = stripped.replace(rel, absolute)
    legacy = tmp_path / "legacy.toml"
    legacy.write_text(stripped)
    resolved = resolve_config(legacy)
    assert resolved.data["transport"]["max_sessions_per_connection"] == 1024
    assert resolved.data["cohort"]["max_registered_epochs"] == 65536


def test_status_freeze_fault_profile_is_external_and_positive_duration() -> None:
    profile = example_profile("faults/status_freeze.toml")
    resolved = resolve_config(CONFIG, {"fault_profile": profile})
    assert resolved.fault_profile is not None
    assert resolved.fault_profile["rules"][0]["kind"] == "status_freeze"
    assert resolved.fault_profile["rules"][0]["delay_ms"] > 0


def test_same_selector_exclusive_fault_rules_are_rejected(tmp_path: Path) -> None:
    fault = tmp_path / "conflicting.toml"
    fault.write_text("""schema = "policy_server_emulator.faults.v1"
service_delay_combination = "max_once_v1"
[[rules]]
session_id = "same"
request_sequence = 1
kind = "protocol_error"
[[rules]]
session_id = "same"
request_sequence = 1
kind = "disconnect"
""")
    with pytest.raises(ValueError, match="conflict"):
        resolve_config(CONFIG, {"fault_profile": fault})


def test_empirical_binding_requires_concrete_and_matching_input_signature(tmp_path: Path) -> None:
    model_data = json.loads((example_profile("models/pi05.json")).read_text())
    for field in model_data["observation"]["fields"]:
        if field["kind"] == "ndarray":
            field["shape"] = [224 if dim is None else dim for dim in field["shape"]]
    model_path = tmp_path / "concrete-model.json"
    model_path.write_text(json.dumps(model_data))
    model = load_model_profile(model_path)
    hardware_path = tmp_path / "observed-hardware.json"
    hardware_path.write_text(
        json.dumps(
            {
                "schema": "policy_server_emulator.hardware.v1",
                "id": "fixture-gpu",
                "device_kind": "gpu",
                "device_name": "Test evidence-format GPU",
                "device_count": 1,
            }
        )
    )
    hardware = load_hardware_profile(hardware_path)
    projection = {
        "model": model.model_id,
        "checkpoint_revision": model.data["checkpoint_revision"],
        "server_revision": "b" * 40,
        "deploy_config_sha256": "c" * 64,
        "configured_max_batch_size": 1,
        "configured_batch_wait_ms": 0,
        "rtc_enabled": False,
        "sampling_parameters": model.data["sampling_parameters"],
        "input_signature": {
            "observation": model.data["observation"],
            "response": model.data["response"],
        },
    }
    canonical = lambda obj: json.dumps(
        obj, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    projection_hash = hashlib.sha256(canonical(projection)).hexdigest()
    service = {
        "schema": "policy_server_emulator.service_profile.v1",
        "kind": "empirical",
        "source_kind": "measured",
        "binding": "measured",
        "timing_basis": "scheduler_active_interval",
        "units": "ns",
        "model": model.model_id,
        "model_profile_sha256": model.sha256,
        "checkpoint_revision": model.data["checkpoint_revision"],
        "source_server_revision": "b" * 40,
        "hardware_profile_sha256": hardware.sha256,
        "hardware_id": hardware.data["id"],
        "source_configuration_sha256": projection_hash,
        "source_route_configuration_sha256": "d" * 64,
        "source_execution_projection": projection,
        "sources": [
            {"run_id": "source", "receipt_sha256": "e" * 64, "joined_requests_sha256": "f" * 64}
        ],
        "samples_by_batch_size": {
            str(size): [{"sample_id": f"batch-{size}", "active_ns": 1000 + size}]
            for size in range(1, 5)
        },
    }
    service_path = tmp_path / "empirical-service.json"
    service_path.write_text(json.dumps(service))
    empirical_overrides = {
        "model_profile": model_path,
        "hardware_profile": hardware_path,
        "service_profile": service_path,
        "service": {"sampling": "independent_v1"},
    }
    resolved = resolve_config(CONFIG, empirical_overrides)
    assert resolved.service_profile.data["kind"] == "empirical"

    wildcard_data = json.loads(
        (example_profile("models/pi05.json")).read_text()
    )
    wildcard_path = tmp_path / "wildcard-model.json"
    wildcard_path.write_text(json.dumps(wildcard_data))
    wildcard_model = load_model_profile(wildcard_path)
    wildcard_service = dict(service)
    wildcard_service["model_profile_sha256"] = wildcard_model.sha256
    wildcard_service["source_execution_projection"] = {
        **projection,
        "input_signature": {
            "observation": wildcard_model.data["observation"],
            "response": wildcard_model.data["response"],
        },
    }
    wildcard_service["source_configuration_sha256"] = hashlib.sha256(
        canonical(wildcard_service["source_execution_projection"])
    ).hexdigest()
    service_path.write_text(json.dumps(wildcard_service))
    with pytest.raises(ValueError, match="concrete ndarray input geometry"):
        resolve_config(CONFIG, {**empirical_overrides, "model_profile": wildcard_path})

    wrong_signature = dict(service)
    wrong_signature["source_execution_projection"] = {
        **projection,
        "input_signature": {"observation": {}, "response": {}},
    }
    wrong_signature["source_configuration_sha256"] = hashlib.sha256(
        canonical(wrong_signature["source_execution_projection"])
    ).hexdigest()
    service_path.write_text(json.dumps(wrong_signature))
    with pytest.raises(ValueError, match="source input/sampling signature"):
        resolve_config(CONFIG, empirical_overrides)
