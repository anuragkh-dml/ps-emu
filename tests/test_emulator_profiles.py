from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ps_emu.profiles import (
    choose_sample,
    load_hardware_profile,
    load_model_profile,
    load_service_profile,
    require_concrete_input_geometry,
)


from ps_emu.resources import example_profile

CONFIG = Path(example_profile("server.toml")).parent


def test_external_model_and_hardware_profiles_load_and_hash() -> None:
    model = load_model_profile(CONFIG / "models/pi05.json")
    hardware = load_hardware_profile(CONFIG / "hardware/synthetic.json")
    assert model.model_id == "pi05"
    assert model.sha256 != model.raw_sha256
    assert len(model.sha256) == 64
    assert hardware.data["id"] == "synthetic"
    assert hardware.sha256


def test_duplicate_keys_and_unknown_fields_are_rejected(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        '{"schema":"policy_server_emulator.hardware.v1","id":"a","id":"b","device_kind":"gpu","device_count":1}'
    )
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_hardware_profile(duplicate)
    source = json.loads((CONFIG / "hardware/synthetic.json").read_text())
    source["gpu_guess"] = 42
    invalid = tmp_path / "invalid.json"
    invalid.write_text(json.dumps(source))
    with pytest.raises(ValueError, match="unknown"):
        load_hardware_profile(invalid)


def test_fixed_service_profile_covers_each_batch_and_is_repeatable() -> None:
    profile = load_service_profile(CONFIG / "service/fixed.json")
    assert [
        choose_sample(profile, batch_size=n, per_size_sequence=0, seed=7).active_ns
        for n in range(1, 5)
    ] == [100000000, 145000000, 190000000, 235000000]
    assert choose_sample(profile, batch_size=2, per_size_sequence=12, seed=1) == choose_sample(
        profile, batch_size=2, per_size_sequence=12, seed=99
    )
    with pytest.raises(ValueError, match="no batch size"):
        choose_sample(profile, batch_size=5, per_size_sequence=0, seed=0)


def test_empirical_sample_selection_is_stable_and_size_scoped(tmp_path: Path) -> None:
    projection = {
        "model": "m",
        "checkpoint_revision": "rev",
        "server_revision": "b" * 40,
        "deploy_config_sha256": "9" * 64,
        "configured_max_batch_size": 1,
        "configured_batch_wait_ms": 0,
        "rtc_enabled": False,
        "sampling_parameters": {"fields": [], "unknown_fields": "reject"},
        "input_signature": {"observation": {}, "response": {}},
    }
    import hashlib

    projection_hash = hashlib.sha256(
        json.dumps(projection, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    raw = {
        "schema": "policy_server_emulator.service_profile.v1",
        "kind": "empirical",
        "source_kind": "measured",
        "binding": "measured",
        "timing_basis": "scheduler_active_interval",
        "units": "ns",
        "model": "m",
        "model_profile_sha256": "a" * 64,
        "checkpoint_revision": "rev",
        "source_server_revision": "b" * 40,
        "hardware_profile_sha256": "c" * 64,
        "hardware_id": "h",
        "source_configuration_sha256": projection_hash,
        "source_route_configuration_sha256": "d" * 64,
        "source_execution_projection": projection,
        "sources": [
            {"run_id": "run", "receipt_sha256": "e" * 64, "joined_requests_sha256": "f" * 64}
        ],
        "samples_by_batch_size": {
            "1": [{"sample_id": f"s{i}", "active_ns": 100 + i} for i in range(8)]
        },
    }
    path = tmp_path / "service.json"
    path.write_text(json.dumps(raw))
    profile = load_service_profile(path)
    first = choose_sample(
        profile, batch_size=1, per_size_sequence=3, seed=11, sampling="independent_v1"
    )
    assert first == choose_sample(
        profile, batch_size=1, per_size_sequence=3, seed=11, sampling="independent_v1"
    )
    assert first.sample_id.startswith("s")
    raw["source_configuration_sha256"] = "0" * 64
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="canonical execution projection"):
        load_service_profile(path)


def test_model_observation_path_segments_remain_literal(tmp_path: Path) -> None:
    data = json.loads((CONFIG / "models/pi05.json").read_text())
    data["id"] = "test"
    path = tmp_path / "model.json"
    path.write_text(json.dumps(data))
    model = load_model_profile(path)
    assert model.data["observation"]["fields"][0]["path"] == ["observation.images.base_0_rgb"]
    assert isinstance(np.zeros((2, 3, 3), dtype=np.uint8), np.ndarray)


def test_empirical_geometry_guard_rejects_wildcards_but_accepts_concrete_contract(
    tmp_path: Path,
) -> None:
    model_data = json.loads((CONFIG / "models/pi05.json").read_text())
    path = tmp_path / "wildcard.json"
    path.write_text(json.dumps(model_data))
    with pytest.raises(ValueError, match="concrete ndarray input geometry"):
        require_concrete_input_geometry(load_model_profile(path))
    for field in model_data["observation"]["fields"]:
        if field["kind"] == "ndarray":
            field["shape"] = [1 if dim is None else dim for dim in field["shape"]]
    concrete = tmp_path / "concrete.json"
    concrete.write_text(json.dumps(model_data))
    require_concrete_input_geometry(load_model_profile(concrete))
