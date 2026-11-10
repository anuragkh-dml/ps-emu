from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ps_emu.contracts import handshake, make_actions, validate_observation
from ps_emu.profiles import load_model_profile


from ps_emu.resources import example_profile

CONFIG = Path(example_profile("models/pi05.json")).parent


def _pi_observation() -> dict[str, object]:
    return {
        "observation.images.base_0_rgb": np.zeros((8, 9, 3), dtype=np.uint8),
        "observation.images.left_wrist_0_rgb": np.ones((7, 8, 3), dtype=np.uint8),
        "state": np.arange(32, dtype=np.float32),
        "prompt": "pick up block",
    }


def test_pi05_handshake_and_deterministic_actions() -> None:
    model = load_model_profile(CONFIG / "pi05.json")
    assert handshake(model)["action_horizon"] == 50
    observation = _pi_observation()
    validate_observation(model, observation)
    first = make_actions(
        model, observation, {"session_id": "s1", "request_id": "r1", "session_sequence": 0}
    )
    assert first.shape == (50, 32) and first.dtype == np.float32
    assert np.array_equal(
        first,
        make_actions(
            model, observation, {"session_id": "s1", "request_id": "r1", "session_sequence": 0}
        ),
    )
    second = make_actions(
        model, observation, {"session_id": "s2", "request_id": "r1", "session_sequence": 0}
    )
    assert not np.array_equal(first, second)


def test_generic_gr00t_handshake_nested_observation_and_action_tree() -> None:
    model = load_model_profile(CONFIG / "gr00t.json")
    assert handshake(model)["embodiment_tag"].startswith("OXE_")
    observation = {
        "video": {
            "exterior_image_1_left": np.zeros((1, 2, 10, 11, 3), dtype=np.uint8),
            "wrist_image_left": np.zeros((1, 2, 10, 11, 3), dtype=np.uint8),
        },
        "state": {
            "eef_9d": np.zeros((1, 1, 9), dtype=np.float32),
            "gripper_position": np.zeros((1, 1, 1), dtype=np.float32),
            "joint_position": np.zeros((1, 1, 7), dtype=np.float32),
        },
        "prompt": "move",
    }
    actions = make_actions(
        model, observation, {"session_id": "s", "request_id": "r", "session_sequence": 1}
    )
    assert {k: v.shape for k, v in actions.items()} == {
        "eef_9d": (1, 40, 9),
        "gripper_position": (1, 40, 1),
        "joint_position": (1, 40, 7),
    }
    assert not np.array_equal(actions["eef_9d"], actions["gripper_position"])
    assert not np.array_equal(actions["eef_9d"][:, 0], actions["eef_9d"][:, 1])


def test_contract_rejects_wrong_dtype_shape_and_prompt() -> None:
    model = load_model_profile(CONFIG / "pi05.json")
    observation = _pi_observation()
    observation["state"] = np.zeros(32, dtype=np.float64)
    with pytest.raises(ValueError, match="dtype"):
        validate_observation(model, observation)
    observation = _pi_observation()
    observation["prompt"] = "  "
    with pytest.raises(ValueError, match="nonempty"):
        validate_observation(model, observation)


def test_rtc_is_declared_external_validated_and_returns_two_action_roles() -> None:
    model = load_model_profile(CONFIG / "pi05.json")
    obs = _pi_observation()
    assert "rtc" not in handshake(model)
    assert handshake(model, {"rtc_enabled": True})["rtc"]["modes"] == ["guided"]
    obs["sampling_params"] = {
        "rtc": {
            "schema_version": 1,
            "mode": "guided",
            "previous_actions": np.zeros((50, 32), dtype=np.float32),
            "inference_delay": 1,
            "execution_horizon": 4,
            "prefix_attention_schedule": "linear",
            "max_guidance_weight": 1.0,
            "correlation": {"generation": 0, "request_id": "r"},
        }
    }
    response = make_actions(
        model,
        obs,
        {"session_id": "s", "request_id": "r", "session_sequence": 0},
        {"rtc_enabled": True},
    )
    assert set(response) == {"actions", "model_actions", "rtc"}
    assert response["rtc"]["correlation"]["request_id"] == "r"
    assert not np.array_equal(response["actions"], response["model_actions"])
    obs["sampling_params"]["rtc"]["execution_horizon"] = 51
    with pytest.raises(ValueError, match="RTC timing"):
        make_actions(
            model,
            obs,
            {"session_id": "s", "request_id": "r", "session_sequence": 0},
            {"rtc_enabled": True},
        )


def test_temporary_profile_swaps_dimensions_without_name_dispatch(tmp_path: Path) -> None:
    data = json.loads((CONFIG / "pi05.json").read_text())
    data["id"] = "new-contract"
    data["features"] = {}
    data["handshake"]["action_horizon"] = 3
    data["handshake"]["action_dim"] = 2
    data["response"]["shape"] = [3, 2]
    data["observation"]["fields"][2]["shape"] = [2]
    data["observation"]["fields"][2]["path"] = ["custom_state"]
    data["generator"]["state_paths"] = [["custom_state"]]
    data["generator"]["offset"] = 2.0
    path = tmp_path / "custom.json"
    path.write_text(json.dumps(data))
    model = load_model_profile(path)
    obs = _pi_observation()
    obs.pop("state")
    obs["custom_state"] = np.zeros(2, dtype=np.float32)
    actions = make_actions(model, obs, {"request_id": "x"})
    assert actions.shape == (3, 2)
    assert actions.min() > 1.9
    from ps_emu.config import resolve_config

    hardware = json.loads((CONFIG.parent / "hardware/synthetic.json").read_text())
    hardware["id"] = "arbitrary-cpu-test-host"
    hardware["device_name"] = "Profile-only fixture device"
    hardware_path = tmp_path / "hardware.json"
    hardware_path.write_text(json.dumps(hardware))
    settings = resolve_config(
        example_profile("server.toml"),
        {
            "model_profile": path,
            "hardware_profile": hardware_path,
        },
    )
    snapshot = settings.snapshot()
    assert snapshot["profiles"]["model"]["data"]["id"] == "new-contract"
    assert snapshot["profiles"]["hardware"]["data"]["id"] == hardware["id"]
