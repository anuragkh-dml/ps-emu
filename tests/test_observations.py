import json

from ps_emu.observations import (
    generate_observation_pool,
    load_observation_profile,
    observation_pool_bytes,
)
from ps_emu.profiles import load_model_profile
from ps_emu.resources import example_profile


def _concrete_model(tmp_path):
    data = json.loads(example_profile("models/pi05.json").read_text())
    for field in data["observation"]["fields"]:
        if field["kind"] == "ndarray":
            field["shape"] = [8 if dim is None else dim for dim in field["shape"]]
    path = tmp_path / "model.json"
    path.write_text(json.dumps(data))
    return load_model_profile(path)


def test_generated_observation_pool_is_bounded_and_repeatable(tmp_path):
    model = _concrete_model(tmp_path)
    profile = load_observation_profile(example_profile("observations.json"))
    size = observation_pool_bytes(model, profile["pool_size"])
    pool_a, bytes_a, identity_a = generate_observation_pool(
        model, profile, "client-a", max_decoded_bytes=size
    )
    pool_b, bytes_b, identity_b = generate_observation_pool(
        model, profile, "client-a", max_decoded_bytes=size
    )
    assert bytes_a == bytes_b == size
    assert identity_a == identity_b
    assert len(pool_a) == len(pool_b) == profile["pool_size"]
    assert pool_a[0]["prompt"] == profile["prompt"]
    assert not pool_a[0]["observation.images.base_0_rgb"].flags.writeable


def test_generated_observation_pool_enforces_memory_limit(tmp_path):
    model = _concrete_model(tmp_path)
    profile = load_observation_profile(example_profile("observations.json"))
    try:
        generate_observation_pool(model, profile, "client-a", max_decoded_bytes=1)
    except ValueError as exc:
        assert "memory budget" in str(exc)
    else:
        raise AssertionError("pool beyond byte limit was accepted")
