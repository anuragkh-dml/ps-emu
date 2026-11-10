# ps-emu

`ps-emu` is a CPU-only policy server for exercising the existing vLLM-Omni
OpenPI MessagePack protocol and admission-control client. It returns profile
shaped dummy actions while consuming real or generated observations. It does
not run a policy model and its timing is synthetic or profile-driven; output
is never GPU performance evidence.

The initial extraction was made from admission-control prototype source commit
`d81d0d57e36142316a63c6f274be86567281d8ae`. This is extraction provenance; the
production vLLM-Omni revision represented by any measured profile remains a
separate identity.

## Install and run

```sh
python -m pip install '.[dev]'
python -m ps_emu --config path/to/server.toml --output-dir /tmp/ps-emu-run
```

The process prints its WebSocket URL after readiness. It writes a final receipt,
events, and status evidence under the configured output location. Use
`ps_emu.resources.example_profile("server.toml")` to locate a packaged example;
other shipped files include `models/pi05.json`, `models/gr00t.json`,
`hardware/synthetic.json`, `service/fixed.json`, `faults/status_freeze.toml`,
and `observations.json`. Relative profile references resolve from the
configuration file's directory.

## Python API

`ps_emu.config.resolve_config` validates and resolves execution profiles.
`ps_emu.profiles` provides strict model, hardware, and service profile loaders.
`ps_emu.observations` validates generated-observation profiles and creates a
bounded, deterministic model-shaped observation pool. `ps_emu.provenance`
reports the package version and source commit only when that identity can be
verified.

## Configuration

See [docs/CONFIGURATION.md](docs/CONFIGURATION.md) for the model contract,
service/execution settings, transport and resource limits, and portable profile
examples. Example resources are accessible using `example_profile(name)`.

The server core depends on NumPy, MessagePack, and websockets. It has no GPU,
Torch, production vLLM-Omni, benchmark, or admission-client dependency.

## Development

```sh
python -m pip install -e '.[dev]'
python -m ruff check src tests
python -m pytest -q
```

This repository is test infrastructure for the vLLM-Omni OpenPI wire contract.
Keep model, GPU, service-time, execution, and fault values in portable external
profiles; preserve schema identifiers, bounded resources, and synthetic
evidence markers. See `AGENTS.md` for ownership and safety invariants.
