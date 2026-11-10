# Configuration reference

The emulator loads external JSON and TOML profiles. Model, hardware, latency,
batching, and fault values never come from model/GPU dispatch tables in Python.
`ps_emu.resources.example_profile(name)` locates a portable packaged profile;
all profile references resolve relative to the containing execution file.
Overrides are applied and validated before the server binds. Profiles are read
once and their normalized and raw digests are saved in the run snapshot.

| Profile | Schema | Purpose |
| --- | --- | --- |
| Model JSON | `policy_server_emulator.model.v1` | Model identity, handshake, observation fields, response/action structure, generator settings, sampling parameters, and capabilities such as RTC. |
| Hardware JSON | `policy_server_emulator.hardware.v1` | Declared device identity/count and available device attributes. It describes the modeled environment; it does not detect or allocate a GPU. |
| Service JSON | `policy_server_emulator.service_profile.v1` | Fixed per-batch service times or empirical per-batch samples, plus source timing and hardware/model provenance. |
| Execution TOML | `policy_server_emulator.execution.v1` | Profile references, transport, scheduling, resource limits, feature switches, lifecycle, and event/status output settings. |
| Fault TOML | `policy_server_emulator.faults.v1` | Explicit request/session selectors and bounded injected outcomes. No selected rule means no fault. |

`evaluation.toml` and evaluation reports are owned by the admission benchmark
project because their criteria compare benchmark results, not server behavior.

## Model contracts

A model profile names a model and declares the exact initial handshake,
observation fields, response tree, action horizon/time axis, output generator,
sampling parameters, and supported protocol features. The emulator returns
synthetic, profile-shaped actions; it does not infer model capabilities or
modify input observations. The reserved `policy_server_emulator` handshake
field is set by the emulator itself.

Observation paths use literal nested string keys. Supported leaf types are
strings and numeric arrays with declared dtype, shape, and finite-value rules.
Wildcard dimensions permit protocol correctness checks against real input
sizes. Generated observation pools require concrete array dimensions so their
memory use can be bounded before allocation. Response declarations support
arrays and nested mappings with explicit exact-key behavior. Unsupported
samplers or response structures fail profile validation.

Model profiles can declare RTC capability and the fields consumed by RTC
requests. RTC remains protocol behavior: the server validates inputs and emits
the configured action format. The profile supplies the model-specific contract
through data, while the implementation remains identity-agnostic.

## Service and execution settings

A service profile provides nonnegative active durations for each configured
batch size. Fixed profiles give one duration per size; empirical profiles give
sample lists and bind those values to model, hardware, execution projection,
and source receipts. A run chooses samples deterministically from the selected
seed and per-size sequence. Optional residual overhead profiles describe
bounded pre-queue and post-service costs. These values predict only the
emulator's configured service schedule; they do not make CPU measurements a
claim about live GPU inference.

Execution profiles name relative model, hardware, service, optional fault, and
optional overhead files. The `transport` table controls bind host/port/path,
compression, keepalive, idle timeout, maximum connections, and maximum frame
size. The `scheduler` table declares the supported FIFO collection policy,
pending request and byte caps, maximum batch size, batch wait, and cancellation
behavior. Current implementation has one active service worker/batch at a
time. Unsupported scheduler names or parallel service settings are rejected.

`service` selects fixed/empirical sample behavior, seed, and positive scale.
`features` toggles RTC and cohort support. The `cohort` table declares the
wire schema, policy identity, and registry cap. Direct users may set the
cohort identity at launch; benchmark orchestration can derive it from its
resolved route configuration and pass it to the server. `lifecycle` bounds
readiness and shutdown grace periods. `evidence` configures status cadence,
recent event capacity, writer queue capacity, and flush cadence. Queue, frame,
connection, session, event, and shutdown resources have explicit limits.

Fault rules are external and deterministic. Multiple compatible delays combine
with the declared max-once policy; conflicting outcome kinds are rejected
rather than resolved by file ordering. The CLI supports profile path and
transport overrides, but every resolved value is validated before bind and
included in the immutable startup snapshot.

## Packaged examples

The default `server.toml` uses `models/pi05.json`,
`hardware/synthetic.json`, and `service/fixed.json`. The model and hardware
files illustrate protocol and profile shape only. `models/gr00t.json` covers a
second response/observation contract. `faults/status_freeze.toml` demonstrates
one deterministic fault. `observations.json` is a portable generated-input
template, not a dataset record. Resolve paths with:

```python
from ps_emu.resources import example_profile

execution = example_profile("server.toml")
model = example_profile("models/gr00t.json")
```

Templates contain no machine-specific checkout paths. Historical result
artifacts retain their original profile snapshots and schema identifiers.
