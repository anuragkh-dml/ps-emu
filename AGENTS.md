# Policy server emulator

This repository owns an independently implemented, CPU-only test server for
the existing vLLM-Omni OpenPI wire contract. It does not emulate production
inference internals.

- Keep model, hardware, service, execution, and fault values in external
  configuration. Runtime algorithms must not branch on model or GPU identity.
- Do not import the admission benchmark/client, production `vllm_omni`, Torch,
  or a sibling checkout.
- Preserve synthetic identity and profile digests. Emulator output is never
  measured GPU evidence or eligible for live calibration.
- Bound sockets, frames, queues/bytes, event evidence, and shutdown. Preserve
  one outstanding request per connection to protect reply association.
- Keep the MessagePack implementation independent from client code and test
  fixed wire fixtures.
- Keep profiles portable and package-shipped examples discoverable through
  `ps_emu.resources.example_profile`.
- Preserve existing `policy_server_emulator.*` and `vllm_omni.*` schema IDs.
