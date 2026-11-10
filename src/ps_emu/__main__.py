"""Run the policy server emulator from an external configuration."""

from __future__ import annotations

import argparse
import asyncio
import signal
from pathlib import Path
from typing import Any

from .config import resolve_config
from .runtime import PolicyServerEmulator


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CPU-only vLLM-Omni policy server emulator")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--status-path", type=Path)
    parser.add_argument("--model-profile", type=Path)
    parser.add_argument("--hardware-profile", type=Path)
    parser.add_argument("--service-profile", type=Path)
    parser.add_argument("--fault-profile", type=Path)
    parser.add_argument("--overhead-profile", type=Path)
    parser.add_argument("--cohort-policy-id", help="resolved workload admission policy identity")
    parser.add_argument("--service-seed", type=int, help="validated empirical service sample seed")
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--path")
    return parser.parse_args()


async def _serve(args: argparse.Namespace) -> None:
    overrides: dict[str, Any] = {}
    for name in ("model_profile", "hardware_profile", "service_profile", "fault_profile", "overhead_profile"):
        value = getattr(args, name)
        if value is not None:
            overrides[name] = str(value)
    transport = {
        name: getattr(args, name)
        for name in ("host", "port", "path")
        if getattr(args, name) is not None
    }
    if transport:
        overrides["transport"] = transport
    if args.cohort_policy_id is not None:
        overrides["cohort"] = {"policy_id": args.cohort_policy_id}
    if args.service_seed is not None:
        overrides["service"] = {"seed": args.service_seed}
    settings = resolve_config(args.config, overrides=overrides or None)
    emulator = PolicyServerEmulator(settings, args.output_dir, args.status_path)
    await emulator.start()
    stopped = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stopped.set)
        except NotImplementedError:
            pass
    print(emulator.url, flush=True)
    stop_task = asyncio.create_task(stopped.wait())
    failure_task = asyncio.create_task(emulator.wait_failed())
    done, pending = await asyncio.wait(
        (stop_task, failure_task), return_when=asyncio.FIRST_COMPLETED
    )
    for task in pending:
        task.cancel()
    await asyncio.gather(*pending, return_exceptions=True)
    if failure_task in done:
        failure = failure_task.result()
        await emulator.close()
        raise RuntimeError(f"policy server emulator failed: {failure}") from failure
    await emulator.close()
    if emulator.receipt.get("status") == "aborted":
        raise RuntimeError(f"policy server emulator aborted: {emulator.receipt.get('failure')}")


def main() -> None:
    asyncio.run(_serve(_arguments()))


if __name__ == "__main__":
    main()
