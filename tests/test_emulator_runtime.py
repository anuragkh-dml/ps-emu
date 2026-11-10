from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import numpy as np
import pytest
from websockets.asyncio.client import connect

from ps_emu.config import resolve_config
from ps_emu.cohort import COHORT_FIELD, COHORT_SCHEMA, membership_hash
from ps_emu.protocol import pack, unpack
from ps_emu.runtime import PolicyServerEmulator
from ps_emu.resources import example_profile


CONFIG = example_profile("server.toml")
pytestmark = pytest.mark.integration


def _observation(request_id: str = "r1") -> dict[str, object]:
    return {
        "session_id": "session",
        "request_id": request_id,
        "observation.images.base_0_rgb": np.zeros((8, 9, 3), dtype=np.uint8),
        "observation.images.left_wrist_0_rgb": np.ones((7, 8, 3), dtype=np.uint8),
        "state": np.arange(32, dtype=np.float32),
        "prompt": "move",
    }


def test_runtime_serves_real_openpi_binary_frames_and_emits_receipt(tmp_path: Path) -> None:
    async def run() -> None:
        output = tmp_path / "run"
        status = tmp_path / "predictive" / "server-one.json"
        settings = resolve_config(CONFIG, {"transport": {"port": 0}, "service": {"scale": 0.01}})
        emulator = PolicyServerEmulator(settings, output, status)
        readiness = await emulator.start()
        assert readiness["ready"] and readiness["url"].endswith("/v1/realtime/robot/openpi")
        async with connect(readiness["url"], compression=None, ping_interval=None) as ws:
            handshake = unpack(await ws.recv())
            assert handshake["action_horizon"] == 50
            assert handshake["policy_server_emulator"]["synthetic"] is True
            assert handshake["policy_server_emulator"]["configuration_sha256"] == settings.sha256
            assert handshake["policy_server_emulator"]["instance_id"]
            await ws.send(pack(_observation()))
            actions = unpack(await ws.recv())
            assert isinstance(actions, np.ndarray)
            assert actions.shape == (50, 32) and actions.dtype == np.float32
        await emulator.close()
        assert json.loads((output / "readiness.json").read_text())["status"] == "ready"
        receipt = json.loads((output / "final_receipt.json").read_text())
        assert receipt["status"] == "complete"
        assert receipt["runtime"]["python"]
        assert set(receipt["runtime"]["packages"]) == {"numpy", "msgpack", "websockets"}
        events = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
        assert {event["kind"] for event in events} >= {
            "queue_enter",
            "forward_start",
            "forward_end",
            "reply_sent",
        }
        queue_event = next(event for event in events if event["kind"] == "queue_enter")
        forward_event = next(event for event in events if event["kind"] == "forward_start")
        assert queue_event["queued_ns"] <= forward_event["forward_start_ns"]
        status_data = json.loads(status.read_text())
        assert status_data["schema"] == "omni_scheduler_progress.v1"
        assert status_data["clock_domain"] == "host_monotonic_ns"
        assert isinstance(status_data.get("hostname"), str) and status_data["hostname"]

    asyncio.run(run())


def test_runtime_rejects_wrong_path_before_handshake(tmp_path: Path) -> None:
    async def run() -> None:
        emulator = PolicyServerEmulator(
            resolve_config(CONFIG, {"transport": {"port": 0}}), tmp_path / "run"
        )
        readiness = await emulator.start()
        bad = readiness["url"].replace("/v1/realtime/robot/openpi", "/wrong")
        async with connect(bad) as ws:
            with pytest.raises(Exception):
                await ws.recv()
        await emulator.close()

    asyncio.run(run())


def test_connections_share_one_bounded_batch_and_service_interval(tmp_path: Path) -> None:
    async def run() -> None:
        output = tmp_path / "run"
        settings = resolve_config(
            CONFIG,
            {
                "transport": {"port": 0},
                "scheduler": {"max_batch_size": 2, "batch_wait_ms": 30},
                "service": {"scale": 0.001},
            },
        )
        emulator = PolicyServerEmulator(settings, output)
        readiness = await emulator.start()
        async with connect(readiness["url"]) as left, connect(readiness["url"]) as right:
            await left.recv()
            await right.recv()
            await asyncio.gather(
                left.send(pack(_observation("left"))), right.send(pack(_observation("right")))
            )
            responses = await asyncio.gather(left.recv(), right.recv())
            assert all(unpack(response).shape == (50, 32) for response in responses)
        await emulator.close()
        events = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
        forwards = [event for event in events if event["kind"] == "forward_start"]
        assert len(forwards) == 1 and forwards[0]["batch_size"] == 2

    asyncio.run(run())


def test_idle_reset_is_acknowledged_and_stream_remains_usable(tmp_path: Path) -> None:
    async def run() -> None:
        emulator = PolicyServerEmulator(
            resolve_config(CONFIG, {"transport": {"port": 0}}), tmp_path / "run"
        )
        readiness = await emulator.start()
        async with connect(readiness["url"]) as ws:
            await ws.recv()
            await ws.send(pack({"type": "reset", "session_id": "session"}))
            assert unpack(await ws.recv()) == {"type": "reset", "status": "ok"}
            await ws.send(pack(_observation()))
            assert unpack(await ws.recv()).shape == (50, 32)
        await emulator.close()

    asyncio.run(run())


@pytest.mark.parametrize("reset", [False, True])
def test_pipelined_frame_during_slow_reply_closes_ambiguous_stream(
    tmp_path: Path, reset: bool
) -> None:
    async def run() -> None:
        fault_path = tmp_path / "fault.toml"
        fault_path.write_text(
            """schema = "policy_server_emulator.faults.v1"\nservice_delay_combination = "max_once_v1"\n[[rules]]\nkind = "reply_delay"\nsession_id = "session"\nrequest_sequence = 0\ndelay_ms = 150\n"""
        )
        settings = resolve_config(
            CONFIG,
            {"transport": {"port": 0}, "service": {"scale": 0.001}, "fault_profile": fault_path},
        )
        emulator = PolicyServerEmulator(settings, tmp_path / "run")
        readiness = await emulator.start()
        async with connect(readiness["url"]) as ws:
            await ws.recv()
            await ws.send(pack(_observation("first")))
            await ws.send(
                pack(
                    {"type": "reset", "session_id": "session"} if reset else _observation("second")
                )
            )
            with pytest.raises(Exception):
                await ws.recv()
        await emulator.close()

    asyncio.run(run())


def test_occupied_port_startup_writes_abort_receipt_without_background_tasks(tmp_path):
    import socket

    async def run():
        with socket.socket() as occupied:
            occupied.bind(("127.0.0.1", 0))
            occupied.listen()
            emulator = PolicyServerEmulator(
                resolve_config(CONFIG, {"transport": {"port": occupied.getsockname()[1]}}),
                tmp_path / "occupied",
            )
            with pytest.raises(OSError):
                await emulator.start()
        assert emulator.receipt["status"] == "aborted"
        receipt = json.loads((tmp_path / "occupied/final_receipt.json").read_text())
        assert receipt["status"] == "aborted" and receipt["failure"]
        assert emulator._server is None and emulator._worker is None
        assert not [
            task for task in asyncio.all_tasks() if task.get_name().startswith("policy-emulator")
        ]

    asyncio.run(run())


def test_blocked_socket_send_does_not_hold_service_or_block_next_client(tmp_path, monkeypatch):
    from websockets.asyncio.server import ServerConnection

    async def run():
        blocked, release = asyncio.Event(), asyncio.Event()
        original_send = ServerConnection.send

        async def slow_send(ws, message, *args, **kwargs):
            if isinstance(unpack(message), np.ndarray) and not blocked.is_set():
                blocked.set()
                await release.wait()
            return await original_send(ws, message, *args, **kwargs)

        monkeypatch.setattr(ServerConnection, "send", slow_send)
        emulator = PolicyServerEmulator(
            resolve_config(
                CONFIG,
                {
                    "transport": {"port": 0},
                    "scheduler": {"max_batch_size": 1, "batch_wait_ms": 0},
                    "service": {"scale": 0.001},
                },
            ),
            tmp_path / "slow-consumer",
        )
        await emulator.start()
        try:
            async with connect(emulator.url) as slow, connect(emulator.url) as fast:
                await slow.recv()
                await fast.recv()
                await slow.send(pack(_observation("slow")))
                await asyncio.wait_for(blocked.wait(), 2)
                await fast.send(pack(_observation("fast")))
                response = unpack(await asyncio.wait_for(fast.recv(), 2))
                assert response.shape == (50, 32)
                assert not release.is_set()
                release.set()
                assert unpack(await asyncio.wait_for(slow.recv(), 2)).shape == (50, 32)
        finally:
            release.set()
            await emulator.close()
        events = [
            json.loads(line)
            for line in (emulator.output_dir / "events.jsonl").read_text().splitlines()
        ]
        starts = {e["request_ids"][0]: e for e in events if e["kind"] == "forward_start"}
        ends = {e["request_ids"][0]: e for e in events if e["kind"] == "forward_end"}
        replies = {e["request_id"]: e for e in events if e["kind"] == "reply_sent"}
        assert ends["slow"]["forward_end_ns"] <= starts["fast"]["forward_start_ns"]
        assert replies["fast"]["reply_sent_ns"] < replies["slow"]["reply_sent_ns"]
        assert emulator._worker.done()
        assert not emulator.queue.items and not emulator._connections

    asyncio.run(run())


def test_disconnect_removes_queued_request_and_next_request_is_served(tmp_path: Path) -> None:
    async def run() -> None:
        settings = resolve_config(
            CONFIG,
            {
                "transport": {"port": 0},
                "scheduler": {"max_batch_size": 2, "batch_wait_ms": 250},
                "service": {"scale": 0.001},
            },
        )
        output = tmp_path / "run"
        emulator = PolicyServerEmulator(settings, output)
        readiness = await emulator.start()
        abandoned = connect(readiness["url"])
        async with abandoned as ws:
            await ws.recv()
            await ws.send(pack(_observation("abandoned")))
        async with connect(readiness["url"]) as ws:
            await ws.recv()
            await ws.send(pack(_observation("survivor")))
            assert unpack(await ws.recv()).shape == (50, 32)
        await emulator.close()
        events = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
        starts = [event for event in events if event["kind"] == "forward_start"]
        assert all("abandoned" not in event["request_ids"] for event in starts)
        assert any(
            event["kind"] == "queue_leave" and event.get("request_id") == "abandoned"
            for event in events
        )

    asyncio.run(run())


def test_runtime_refuses_existing_output_directory(tmp_path: Path) -> None:
    async def run() -> None:
        output = tmp_path / "existing"
        output.mkdir()
        emulator = PolicyServerEmulator(resolve_config(CONFIG, {"transport": {"port": 0}}), output)
        with pytest.raises(FileExistsError):
            await emulator.start()

    asyncio.run(run())


def test_shutdown_grace_bounds_a_long_active_service(tmp_path: Path) -> None:
    async def run() -> None:
        service = json.loads(
            example_profile("service/fixed.json").read_text()
        )
        service["active_ns_by_batch_size"]["1"] = 5_000_000_000
        service_path = tmp_path / "long-service.json"
        service_path.write_text(json.dumps(service))
        settings = resolve_config(
            CONFIG,
            {
                "transport": {"port": 0},
                "lifecycle": {"shutdown_grace_s": 0.05},
                "service_profile": service_path,
            },
        )
        output = tmp_path / "run"
        emulator = PolicyServerEmulator(settings, output)
        readiness = await emulator.start()
        ws = await connect(readiness["url"])
        await ws.recv()
        await ws.send(pack(_observation()))
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline:
            event_path = output / "events.jsonl"
            if event_path.exists() and any(
                json.loads(line)["kind"] == "forward_start"
                for line in event_path.read_text().splitlines()
            ):
                break
            await asyncio.sleep(0.005)
        await asyncio.sleep(0.02)
        started = time.monotonic()
        await emulator.close()
        assert time.monotonic() - started < 0.5
        await ws.close()
        receipt = json.loads((output / "final_receipt.json").read_text())
        assert receipt["status"] == "aborted"
        assert "shutdown grace exceeded" in receipt["failure"]

    asyncio.run(run())


def test_pending_queue_capacity_returns_bounded_protocol_error(tmp_path: Path) -> None:
    async def run() -> None:
        service = json.loads(
            example_profile("service/fixed.json").read_text()
        )
        service["active_ns_by_batch_size"]["1"] = 300_000_000
        service_path = tmp_path / "service.json"
        service_path.write_text(json.dumps(service))
        settings = resolve_config(
            CONFIG,
            {
                "transport": {"port": 0},
                "scheduler": {"max_pending_requests": 1, "max_batch_size": 1},
                "service_profile": service_path,
            },
        )
        output = tmp_path / "run"
        emulator = PolicyServerEmulator(settings, output)
        readiness = await emulator.start()
        clients = [await connect(readiness["url"]) for _ in range(3)]
        for ws in clients:
            await ws.recv()
        await clients[0].send(pack(_observation("active")))
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline:
            event_path = output / "events.jsonl"
            if event_path.exists() and any(
                json.loads(line)["kind"] == "forward_start"
                for line in event_path.read_text().splitlines()
            ):
                break
            await asyncio.sleep(0.005)
        await clients[1].send(pack(_observation("queued")))
        await asyncio.sleep(0.02)
        await clients[2].send(pack(_observation("overflow")))
        error = unpack(await clients[2].recv())
        assert error["type"] == "error" and "capacity" in error["message"]
        await asyncio.gather(*(ws.close() for ws in clients))
        await emulator.close()

    asyncio.run(run())


def test_idle_timeout_closes_inactive_connection(tmp_path: Path) -> None:
    async def run() -> None:
        settings = resolve_config(CONFIG, {"transport": {"port": 0, "idle_timeout_s": 0.05}})
        emulator = PolicyServerEmulator(settings, tmp_path / "run")
        readiness = await emulator.start()
        async with connect(readiness["url"]) as ws:
            await ws.recv()
            with pytest.raises(Exception):
                await ws.recv()
        await emulator.close()

    asyncio.run(run())


def test_status_writer_failure_is_reported_and_close_does_not_retry_forever(
    tmp_path: Path, monkeypatch
) -> None:
    from ps_emu.evidence import EvidenceWriter

    original = EvidenceWriter.snapshot

    def fail_progress(self, value, path=None):
        target = path or self.status_path
        if target.name == "status.json":
            raise OSError("injected progress write failure")
        return original(self, value, path)

    async def run() -> None:
        output = tmp_path / "run"
        emulator = PolicyServerEmulator(resolve_config(CONFIG, {"transport": {"port": 0}}), output)
        await emulator.start()
        monkeypatch.setattr(EvidenceWriter, "snapshot", fail_progress)
        emulator._publish_status()
        failure = await asyncio.wait_for(emulator.wait_failed(), timeout=1)
        assert "injected progress write failure" in str(failure)
        await asyncio.wait_for(emulator.close(), timeout=1)
        receipt = json.loads((output / "final_receipt.json").read_text())
        assert receipt["status"] == "aborted"

    asyncio.run(run())


def test_worker_failure_settles_request_and_aborts_run(tmp_path: Path, monkeypatch) -> None:
    import ps_emu.runtime as runtime_module

    def fail_actions(*args, **kwargs):
        raise RuntimeError("injected action worker failure")

    async def run() -> None:
        output = tmp_path / "run"
        emulator = PolicyServerEmulator(resolve_config(CONFIG, {"transport": {"port": 0}}), output)
        readiness = await emulator.start()
        monkeypatch.setattr(runtime_module, "make_actions", fail_actions)
        async with connect(readiness["url"]) as ws:
            await ws.recv()
            await ws.send(pack(_observation()))
            with pytest.raises(Exception):
                await ws.recv()
        failure = await asyncio.wait_for(emulator.wait_failed(), timeout=1)
        assert "injected action worker failure" in str(failure)
        await asyncio.wait_for(emulator.close(), timeout=1)
        receipt = json.loads((output / "final_receipt.json").read_text())
        assert receipt["status"] == "aborted"

    asyncio.run(run())


def test_invalid_cohort_identity_uses_status_safe_id_and_keeps_diagnostic(tmp_path: Path) -> None:
    async def run() -> None:
        output = tmp_path / "run"
        status = tmp_path / "status.json"
        settings = resolve_config(
            CONFIG,
            {
                "transport": {"port": 0},
                "features": {"cohort_enabled": True},
                "cohort": {"policy_id": "policy-1"},
            },
        )
        emulator = PolicyServerEmulator(settings, output, status)
        readiness = await emulator.start()
        async with connect(readiness["url"]) as ws:
            await ws.recv()
            payload = _observation()
            payload.pop("request_id")
            payload["vllm_omni_cohort"] = {"schema": "bad-schema"}
            await ws.send(pack(payload))
            assert unpack(await ws.recv())["type"] == "error"
            await asyncio.sleep(0.03)
        parsed = json.loads(status.read_text())
        rejected = next(
            event for event in parsed["recent_events"] if event["kind"] == "cohort_metadata_rejected"
        )
        assert isinstance(rejected["request_id"], str) and rejected["request_id"]
        events = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
        diagnostic = next(event for event in events if event["kind"] == "cohort_metadata_rejected")
        assert diagnostic["reported_request_id"] is not None
        await emulator.close()

    asyncio.run(run())


def test_pending_cohort_disconnect_is_rollback_not_closed_receipt(tmp_path: Path) -> None:
    async def run() -> None:
        output = tmp_path / "run"
        settings = resolve_config(
            CONFIG,
            {
                "transport": {"port": 0},
                "scheduler": {"max_batch_size": 2, "batch_wait_ms": 100},
                "features": {"cohort_enabled": True},
                "cohort": {"policy_id": "policy-1"},
            },
        )
        emulator = PolicyServerEmulator(settings, output)
        readiness = await emulator.start()
        members = ["member-a", "member-b"]
        request_id = "request-a"
        payload = _observation(request_id)
        payload["vllm_omni_cohort"] = {
            "schema": COHORT_SCHEMA,
            "epoch_id": "epoch-1",
            "member_ids": members,
            "membership_hash": membership_hash(members),
            "deadline_ns": time.monotonic_ns() + 1_000_000_000,
            "policy_id": "policy-1",
            "member_id": "member-a",
            "request_id": request_id,
        }
        async with connect(readiness["url"]) as ws:
            await ws.recv()
            await ws.send(pack(payload))
            deadline = time.monotonic() + 1
            while time.monotonic() < deadline and not any(
                event["kind"] == "queue_enter" for event in emulator.evidence.recent
            ):
                await asyncio.sleep(0.005)
        await asyncio.sleep(0.02)
        await emulator.close()
        events = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
        assert not any(event["kind"] == "cohort_close" for event in events)
        assert any(
            event["kind"] == "queue_leave" and event.get("reason") == "disconnect_or_pipeline"
            for event in events
        )

    asyncio.run(run())


def test_status_freeze_fault_pauses_and_resumes_real_progress_file(tmp_path: Path) -> None:
    async def run() -> None:
        output = tmp_path / "run"
        status = tmp_path / "status.json"
        fault = example_profile("faults/status_freeze.toml")
        settings = resolve_config(
            CONFIG, {"transport": {"port": 0}, "fault_profile": fault, "service": {"scale": 0.001}}
        )
        emulator = PolicyServerEmulator(settings, output, status)
        readiness = await emulator.start()
        initial = json.loads(status.read_text())["observed_ns"]
        async with connect(readiness["url"]) as ws:
            await ws.recv()
            await ws.send(pack(_observation()))
            assert unpack(await ws.recv()).shape == (50, 32)
        freeze_event = None
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline:
            freeze_event = next(
                (
                    event
                    for event in emulator.evidence.recent
                    if event["kind"] == "fault_applied"
                    and event.get("fault_kind") == "status_freeze"
                ),
                None,
            )
            if freeze_event is not None:
                break
            await asyncio.sleep(0.005)
        assert freeze_event is not None
        await asyncio.sleep(0.03)
        frozen = json.loads(status.read_text())["observed_ns"]
        assert frozen >= initial
        await asyncio.sleep(0.05)
        assert json.loads(status.read_text())["observed_ns"] == frozen
        resumed = None
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline:
            resumed = json.loads(status.read_text())["observed_ns"]
            if resumed > frozen:
                break
            await asyncio.sleep(0.01)
        assert resumed is not None and resumed > frozen
        assert json.loads(status.read_text())["observed_ns"] >= frozen
        await emulator.close()
        assert json.loads(status.read_text())["observed_ns"] > frozen

    asyncio.run(run())


def test_cohort_epoch_batches_members_and_records_close_receipt(tmp_path: Path) -> None:
    async def run() -> None:
        output = tmp_path / "run"
        settings = resolve_config(
            CONFIG,
            {
                "transport": {"port": 0},
                "scheduler": {"max_batch_size": 2, "batch_wait_ms": 100},
                "features": {"cohort_enabled": True},
                "cohort": {"policy_id": "policy-1"},
                "service": {"scale": 0.001},
            },
        )
        emulator = PolicyServerEmulator(settings, output)
        readiness = await emulator.start()
        members = ["member-a", "member-b"]
        digest = membership_hash(members)
        deadline_ns = time.monotonic_ns() + 1_000_000_000
        clients = [await connect(readiness["url"]) for _ in members]
        for ws in clients:
            await ws.recv()
        payloads = []
        for member, request_id in zip(members, ("r-a", "r-b"), strict=True):
            payload = _observation(request_id)
            payload[COHORT_FIELD] = {
                "schema": COHORT_SCHEMA,
                "epoch_id": "epoch-1",
                "member_ids": members,
                "membership_hash": digest,
                "deadline_ns": deadline_ns,
                "policy_id": "policy-1",
                "member_id": member,
                "request_id": request_id,
            }
            payloads.append(pack(payload))
        await asyncio.gather(*(ws.send(frame) for ws, frame in zip(clients, payloads, strict=True)))
        responses = await asyncio.gather(*(ws.recv() for ws in clients))
        assert all(unpack(response).shape == (50, 32) for response in responses)
        await asyncio.gather(*(ws.close() for ws in clients))
        await emulator.close()
        events = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
        close = next(event for event in events if event["kind"] == "cohort_close")
        assert close["expected_count"] == close["arrived_count"] == 2
        assert close["missing_member_ids"] == [] and close["close_reason"] == "expected_count"

    asyncio.run(run())


def test_session_sequence_is_connection_scoped_and_session_count_bounded(tmp_path: Path) -> None:
    async def run() -> None:
        output = tmp_path / "run"
        settings = resolve_config(
            CONFIG,
            {
                "transport": {"port": 0, "max_sessions_per_connection": 1},
                "scheduler": {"max_batch_size": 2, "batch_wait_ms": 50},
                "service": {"scale": 0.001},
            },
        )
        emulator = PolicyServerEmulator(settings, output)
        readiness = await emulator.start()
        clients = [await connect(readiness["url"]) for _ in range(2)]
        for ws in clients:
            await ws.recv()
        shared = _observation("same")
        shared["session_id"] = "same-session"
        await asyncio.gather(*(ws.send(pack(shared)) for ws in clients))
        actions = [unpack(await ws.recv()) for ws in clients]
        assert np.array_equal(actions[0], actions[1])
        second_session = _observation("other")
        second_session["session_id"] = "second-session"
        await clients[0].send(pack(second_session))
        error = unpack(await clients[0].recv())
        assert error["type"] == "error" and "session capacity" in error["message"]
        await clients[0].send(pack({"type": "reset", "session_id": "same-session"}))
        await clients[0].recv()
        await clients[0].send(pack(second_session))
        assert unpack(await clients[0].recv()).shape == (50, 32)
        await asyncio.gather(*(ws.close() for ws in clients))
        await emulator.close()

    asyncio.run(run())
