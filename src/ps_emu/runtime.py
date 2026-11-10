"""CPU-only asynchronous OpenPI policy server emulator."""

from __future__ import annotations

import asyncio
import json
import os
import platform
from importlib.metadata import version
import socket
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import quote

import websockets
from websockets.exceptions import ConnectionClosed

from .cohort import COHORT_FIELD, CohortRegistry, validate_cohort
from .contracts import handshake, make_actions, validate_observation
from .evidence import EvidenceWriter
from .profiles import choose_sample
from .protocol import ProtocolError, pack, unpack
from .provenance import source_identity
from .scheduler import BoundedBatchQueue, QueueFull, ScheduledRequest


class PolicyServerEmulator:
    """A single-process shared-service emulator, with bounded queue and one worker."""

    def __init__(
        self, settings: Any, output_dir: str | Path, status_path: str | Path | None = None
    ) -> None:
        self.settings = settings
        self.output_dir = Path(output_dir)
        self.status_path = Path(status_path) if status_path else None
        data = settings.data
        transport, scheduler = data["transport"], data["scheduler"]
        self.host, self.port, self.path = (
            transport["host"],
            int(transport["port"]),
            transport["path"],
        )
        self.max_frame = int(transport["max_frame_bytes"])
        self.max_connections = int(transport["max_connections"])
        self.idle_timeout = float(transport["idle_timeout_s"])
        self.max_batch_size = int(scheduler["max_batch_size"])
        self.queue = BoundedBatchQueue(
            int(scheduler["max_pending_requests"]),
            int(scheduler["max_pending_bytes"]),
            self.max_batch_size,
            float(scheduler["batch_wait_ms"]) / 1000,
        )
        self.features = data.get("features", {})
        self.service = data["service"]
        self.seed = int(self.service.get("seed", 0))
        self.scale = float(self.service.get("scale", 1.0))
        self.cohort_enabled = bool(self.features.get("cohort_enabled", False))
        self.cohort_policy = data.get("cohort", {}).get("policy_id", "")
        self.registry = CohortRegistry(
            max_registered_epochs=int(data["cohort"]["max_registered_epochs"])
        )
        self.evidence = EvidenceWriter(
            self.output_dir,
            recent_capacity=int(data["evidence"]["recent_event_capacity"]),
            writer_capacity=int(data["evidence"]["writer_queue_capacity"]),
        )
        self._server: Any = None
        self._worker: asyncio.Task[None] | None = None
        self._heartbeat: asyncio.Task[None] | None = None
        self._evidence_task: asyncio.Task[None] | None = None
        self._status_version = 0
        self._status_written_version = 0
        self._force_status_version = 0
        self._status_freeze_until_ns = 0
        self._status_task: asyncio.Task[None] | None = None
        self._condition = asyncio.Condition()
        self._closing = False
        self._failure: BaseException | None = None
        self._failure_event = asyncio.Event()
        self._active: dict[str, Any] | None = None
        self._connections: set[Any] = set()
        self._connection_ids: set[str] = set()
        self._state_version = 0
        self._sequence_by_session: dict[tuple[str, str], int] = {}
        self._sessions_by_connection: dict[str, set[str]] = {}
        self._batch_sequence: dict[int, int] = {}
        self._receipt: dict[str, Any] = {
            "schema": "policy_server_emulator.receipt.v1",
            "status": "starting",
            "emulator_source": source_identity(),
            "runtime": {
                "python": platform.python_version(),
                "implementation": platform.python_implementation(),
                "platform": platform.platform(),
                "machine": platform.machine(),
                "hostname": socket.gethostname(),
                "cpu_count": os.cpu_count(),
                "packages": {name: version(name) for name in ("numpy", "msgpack", "websockets")},
            },
        }
        self._instance_id = uuid.uuid4().hex

    @property
    def url(self) -> str:
        if self._server is None or not self._server.sockets:
            return f"ws://{self.host}:{self.port}{self.path}"
        port = self._server.sockets[0].getsockname()[1]
        host = "127.0.0.1" if self.host in ("0.0.0.0", "::") else self.host
        return f"ws://{host}:{port}{self.path}"

    @property
    def receipt(self) -> dict[str, Any]:
        return dict(self._receipt)

    @property
    def readiness(self) -> dict[str, Any]:
        return {
            "ready": self._server is not None and not self._closing,
            "url": self.url,
            "profile_sha256": self.settings.sha256,
            "model_id": self.settings.model.model_id,
            "instance_id": self._instance_id,
        }

    async def start(self) -> dict[str, Any]:
        self.output_dir.parent.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(exist_ok=False)
        try:
            self._server = await websockets.serve(
                self._accept,
                self.host,
                self.port,
                compression=None
                if self.settings.data["transport"]["compression"] == "none"
                else self.settings.data["transport"]["compression"],
                ping_interval=(
                    None
                    if self.settings.data["transport"]["ping_interval_s"] == 0
                    else float(self.settings.data["transport"]["ping_interval_s"])
                ),
                max_size=self.max_frame,
                max_queue=int(self.settings.data["transport"]["receive_queue_capacity"]),
                close_timeout=1,
                server_header=None,
            )
        except BaseException as exc:
            self._set_failure(exc)
            self._receipt.update(
                {"status": "aborted", "failure": str(exc), "failed_start_ns": time.monotonic_ns()}
            )
            self.evidence.initial_receipt(self._receipt)
            self.evidence.receipt(self._receipt)
            raise
        self._worker = asyncio.create_task(self._run_scheduler(), name="policy-emulator-scheduler")
        self._heartbeat = asyncio.create_task(
            self._run_heartbeat(), name="policy-emulator-heartbeat"
        )
        self._evidence_task = asyncio.create_task(
            self._run_evidence_writer(), name="policy-emulator-evidence"
        )
        self._receipt.update(
            {
                "status": "ready",
                "url": self.url,
                "instance_id": self._instance_id,
                "profile_sha256": self.settings.sha256,
                "model_profile_sha256": self.settings.model.sha256,
                "hardware_profile_sha256": self.settings.hardware.sha256,
                "service_profile_sha256": self.settings.service_profile.sha256,
                "started_ns": time.monotonic_ns(),
            }
        )
        await asyncio.to_thread(
            self.evidence.snapshot,
            self.settings.snapshot(),
            self.output_dir / "config_snapshot.json",
        )
        self.evidence.initial_receipt(self._receipt)
        self._publish_status(force=True)
        await self._flush_status()
        return self.readiness

    async def close(self) -> None:
        if self._server is None:
            return
        self._closing = True
        async with self._condition:
            self._condition.notify_all()
        try:
            self._server.close(close_connections=False)
        except TypeError:
            self._server.close()
        grace = float(self.settings.data["lifecycle"]["shutdown_grace_s"])
        if self._worker is not None:
            try:
                await asyncio.wait_for(self._worker, timeout=grace)
            except TimeoutError:
                self._set_failure(
                    RuntimeError("shutdown grace exceeded; active work was cancelled")
                )
                self._worker.cancel()
                await asyncio.gather(self._worker, return_exceptions=True)
                await self._abort(self._failure)
        for ws in tuple(self._connections):
            await ws.close(code=1001, reason="server shutdown")
        try:
            await asyncio.wait_for(self._server.wait_closed(), timeout=grace)
        except TimeoutError:
            pass
        if self._heartbeat is not None:
            self._heartbeat.cancel()
            await asyncio.gather(self._heartbeat, return_exceptions=True)
        if self._evidence_task is not None:
            self._evidence_task.cancel()
            await asyncio.gather(self._evidence_task, return_exceptions=True)
        await asyncio.to_thread(self.evidence.flush_events)
        status = "aborted" if self._failure else "complete"
        self._receipt.update(
            {
                "status": status,
                "closed_ns": time.monotonic_ns(),
                "failure": str(self._failure) if self._failure else None,
                "event_count": self.evidence.event_count,
            }
        )
        self._publish_status(force=True)
        try:
            await self._flush_status()
        except RuntimeError as exc:
            self._set_failure(self._failure or exc)
            self._receipt.update({"status": "aborted", "failure": str(self._failure)})
        self.evidence.receipt(self._receipt)

    async def _accept(self, ws: Any) -> None:
        if len(self._connections) >= self.max_connections or self._closing:
            await ws.close(code=1013, reason="connection capacity or shutdown")
            return
        self._connections.add(ws)
        conn_id = uuid.uuid4().hex
        recv_task: asyncio.Task[Any] | None = None
        try:
            if ws.request.path != self.path:
                await ws.close(code=1008, reason="unexpected path")
                return
            await ws.send(pack(handshake(self.settings.model, self._feature_args())))
            while not self._closing:
                if recv_task is None:
                    recv_task = asyncio.create_task(ws.recv())
                try:
                    frame = await asyncio.wait_for(recv_task, timeout=self.idle_timeout)
                    recv_task = None
                except TimeoutError:
                    await ws.close(code=1001, reason="idle timeout")
                    return
                if isinstance(frame, str):
                    await self._send_error(ws, "text frames are unsupported")
                    continue
                payload: Any = None
                try:
                    payload = unpack(frame, max_bytes=self.max_frame)
                    if isinstance(payload, dict) and payload.get("type") == "reset":
                        if set(payload) - {"type", "session_id"}:
                            raise ValueError("reset message has unsupported fields")
                        session = payload.get("session_id", conn_id)
                        if not isinstance(session, str) or not session or len(session) > 256:
                            raise ValueError("reset session_id must be a nonempty bounded string")
                        self._sequence_by_session.pop((conn_id, session), None)
                        self._sessions_by_connection.get(conn_id, set()).discard(session)
                        self._event("session_reset", session_id=session)
                        await ws.send(pack({"type": "reset", "status": "ok"}))
                        continue
                    request = self._make_request(conn_id, payload, ws)
                except (ValueError, TypeError, ProtocolError, KeyError) as exc:
                    if isinstance(payload, dict) and COHORT_FIELD in payload:
                        raw_request_id = payload.get("request_id")
                        valid_request_id = (
                            raw_request_id
                            if isinstance(raw_request_id, str)
                            and raw_request_id
                            and len(raw_request_id) <= 256
                            else f"rejected-{conn_id}"
                        )
                        diagnostic = (
                            None
                            if valid_request_id == raw_request_id
                            else f"{type(raw_request_id).__name__}:{str(raw_request_id)[:256]}"
                        )
                        self._event(
                            "cohort_metadata_rejected",
                            request_id=valid_request_id,
                            reported_request_id=diagnostic,
                            reason=str(exc)[:512],
                        )
                    await self._send_error(ws, str(exc))
                    continue
                try:
                    overhead = self.settings.overhead_profile
                    if overhead is not None:
                        count = str(len(self._connections))
                        delays = overhead.data["before_queue_ns_by_connection_count"]
                        if count not in delays:
                            raise ValueError(
                                "overhead profile lacks current connection-count coverage"
                            )
                        await self._overhead_wait(request, "before_queue", delays[count])
                        request.queued_ns = time.monotonic_ns()
                    await self._enqueue(request)
                except QueueFull as exc:
                    await self._send_error(ws, str(exc))
                    continue
                except ValueError as exc:
                    await self._send_error(ws, str(exc))
                    continue
                # Keep the single receive task live while this request is active. Any
                # second message before its reply is a protocol violation.
                recv_task = asyncio.create_task(ws.recv())
                done, _ = await asyncio.wait(
                    (request.future, recv_task), return_when=asyncio.FIRST_COMPLETED
                )
                if recv_task in done:
                    removed = await self._remove_pending(request, "disconnect_or_pipeline")
                    try:
                        pipelined = recv_task.result()
                    except ConnectionClosed:
                        if not removed:
                            self._event(
                                "client_disconnected_active",
                                request_id=request.request_id,
                                session_id=request.session_id,
                            )
                        return
                    if pipelined is not None:
                        self._event(
                            "protocol_violation",
                            request_id=request.request_id,
                            session_id=request.session_id,
                            reason="pipelined_frame",
                        )
                        await ws.close(code=1008, reason="one outstanding request per connection")
                    return
                reply_task = asyncio.create_task(
                    self._deliver(ws, request), name=f"emulator-reply-{request.request_id}"
                )
                done, _ = await asyncio.wait(
                    (reply_task, recv_task), return_when=asyncio.FIRST_COMPLETED
                )
                if recv_task in done:
                    reply_task.cancel()
                    await asyncio.gather(reply_task, return_exceptions=True)
                    removed = await self._remove_pending(request, "disconnect_or_pipeline")
                    try:
                        pipelined = recv_task.result()
                    except ConnectionClosed:
                        if not removed:
                            self._event(
                                "client_disconnected_active",
                                request_id=request.request_id,
                                session_id=request.session_id,
                            )
                        return
                    if pipelined is not None:
                        self._event(
                            "protocol_violation",
                            request_id=request.request_id,
                            session_id=request.session_id,
                            reason="pipelined_frame",
                        )
                        await ws.close(code=1008, reason="one outstanding request per connection")
                    return
                try:
                    await reply_task
                except (Exception, ConnectionClosed):
                    return
                finally:
                    recv_task.cancel()
                    await asyncio.gather(recv_task, return_exceptions=True)
                    recv_task = None
        except ConnectionClosed:
            return
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            await self._abort(exc)
        finally:
            self._connections.discard(ws)
            self._connection_ids.discard(conn_id)
            for session in self._sessions_by_connection.pop(conn_id, set()):
                self._sequence_by_session.pop((conn_id, session), None)
            if recv_task is not None:
                recv_task.cancel()

    async def _deliver(self, ws: Any, request: ScheduledRequest) -> None:
        response = request.future.result()
        overhead = self.settings.overhead_profile
        if overhead is not None:
            await self._overhead_wait(
                request,
                "after_service",
                overhead.data["after_service_ns_by_batch_size"][str(request.executed_batch_size)],
            )
        faults = request.faults or []
        for fault in faults:
            if fault["kind"] in {
                "reply_delay",
                "disconnect",
                "protocol_error",
                "invalid_action_shape",
            }:
                self._event(
                    "fault_applied",
                    request_id=request.request_id,
                    fault_kind=fault["kind"],
                    delay_ms=fault.get("delay_ms"),
                )
        if any(item["kind"] == "disconnect" for item in faults):
            await ws.close(code=1011, reason="configured emulator fault")
            return
        delay_ms = max(
            (float(item.get("delay_ms", 0)) for item in faults if item["kind"] == "reply_delay"),
            default=0.0,
        )
        if delay_ms:
            await asyncio.sleep(delay_ms / 1000)
        if any(item["kind"] == "invalid_action_shape" for item in faults):
            response = self._corrupt_response(response)
        await asyncio.wait_for(ws.send(pack(response)), timeout=self.idle_timeout)
        self._event(
            "reply_sent",
            request_id=request.request_id,
            session_id=request.session_id,
            forward_id=request.forward_id,
            reply_sent_ns=time.monotonic_ns(),
        )

    async def _overhead_wait(self, request: ScheduledRequest, stage: str, target_ns: int) -> None:
        start = time.monotonic_ns()
        if target_ns:
            await asyncio.sleep(target_ns / 1e9)
        self._event(
            "overhead_applied",
            request_id=request.request_id,
            stage=stage,
            target_ns=target_ns,
            actual_ns=time.monotonic_ns() - start,
            started_ns=start,
            profile_sha256=self.settings.overhead_profile.sha256,
        )

    async def _remove_pending(self, request: ScheduledRequest, reason: str) -> bool:
        async with self._condition:
            removed = self.queue.remove(request)
            if removed:
                if not request.future.done():
                    request.future.cancel()
                if request.cohort:
                    self.registry.unregister(request.cohort)
                self._state_version += 1
                self._condition.notify_all()
        if removed:
            self._event("queue_leave", request_id=request.request_id, reason=reason)
        return removed

    def _make_request(self, connection_id: str, payload: Any, ws: Any) -> ScheduledRequest:
        if not isinstance(payload, dict):
            raise ValueError("request must be a mapping")
        session = payload.get("session_id", connection_id)
        request_id = payload.get("request_id", uuid.uuid4().hex)
        if not isinstance(session, str) or not session or len(session) > 256:
            raise ValueError("session_id must be a nonempty string of at most 256 characters")
        if not isinstance(request_id, str) or not request_id or len(request_id) > 256:
            raise ValueError("request_id must be a nonempty string of at most 256 characters")
        if "reset" in payload:
            raise ValueError("reset messages are only valid while idle")
        self._connection_ids.add(connection_id)
        observation = dict(payload)
        cohort = observation.pop(COHORT_FIELD, None)
        clean = {
            key: value
            for key, value in observation.items()
            if key not in ("session_id", "request_id")
        }
        validate_observation(self.settings.model, clean, self._feature_args())
        sampling = payload.get("sampling_params", {})
        self._validate_sampling(sampling)
        if cohort is not None:
            if not self.cohort_enabled:
                raise ValueError("cohort requests are disabled")
            cohort = validate_cohort(
                cohort,
                request_id=request_id,
                policy_id=self.cohort_policy,
                max_batch_size=self.max_batch_size,
            )
        loop = asyncio.get_running_loop()
        session_key = (connection_id, session)
        connection_sessions = self._sessions_by_connection.setdefault(connection_id, set())
        if session_key not in self._sequence_by_session and len(connection_sessions) >= int(
            self.settings.data["transport"]["max_sessions_per_connection"]
        ):
            raise ValueError("session capacity exceeded for this connection")
        sequence = self._sequence_by_session.get(session_key, 0)
        self._sequence_by_session[session_key] = sequence + 1
        connection_sessions.add(session)
        faults = self._matching_faults(session, request_id, sequence)
        freeze_ms = max(
            (int(item["delay_ms"]) for item in faults if item["kind"] == "status_freeze"), default=0
        )
        if freeze_ms:
            self._status_freeze_until_ns = max(
                self._status_freeze_until_ns, time.monotonic_ns() + freeze_ms * 1_000_000
            )
            self._event(
                "fault_applied",
                request_id=request_id,
                fault_kind="status_freeze",
                delay_ms=freeze_ms,
                frozen_until_ns=self._status_freeze_until_ns,
            )
        protocol_fault = next((item for item in faults if item["kind"] == "protocol_error"), None)
        if protocol_fault:
            self._event("fault_applied", request_id=request_id, fault_kind="protocol_error")
            raise ValueError(str(protocol_fault.get("message", "configured protocol error")))
        received_ns = time.monotonic_ns()
        self._event(
            "request_received",
            request_id=request_id,
            session_id=session,
            request_received_ns=received_ns,
        )
        return ScheduledRequest(
            connection_id,
            request_id,
            session,
            clean,
            len(pack(payload)),
            received_ns,
            loop.create_future(),
            cohort,
            sequence,
            faults,
        )

    def _matching_faults(
        self, session: str, request_id: str, sequence: int
    ) -> list[dict[str, Any]]:
        profile = self.settings.fault_profile
        if not profile:
            return []
        matched = []
        for rule in profile.get("rules", []):
            selector_ok = (
                ("request_id" not in rule or rule["request_id"] == request_id)
                and ("session_id" not in rule or rule["session_id"] in ("*", session))
                and ("request_sequence" not in rule or rule["request_sequence"] == sequence)
            )
            if selector_ok:
                matched.append(rule)
        return matched

    @staticmethod
    def _corrupt_response(response: Any) -> Any:
        import numpy as np

        if isinstance(response, np.ndarray):
            return response[:-1]
        if isinstance(response, dict):
            result = dict(response)
            for key, value in result.items():
                if isinstance(value, np.ndarray):
                    result[key] = value[:-1]
                    return result
            result["unexpected_fault_field"] = None
            return result
        return response

    def _validate_sampling(self, sampling: Any) -> None:
        if sampling is None:
            return
        if not isinstance(sampling, dict):
            raise ValueError("sampling_params must be a mapping")
        allowed = {
            item.get("name")
            for item in self.settings.model.data.get("sampling_parameters", {}).get("fields", [])
        }
        if self.features.get("rtc_enabled"):
            allowed.add("rtc")
        unknown = set(sampling) - allowed
        if unknown:
            raise ValueError(f"unsupported sampling parameters: {sorted(unknown)}")

    async def _enqueue(self, request: ScheduledRequest) -> None:
        async with self._condition:
            self.queue.enqueue(request)
            try:
                if request.cohort:
                    self.registry.register(request.cohort)
                elif self.cohort_enabled:
                    self.registry.register_uncoordinated()
            except ValueError:
                self.queue.remove(request)
                raise
            self._state_version += 1
            self._event(
                "queue_enter",
                request_id=request.request_id,
                session_id=request.session_id,
                queued_ns=request.queued_ns,
                queued_request_ids=[item.request_id for item in self.queue.items],
            )
            self._condition.notify_all()

    async def _run_scheduler(self) -> None:
        try:
            while True:
                async with self._condition:
                    while not self.queue.items and not self._closing:
                        await self._condition.wait()
                    if self._closing and not self.queue.items:
                        return
                    first = self.queue.items[0]
                    epoch = first.cohort.get("epoch_id") if first.cohort else None
                    expected = (
                        len(first.cohort["member_ids"]) if first.cohort else self.max_batch_size
                    )
                    capacity = min(expected, self.max_batch_size)
                    timeout_deadline_ns = first.queued_ns + int(self.queue.batch_wait_s * 1e9)
                    cohort_deadline_ns = (
                        first.cohort["deadline_ns"] if first.cohort else timeout_deadline_ns
                    )
                    deadline_ns = min(timeout_deadline_ns, cohort_deadline_ns)

                    def prefix_count() -> int:
                        count = 0
                        for item in self.queue.items:
                            item_epoch = item.cohort.get("epoch_id") if item.cohort else None
                            if item_epoch != epoch or count >= capacity:
                                break
                            count += 1
                        return count

                    while self.queue.items and prefix_count() < capacity and not self._closing:
                        remaining = (deadline_ns - time.monotonic_ns()) / 1e9
                        if remaining <= 0:
                            break
                        try:
                            await asyncio.wait_for(self._condition.wait(), remaining)
                        except TimeoutError:
                            break
                    if not self.queue.items:
                        continue
                    batch = self.queue.pop_batch()
                await self._execute(batch)
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            await self._abort(exc)

    async def _execute(self, batch: list[ScheduledRequest]) -> None:
        if not batch:
            return
        start = time.monotonic_ns()
        batch_id = uuid.uuid4().hex
        self._active = {
            "forward_id": batch_id,
            "start_ns": start,
            "request_ids": [req.request_id for req in batch],
        }
        for req in batch:
            req.forward_id = batch_id
            req.executed_batch_size = len(batch)
            self._event(
                "queue_leave",
                request_id=req.request_id,
                queue_enter_ns=req.queued_ns,
                forward_id=batch_id,
            )
        self._event(
            "forward_start",
            forward_id=batch_id,
            request_ids=self._active["request_ids"],
            batch_size=len(batch),
            forward_start_ns=start,
        )
        if batch[0].cohort:
            cohort = batch[0].cohort
            expected_count = len(cohort["member_ids"])
            wait_timeout_ns = batch[0].queued_ns + int(self.queue.batch_wait_s * 1e9)
            close_reason = (
                "expected_count"
                if len(batch) >= expected_count
                else "max_batch"
                if len(batch) >= self.max_batch_size
                else "cohort_deadline"
                if cohort["deadline_ns"] <= wait_timeout_ns
                else "configured_timeout"
            )
            self.registry.close(cohort["epoch_id"])
            self._event(
                "cohort_close",
                schema="vllm_omni.cohort_receipt.v1",
                epoch_id=cohort["epoch_id"],
                membership_hash=cohort["membership_hash"],
                expected_count=expected_count,
                arrived_count=len(batch),
                missing_member_ids=[
                    member
                    for member in cohort["member_ids"]
                    if member not in {req.cohort["member_id"] for req in batch}
                ],
                close_ns=start,
                close_reason=close_reason,
                first_queue_enter_ns=min(req.queued_ns for req in batch),
                last_queue_enter_ns=max(req.queued_ns for req in batch),
                configured_wait_ns=max(0, start - batch[0].queued_ns),
                executed_count=len(batch),
                forward_start_ns=start,
                metadata_validation_reason=None,
            )
        self._publish_status()
        per_size = self._batch_sequence.get(len(batch), 0)
        self._batch_sequence[len(batch)] = per_size + 1
        sample = choose_sample(
            self.settings.service_profile,
            batch_size=len(batch),
            per_size_sequence=per_size,
            seed=self.seed,
        )
        delay_ns = max(
            (
                int(float(fault.get("delay_ms", 0)) * 1_000_000)
                for req in batch
                for fault in (req.faults or [])
                if fault["kind"] == "service_delay"
            ),
            default=0,
        )
        for req in batch:
            for fault in req.faults or []:
                if fault["kind"] == "service_delay":
                    self._event(
                        "fault_applied",
                        request_id=req.request_id,
                        fault_kind=fault["kind"],
                        delay_ms=fault.get("delay_ms"),
                        applied_delay_ns=delay_ns,
                    )
        target_ns = int(sample.active_ns * self.scale) + delay_ns
        generated: list[Any] = []
        try:
            for req in batch:
                output = make_actions(
                    self.settings.model,
                    req.observation,
                    {
                        "session_id": req.session_id,
                        "request_id": req.request_id,
                        "request_sequence": req.sequence,
                        "session_sequence": req.sequence,
                        "batch_id": batch_id,
                        "batch_index": len(generated),
                    },
                    self._feature_args(),
                )
                generated.append(output)
            remaining = target_ns - (time.monotonic_ns() - start)
            if remaining > 0:
                await asyncio.sleep(remaining / 1e9)
            end = time.monotonic_ns()
            for req, output in zip(batch, generated, strict=True):
                if not req.future.done():
                    req.future.set_result(output)
            self._event(
                "forward_end",
                forward_id=batch_id,
                request_ids=[req.request_id for req in batch],
                batch_size=len(batch),
                sample_id=sample.sample_id,
                target_active_ns=target_ns,
                actual_active_ns=end - start,
                forward_start_ns=start,
                forward_end_ns=end,
                overshoot_ns=max(0, end - start - target_ns),
            )
        except BaseException as exc:
            for req in batch:
                if not req.future.done():
                    req.future.set_exception(exc)
            raise
        finally:
            self._active = None
            self._publish_status()

    async def _run_heartbeat(self) -> None:
        interval = float(self.settings.data["evidence"]["status_interval_ms"]) / 1000
        while True:
            self._publish_status()
            await asyncio.sleep(interval)

    async def _run_evidence_writer(self) -> None:
        interval = float(self.settings.data["evidence"]["flush_interval_ms"]) / 1000
        try:
            while True:
                await asyncio.sleep(interval)
                await asyncio.to_thread(self.evidence.flush_events)
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            await self._abort(exc)

    def _status(self) -> dict[str, Any]:
        observed = time.monotonic_ns()
        return {
            "schema": "omni_scheduler_progress.v1",
            "clock_domain": "host_monotonic_ns",
            "hostname": socket.gethostname(),
            "observed_ns": observed,
            "instance_id": self._instance_id,
            "active": dict(self._active) if self._active else None,
            "queued_request_ids": [item.request_id for item in self.queue.items],
            "recent_events": [
                event
                for event in self.evidence.recent
                if event["kind"]
                in {
                    "queue_enter",
                    "queue_leave",
                    "cohort_close",
                    "cohort_metadata_rejected",
                    "forward_start",
                    "forward_end",
                }
            ],
        }

    def _publish_status(self, *, force: bool = False) -> None:
        try:
            if not force and time.monotonic_ns() < self._status_freeze_until_ns:
                return
            self._status_version += 1
            if force:
                self._force_status_version = self._status_version
            if self._status_task is None or self._status_task.done():
                self._status_task = asyncio.create_task(self._drain_status())
        except Exception as exc:
            self._set_failure(exc)

    async def _drain_status(self) -> None:
        while self._status_written_version < self._status_version:
            version = self._status_version
            if (
                version > self._force_status_version
                and time.monotonic_ns() < self._status_freeze_until_ns
            ):
                return
            status = self._status()
            try:
                await asyncio.to_thread(self.evidence.snapshot, status)
                if self.status_path:
                    await asyncio.to_thread(self.evidence.snapshot, status, self.status_path)
            except Exception as exc:
                self._set_failure(exc)
                return
            self._status_written_version = version

    async def _flush_status(self) -> None:
        while self._status_written_version < self._status_version:
            if self._failure is not None:
                raise RuntimeError("status writer failed") from self._failure
            if self._status_task is None or self._status_task.done():
                self._status_task = asyncio.create_task(self._drain_status())
            await self._status_task

    async def wait_failed(self) -> BaseException:
        await self._failure_event.wait()
        assert self._failure is not None
        return self._failure

    def _set_failure(self, error: BaseException) -> None:
        self._failure = error
        self._failure_event.set()

    def _event(self, kind: str, **fields: Any) -> None:
        self.evidence.event(kind, time.monotonic_ns(), **fields)

    async def _send_error(self, ws: Any, message: str) -> None:
        try:
            await ws.send(pack({"type": "error", "message": message[:512]}))
        except Exception:
            pass

    async def _abort(self, exc: BaseException) -> None:
        self._set_failure(exc)
        self._closing = True
        async with self._condition:
            for request in self.queue.items:
                if not request.future.done():
                    request.future.set_exception(RuntimeError("emulator aborted"))
            self.queue.items.clear()
            self.queue.bytes = 0
            self._condition.notify_all()

    def _feature_args(self) -> dict[str, Any]:
        return {
            "rtc_enabled": bool(self.features.get("rtc_enabled", False)),
            "cohort_enabled": self.cohort_enabled,
            "emulator_marker": {
                "synthetic": True,
                "schema": "policy_server_emulator.v1",
                "instance_id": self._instance_id,
                "configuration_sha256": self.settings.sha256,
                "model_profile_sha256": self.settings.model.sha256,
                "hardware_profile_sha256": self.settings.hardware.sha256,
                "service_profile_sha256": self.settings.service_profile.sha256,
            },
        }
