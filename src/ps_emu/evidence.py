"""Bounded durable JSONL events, atomic progress snapshots, and receipts."""

from __future__ import annotations

import json
import os
import tempfile
from threading import Lock
from collections import deque
from pathlib import Path
from typing import Any


class EvidenceWriter:
    def __init__(
        self, output_dir: str | Path, *, recent_capacity: int = 256, writer_capacity: int = 4096
    ) -> None:
        self.root = Path(output_dir)
        self.events_path = self.root / "events.jsonl"
        self.status_path = self.root / "status.json"
        self.recent: deque[dict[str, Any]] = deque(maxlen=recent_capacity)
        self.writer_capacity = writer_capacity
        self.pending: deque[str] = deque()
        self.event_count = 0
        self._lock = Lock()
        self._flush_lock = Lock()

    def event(self, kind: str, at_ns: int, **fields: Any) -> dict[str, Any]:
        item = {"kind": kind, "at_ns": at_ns, **fields}
        encoded = json.dumps(item, sort_keys=True, separators=(",", ":"), allow_nan=False)
        with self._lock:
            if len(self.pending) >= self.writer_capacity:
                raise RuntimeError("evidence event capacity exceeded")
            self.pending.append(encoded)
            self.event_count += 1
            self.recent.append(item)
        return item

    def flush_events(self) -> None:
        with self._flush_lock:
            with self._lock:
                if not self.pending:
                    return
                batch = list(self.pending)
            self.root.mkdir(parents=True, exist_ok=True)
            with self.events_path.open("a", encoding="utf-8") as stream:
                stream.write("\n".join(batch) + "\n")
                stream.flush()
            with self._lock:
                for _ in batch:
                    self.pending.popleft()

    def snapshot(self, value: dict[str, Any], path: Path | None = None) -> None:
        target = path or self.status_path
        target.parent.mkdir(parents=True, exist_ok=True)
        data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
        except BaseException:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise

    def initial_receipt(self, payload: dict[str, Any]) -> None:
        self.snapshot(payload, self.root / "readiness.json")

    def receipt(self, payload: dict[str, Any]) -> None:
        self.snapshot(payload, self.root / "final_receipt.json")
