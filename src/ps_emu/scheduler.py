"""Small bounded FIFO batch collector used by the emulator runtime."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any


class QueueFull(RuntimeError):
    """The configured request or byte budget is exhausted."""


@dataclass
class ScheduledRequest:
    connection_id: str
    request_id: str
    session_id: str
    observation: dict[str, Any]
    payload_bytes: int
    queued_ns: int
    future: Any
    cohort: dict[str, Any] | None = None
    sequence: int = 0
    faults: list[dict[str, Any]] | None = None
    executed_batch_size: int = 0


class BoundedBatchQueue:
    """Count- and byte-bounded FIFO; the first item fixes each batch deadline."""

    def __init__(
        self, max_requests: int, max_bytes: int, max_batch_size: int, batch_wait_s: float
    ) -> None:
        self.max_requests, self.max_bytes = max_requests, max_bytes
        self.max_batch_size, self.batch_wait_s = max_batch_size, batch_wait_s
        self.items: deque[ScheduledRequest] = deque()
        self.bytes = 0

    def enqueue(self, request: ScheduledRequest) -> None:
        if (
            len(self.items) >= self.max_requests
            or self.bytes + request.payload_bytes > self.max_bytes
        ):
            raise QueueFull("emulator pending queue capacity exceeded")
        self.items.append(request)
        self.bytes += request.payload_bytes

    def remove(self, request: ScheduledRequest) -> bool:
        try:
            self.items.remove(request)
        except ValueError:
            return False
        self.bytes -= request.payload_bytes
        return True

    def pop_batch(self) -> list[ScheduledRequest]:
        if not self.items:
            return []
        first = self.items[0]
        epoch = first.cohort.get("epoch_id") if first.cohort else None
        count = 0
        for item in self.items:
            item_epoch = item.cohort.get("epoch_id") if item.cohort else None
            if item_epoch != epoch or count >= self.max_batch_size:
                break
            count += 1
        batch = [self.items.popleft() for _ in range(count)]
        self.bytes -= sum(item.payload_bytes for item in batch)
        return batch
