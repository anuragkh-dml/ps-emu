from __future__ import annotations

import pytest

from ps_emu.scheduler import BoundedBatchQueue, QueueFull, ScheduledRequest


def _request(name: str, size: int = 1, cohort: str | None = None) -> ScheduledRequest:
    return ScheduledRequest(
        "c", name, "s", {}, size, 1, None, ({"epoch_id": cohort} if cohort else None)
    )


def test_queue_enforces_count_and_byte_budgets_and_fifo() -> None:
    queue = BoundedBatchQueue(2, 4, 2, 0)
    a, b = _request("a", 2), _request("b", 2)
    queue.enqueue(a)
    queue.enqueue(b)
    with pytest.raises(QueueFull):
        queue.enqueue(_request("c", 1))
    assert [request.request_id for request in queue.pop_batch()] == ["a", "b"]
    assert queue.bytes == 0


def test_queue_batches_only_contiguous_matching_cohort_members() -> None:
    queue = BoundedBatchQueue(5, 10, 4, 0)
    for request in (
        _request("a", cohort="epoch-a"),
        _request("b", cohort="epoch-a"),
        _request("x"),
        _request("c", cohort="epoch-b"),
    ):
        queue.enqueue(request)
    assert [r.request_id for r in queue.pop_batch()] == ["a", "b"]
    assert [r.request_id for r in queue.pop_batch()] == ["x"]


def test_removal_accounts_bytes() -> None:
    queue = BoundedBatchQueue(3, 10, 2, 0)
    item = _request("a", 5)
    queue.enqueue(item)
    assert queue.remove(item)
    assert queue.bytes == 0 and not queue.items
