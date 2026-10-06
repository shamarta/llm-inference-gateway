"""Tests for the dynamic batching engine."""

import asyncio
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager

import pytest

from gateway.api.schemas import ChatCompletionRequest, ChatCompletionResponse, ChatMessage
from gateway.infrastructure.mock_backend import MockLLMBackend
from gateway.services.batcher import BatcherNotRunningError, DynamicBatcher


def make_request(text: str) -> ChatCompletionRequest:
    return ChatCompletionRequest(model="m", messages=[ChatMessage(role="user", content=text)])


def fast_backend() -> MockLLMBackend:
    return MockLLMBackend(base_latency_ms=0, per_item_latency_ms=0)


@asynccontextmanager
async def running(batcher: DynamicBatcher) -> AsyncIterator[DynamicBatcher]:
    await batcher.start()
    try:
        yield batcher
    finally:
        await batcher.stop()


class FlakyBackend:
    """Fails the first `fail_times` calls, then behaves normally."""

    def __init__(self, fail_times: int = 1) -> None:
        self.remaining_failures = fail_times
        self._inner = fast_backend()

    async def generate_batch(
        self, requests: Sequence[ChatCompletionRequest]
    ) -> list[ChatCompletionResponse]:
        if self.remaining_failures > 0:
            self.remaining_failures -= 1
            raise RuntimeError("boom")
        return await self._inner.generate_batch(requests)


class ShortBackend:
    """Returns fewer responses than requests (a contract violation)."""

    async def generate_batch(
        self, requests: Sequence[ChatCompletionRequest]
    ) -> list[ChatCompletionResponse]:
        return []


async def test_flushes_immediately_when_max_batch_size_reached() -> None:
    backend = fast_backend()
    batcher = DynamicBatcher(backend, max_batch_size=4, batch_timeout_ms=500)
    async with running(batcher):
        await asyncio.gather(*(batcher.submit(make_request(f"p{i}")) for i in range(4)))
    assert backend.batch_sizes == [4]


async def test_flushes_on_timeout_with_partial_batch() -> None:
    backend = fast_backend()
    batcher = DynamicBatcher(backend, max_batch_size=8, batch_timeout_ms=20)
    async with running(batcher):
        await asyncio.gather(*(batcher.submit(make_request(f"p{i}")) for i in range(3)))
    assert backend.batch_sizes == [3]


async def test_splits_requests_exceeding_max_batch_size() -> None:
    backend = fast_backend()
    batcher = DynamicBatcher(backend, max_batch_size=4, batch_timeout_ms=50)
    async with running(batcher):
        await asyncio.gather(*(batcher.submit(make_request(f"p{i}")) for i in range(10)))
    assert sorted(backend.batch_sizes) == [2, 4, 4]


async def test_each_caller_gets_its_own_response() -> None:
    batcher = DynamicBatcher(fast_backend(), max_batch_size=8, batch_timeout_ms=20)
    async with running(batcher):
        responses = await asyncio.gather(*(batcher.submit(make_request(f"q{i}")) for i in range(5)))
    contents = [r.choices[0].message.content for r in responses]
    assert contents == [f"Mock response to: q{i}" for i in range(5)]


async def test_backend_error_fails_batch_then_batcher_recovers() -> None:
    batcher = DynamicBatcher(FlakyBackend(fail_times=1), max_batch_size=2, batch_timeout_ms=20)
    async with running(batcher):
        results = await asyncio.gather(
            batcher.submit(make_request("a")),
            batcher.submit(make_request("b")),
            return_exceptions=True,
        )
        assert all(isinstance(r, RuntimeError) for r in results)

        ok = await batcher.submit(make_request("c"))
        assert ok.choices[0].message.content == "Mock response to: c"


async def test_response_count_mismatch_raises() -> None:
    batcher = DynamicBatcher(ShortBackend(), batch_timeout_ms=10)
    async with running(batcher):
        with pytest.raises(RuntimeError, match="returned 0 responses"):
            await batcher.submit(make_request("a"))


async def test_submit_before_start_raises() -> None:
    batcher = DynamicBatcher(fast_backend())
    with pytest.raises(BatcherNotRunningError):
        await batcher.submit(make_request("a"))


async def test_stop_fails_inflight_requests() -> None:
    slow = MockLLMBackend(base_latency_ms=10_000, per_item_latency_ms=0)
    batcher = DynamicBatcher(slow, batch_timeout_ms=10)
    await batcher.start()
    task = asyncio.create_task(batcher.submit(make_request("a")))
    await asyncio.sleep(0.1)  # let the batch be dispatched to the slow backend
    await batcher.stop()
    with pytest.raises(RuntimeError, match="batcher stopped"):
        await task


async def test_on_batch_callback_receives_sizes() -> None:
    sizes: list[int] = []
    batcher = DynamicBatcher(
        fast_backend(), max_batch_size=3, batch_timeout_ms=20, on_batch=sizes.append
    )
    async with running(batcher):
        await asyncio.gather(*(batcher.submit(make_request(f"p{i}")) for i in range(3)))
    assert sizes == [3]


async def test_cancelled_caller_does_not_break_batch() -> None:
    batcher = DynamicBatcher(fast_backend(), max_batch_size=8, batch_timeout_ms=50)
    async with running(batcher):
        cancelled = asyncio.create_task(batcher.submit(make_request("a")))
        survivor = asyncio.create_task(batcher.submit(make_request("b")))
        await asyncio.sleep(0.01)
        cancelled.cancel()

        response = await survivor
        assert response.choices[0].message.content == "Mock response to: b"
        with pytest.raises(asyncio.CancelledError):
            await cancelled


async def test_start_is_idempotent_and_stop_without_start_is_safe() -> None:
    batcher = DynamicBatcher(fast_backend())
    await batcher.stop()
    await batcher.start()
    await batcher.start()
    assert batcher.queue_size == 0
    await batcher.stop()


@pytest.mark.parametrize(
    "kwargs",
    [{"max_batch_size": 0}, {"batch_timeout_ms": 0}, {"max_concurrent_batches": 0}],
)
def test_invalid_parameters_rejected(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        DynamicBatcher(fast_backend(), **kwargs)
