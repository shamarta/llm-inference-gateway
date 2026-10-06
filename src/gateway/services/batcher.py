"""Dynamic batching engine: groups concurrent requests into one backend call."""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass

from gateway.api.schemas import ChatCompletionRequest, ChatCompletionResponse
from gateway.domain.interfaces import LLMBackend


class BatcherNotRunningError(RuntimeError):
    """Raised when a request is submitted to a batcher that is not running."""


@dataclass(slots=True)
class _PendingItem:
    """A queued request together with the future its caller is awaiting."""

    request: ChatCompletionRequest
    future: asyncio.Future[ChatCompletionResponse]


class DynamicBatcher:
    """Aggregates requests until `max_batch_size` is reached or `batch_timeout_ms` elapses."""

    def __init__(
        self,
        backend: LLMBackend,
        max_batch_size: int = 8,
        batch_timeout_ms: int = 20,
        max_concurrent_batches: int = 4,
        on_batch: Callable[[int], None] | None = None,
    ) -> None:
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be >= 1")
        if batch_timeout_ms < 1:
            raise ValueError("batch_timeout_ms must be >= 1")
        if max_concurrent_batches < 1:
            raise ValueError("max_concurrent_batches must be >= 1")

        self._backend = backend
        self._max_batch_size = max_batch_size
        self._timeout_s = batch_timeout_ms / 1000
        self._on_batch: Callable[[int], None] = on_batch or (lambda _size: None)

        self._queue: asyncio.Queue[_PendingItem] = asyncio.Queue()
        self._semaphore = asyncio.Semaphore(max_concurrent_batches)
        self._inflight: set[asyncio.Task[None]] = set()
        self._worker: asyncio.Task[None] | None = None
        self._stopping = False

    @property
    def queue_size(self) -> int:
        """Number of requests waiting to be batched."""
        return self._queue.qsize()

    async def start(self) -> None:
        """Start the background worker (idempotent)."""
        if self._worker is not None:
            return
        self._stopping = False
        self._worker = asyncio.get_running_loop().create_task(self._run())

    async def stop(self) -> None:
        """Stop the worker; every still-pending request fails with RuntimeError."""
        if self._worker is None:
            return
        self._stopping = True
        tasks = [self._worker, *self._inflight]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

        while not self._queue.empty():
            self._fail([self._queue.get_nowait()], RuntimeError("batcher stopped"))
        self._worker = None

    async def submit(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Queue a request and wait for its response."""
        if self._worker is None or self._stopping:
            raise BatcherNotRunningError("batcher is not running")
        future: asyncio.Future[ChatCompletionResponse] = asyncio.get_running_loop().create_future()
        self._queue.put_nowait(_PendingItem(request=request, future=future))
        return await future

    async def _run(self) -> None:
        """Worker loop: collect a batch, then dispatch it without blocking the next one."""
        loop = asyncio.get_running_loop()
        while True:
            batch: list[_PendingItem] = []
            try:
                await self._fill_batch(batch)
                await self._semaphore.acquire()
            except asyncio.CancelledError:
                self._fail(batch, RuntimeError("batcher stopped"))
                raise
            task = loop.create_task(self._process(batch))
            self._inflight.add(task)
            task.add_done_callback(self._inflight.discard)

    async def _fill_batch(self, batch: list[_PendingItem]) -> None:
        """Block for the first item, then keep adding until size or deadline is hit."""
        loop = asyncio.get_running_loop()
        batch.append(await self._queue.get())
        deadline = loop.time() + self._timeout_s
        while len(batch) < self._max_batch_size:
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            try:
                batch.append(await asyncio.wait_for(self._queue.get(), timeout=remaining))
            except TimeoutError:
                break

    async def _process(self, batch: list[_PendingItem]) -> None:
        """Send one batch to the backend and resolve each caller's future."""
        try:
            self._on_batch(len(batch))
            responses = await self._backend.generate_batch([item.request for item in batch])
            if len(responses) != len(batch):
                raise RuntimeError(
                    f"backend returned {len(responses)} responses for {len(batch)} requests"
                )
            for item, response in zip(batch, responses, strict=True):
                if not item.future.done():  # the caller may have been cancelled
                    item.future.set_result(response)
        except asyncio.CancelledError:
            self._fail(batch, RuntimeError("batcher stopped"))
            raise
        except Exception as exc:
            self._fail(batch, exc)
        finally:
            self._semaphore.release()

    @staticmethod
    def _fail(batch: list[_PendingItem], exc: BaseException) -> None:
        for item in batch:
            if not item.future.done():
                item.future.set_exception(exc)
