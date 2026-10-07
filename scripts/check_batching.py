"""Manual check: 20 concurrent requests are grouped into a few backend calls."""

import asyncio
import time

from gateway.api.schemas import ChatCompletionRequest, ChatMessage
from gateway.infrastructure.mock_backend import MockLLMBackend
from gateway.services.batcher import DynamicBatcher

TOTAL_REQUESTS = 20
BASE_LATENCY_MS = 150
PER_ITEM_LATENCY_MS = 10


def make_request(index: int) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="demo", messages=[ChatMessage(role="user", content=f"question {index}")]
    )


async def main() -> None:
    backend = MockLLMBackend(BASE_LATENCY_MS, PER_ITEM_LATENCY_MS)
    batcher = DynamicBatcher(backend, max_batch_size=8, batch_timeout_ms=20)
    await batcher.start()

    started = time.perf_counter()
    responses = await asyncio.gather(
        *(batcher.submit(make_request(i)) for i in range(TOTAL_REQUESTS))
    )
    elapsed_ms = (time.perf_counter() - started) * 1000
    await batcher.stop()

    unbatched_ms = TOTAL_REQUESTS * (BASE_LATENCY_MS + PER_ITEM_LATENCY_MS)
    print(f"requests sent:        {TOTAL_REQUESTS}")
    print(f"backend calls:        {backend.call_count}")
    print(f"batch sizes:          {sorted(backend.batch_sizes, reverse=True)}")
    print(f"total time:           {elapsed_ms:.0f} ms")
    print(f"one-by-one estimate:  {unbatched_ms} ms")
    print(f"first answer:         {responses[0].choices[0].message.content}")
    print(f"last answer:          {responses[-1].choices[0].message.content}")


if __name__ == "__main__":
    asyncio.run(main())
