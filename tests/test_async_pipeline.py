import asyncio
import random

from bridgenlp.base import BridgeBase
from bridgenlp.pipeline_async import AsyncPipeline
from bridgenlp.result import BridgeResult
from bridgenlp.config import BridgeConfig


class AsyncDelayedAdapter(BridgeBase):
    def __init__(self, name="mock", max_delay=0.05, config=None):
        super().__init__(config)
        self.name = name
        self.max_delay = max_delay
        self.calls = 0
        self.lock = asyncio.Lock()

    async def from_text(self, text):
        async with self.lock:
            self.calls += 1
        await asyncio.sleep(random.uniform(0, self.max_delay))
        return BridgeResult(tokens=text.split())

    async def from_tokens(self, tokens):
        async with self.lock:
            self.calls += 1
        await asyncio.sleep(random.uniform(0, self.max_delay))
        return BridgeResult(tokens=list(tokens))

    async def from_spacy(self, doc):
        async with self.lock:
            self.calls += 1
        await asyncio.sleep(random.uniform(0, self.max_delay))
        return BridgeResult(tokens=[t.text for t in doc]).attach_to_spacy(doc)


def test_async_pipeline_concurrent():
    async def run():
        adapters = [AsyncDelayedAdapter(name=f"a{i}") for i in range(3)]
        config = BridgeConfig(cache_results=True, cache_size=20)
        async with AsyncPipeline(adapters, config) as pipeline:
            async def worker(idx):
                results = []
                for i in range(5):
                    text = f"test {idx} {i}"
                    results.append(await pipeline.from_text(text))
                return results

            tasks = [asyncio.create_task(worker(i)) for i in range(5)]
            results = await asyncio.gather(*tasks)
        return adapters, results

    adapters, results = asyncio.run(run())
    assert len(results) == 5
    for res in results:
        assert len(res) == 5
    # ensure adapters were called
    for ad in adapters:
        assert ad.calls >= 5
