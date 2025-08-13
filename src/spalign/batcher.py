"""vLLM async batcher for efficient inference."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from vllm import LLM, SamplingParams


@dataclass
class _Request:
    """Internal request structure for batching."""

    prompt: str
    future: asyncio.Future[str]


class VLLMBatcher:
    """Collect prompts, run vLLM generate() in bulk, return result to futures."""

    def __init__(self, llm: LLM, max_batch: int = 32, max_latency_ms: int = 25):
        self.llm = llm
        self.queue: asyncio.Queue[_Request] = asyncio.Queue()
        self.max_batch = max_batch
        self.max_latency = max_latency_ms / 1000.0

    async def put(self, prompt: str) -> asyncio.Future[str]:
        """Submit a prompt for batched generation."""
        fut: asyncio.Future[str] = asyncio.get_event_loop().create_future()
        await self.queue.put(_Request(prompt, fut))
        return fut

    async def _runner(self):
        """Background task that actually calls vLLM."""
        while True:
            # Guarantee at least one item.
            req = await self.queue.get()
            prompts = [req.prompt]
            futures = [req.future]
            start = asyncio.get_event_loop().time()
            # Collect until latency or max_batch reached.
            while len(prompts) < self.max_batch:
                timeout = self.max_latency - (asyncio.get_event_loop().time() - start)
                if timeout <= 0:
                    break
                try:
                    req = await asyncio.wait_for(self.queue.get(), timeout)
                    prompts.append(req.prompt)
                    futures.append(req.future)
                except asyncio.TimeoutError:
                    break

            # Generate in bulk (blocking CPU thread; offload to default executor).
            loop = asyncio.get_running_loop()
            try:
                outs = await loop.run_in_executor(
                    None,
                    lambda: self.llm.generate(prompts, self._get_sampling_params()),
                )
                # Send results back.
                for fut, out in zip(futures, outs):
                    if not fut.cancelled():
                        fut.set_result(out.outputs[0].text.strip())
            except Exception as e:
                print(f"vLLM generate error: {e}")
                # Return empty string for all futures in this batch
                for fut in futures:
                    if not fut.cancelled():
                        fut.set_result("")

    def _get_sampling_params(self) -> SamplingParams:
        """Get default sampling parameters."""
        return SamplingParams(
            temperature=0.15,
            top_p=0.9,
            top_k=32,
            max_tokens=128,
            skip_special_tokens=False,
        )

    def start(self):
        """Start the background batching task."""
        print("VLLMBatcher: バックグラウンドタスクを開始中...")
        task = asyncio.create_task(self._runner())
        print(f"VLLMBatcher: タスク作成完了: {task}")
        return task
