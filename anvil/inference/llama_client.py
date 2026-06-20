import asyncio
import logging
import time
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)


class LlamaServerError(Exception):
    pass


@dataclass
class CompletionResult:
    content: str
    tokens_generated: int
    tokens_evaluated: int
    timing_ms: float
    success: bool
    error: str | None = None


@dataclass
class LlamaClientConfig:
    host: str = "127.0.0.1"
    port: int = 8081
    timeout_s: float = 120.0
    max_retries: int = 2


class LlamaClient:
    def __init__(self, config: LlamaClientConfig | None = None):
        self.config = config or LlamaClientConfig()
        self._base_url = f"http://{self.config.host}:{self.config.port}"
        self._client: httpx.AsyncClient | None = None

    async def _ensure_client(self):
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=httpx.Timeout(self.config.timeout_s),
            )

    async def _request(self, method: str, path: str, **kwargs):
        await self._ensure_client()
        for attempt in range(self.config.max_retries + 1):
            try:
                resp = await self._client.request(method, path, **kwargs)
                resp.raise_for_status()
                return resp.json()
            except httpx.HTTPStatusError as e:
                if attempt < self.config.max_retries and e.response.status_code >= 500:
                    await asyncio.sleep(0.5 * (attempt + 1))
                    continue
                raise LlamaServerError(f"HTTP {e.response.status_code}: {e.response.text}") from e
            except httpx.RequestError as e:
                if attempt < self.config.max_retries:
                    await asyncio.sleep(1.0 * (attempt + 1))
                    continue
                raise LlamaServerError(f"Request failed: {e}") from e

    async def health(self) -> bool:
        try:
            data = await self._request("GET", "/health")
            return data.get("status") == "ok"
        except LlamaServerError:
            return False

    async def wait_until_ready(self, timeout_s: float = 30.0, interval_s: float = 1.0) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if await self.health():
                return True
            await asyncio.sleep(interval_s)
        return False

    async def complete(
        self,
        prompt: str,
        *,
        n_predict: int = 512,
        temperature: float = 0.1,
        grammar: str | None = None,
        cache_prompt: bool = True,
        stop: list[str] | None = None,
    ) -> CompletionResult:
        body = {
            "prompt": prompt,
            "n_predict": n_predict,
            "temperature": temperature,
            "cache_prompt": cache_prompt,
        }
        if grammar is not None:
            body["grammar"] = grammar
        if stop is not None:
            body["stop"] = stop

        t0 = time.monotonic()
        data = await self._request("POST", "/completion", json=body)
        elapsed_ms = (time.monotonic() - t0) * 1000

        return CompletionResult(
            content=data.get("content", ""),
            tokens_generated=data.get("tokens_predicted", 0),
            tokens_evaluated=data.get("tokens_evaluated", 0),
            timing_ms=elapsed_ms,
            success=True,
        )

    async def tokenize(self, content: str) -> list[int]:
        data = await self._request("POST", "/tokenize", json={"content": content})
        return data.get("tokens", [])

    async def detokenize(self, tokens: list[int]) -> str:
        data = await self._request("POST", "/detokenize", json={"tokens": tokens})
        return data.get("content", "")

    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None
