import asyncio
import contextlib
from contextlib import nullcontext

import aiolimiter
from kani.ai_function import AIFunction
from kani.engines.base import BaseCompletion, WrapperEngine
from kani.models import ChatMessage


class RatelimitedEngine(WrapperEngine):
    def __init__(
        self,
        engine,
        *args,
        max_concurrency: int = None,
        rpm_limit: float = None,
        rpm_period: float = 60,
        tpm_limit: float = None,
        tpm_period: float = 60,
        **kwargs
    ):
        """
        A wrapper engine to enforce request-per-minute (RPM), token-per-minute (TPM), and/or max-concurrency ratelimits
        before making requests to the underlying engine.

        .. code-blocK:: python

            # limit requests to GPT-4 to 10 req/min and 30k tokens/min
            oai_engine = OpenAIEngine(api_key, model="gpt-4")
            engine = RatelimitedEngine(oai_engine, rpm_limit=10, tpm_limit=30_000)

        This engine will pass-through attribute accesses to the wrapped engine.

        :param engine: The engine to wrap.
        :param max_concurrency: The maximum number of concurrent requests to serve at once (default unlimited).
        :param rpm_limit: The maximum number of requests to serve per *rpm_period* (default unlimited).
        :param rpm_period: The duration, in seconds, of the time period in which to limit the rate. Note that up to
            *rpm_limit* requests are allowed within this time period in a burst (default 60s).
        :param tpm_limit: The maximum number of tokens to send in requests per *tpm_period* (default unlimited).
        :param tpm_period: The duration, in seconds, of the time period in which to limit the rate. Note that up to
            *tpm_limit* tokens are allowed within this time period in a burst (default 60s).
        """
        super().__init__(engine, *args, **kwargs)

        if max_concurrency is None:
            self.concurrency_semaphore = nullcontext()
        else:
            self.concurrency_semaphore = asyncio.Semaphore(max_concurrency)

        if rpm_limit is not None:
            self.rpm_limiter = aiolimiter.AsyncLimiter(rpm_limit, rpm_period)
        else:
            self.rpm_limiter = None

        if tpm_limit is not None:
            self.tpm_limiter = aiolimiter.AsyncLimiter(tpm_limit, tpm_period)
        else:
            self.tpm_limiter = None

    @contextlib.asynccontextmanager
    async def _ratelimit_ctx(self, messages: list[ChatMessage], functions: list[AIFunction] | None):
        if self.rpm_limiter:
            await self.rpm_limiter.acquire()
        if self.tpm_limiter:
            n_toks = self.function_token_reserve(functions) + sum(self.message_len(m) for m in messages)
            await self.tpm_limiter.acquire(n_toks)
        async with self.concurrency_semaphore:
            yield

    async def predict(
        self, messages: list[ChatMessage], functions: list[AIFunction] | None = None, **hyperparams
    ) -> BaseCompletion:
        async with self._ratelimit_ctx(messages, functions):
            return await super().predict(messages, functions, **hyperparams)

    async def stream(self, messages: list[ChatMessage], functions: list[AIFunction] | None = None, **hyperparams):
        async with self._ratelimit_ctx(messages, functions):
            async for elem in super().stream(messages, functions, **hyperparams):
                yield elem
