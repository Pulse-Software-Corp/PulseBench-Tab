"""Extend parse (https://docs.extend.ai, API version 2026-02-09 via extend-ai SDK 1.19):
files.upload -> parse_runs.create -> poll parse_runs.retrieve.

config:
  engine          parse_performance | parse_light
  engine_version  pinned ("2.0.0" for Parse 2.0, "1.0.0" for Light)
Tables requested as HTML blocks; prediction = every table block, in order. Everything else vendor default.
"""
from __future__ import annotations

import asyncio
import os
import time

from .common import Provider, ProviderResult, Sample, table_output, backoff_sleep



class ExtendProvider(Provider):
    async def start(self) -> None:
        from extend_ai import AsyncExtend
        self._client = AsyncExtend(token=os.environ["EXTEND_API_KEY"], timeout=120.0)
        self.poll_interval = float(self.config.get("poll_interval", 2.0))
        self.job_timeout = float(self.config.get("job_timeout", 900))

    def parse_config(self) -> dict:
        cfg = {
            "target": "markdown",
            "engine": self.config["engine"],
            "block_options": {"tables": {"target_format": "html"}},
        }
        if self.config.get("engine_version"):
            cfg["engine_version"] = self.config["engine_version"]
        return cfg

    async def _call(self, fn, *a, max_tries: int = 8, **kw):
        from extend_ai.core.api_error import ApiError
        for attempt in range(max_tries):
            try:
                return await fn(*a, **kw)
            except ApiError as e:
                sc = getattr(e, "status_code", None)
                if sc in (408, 429, 500, 502, 503, 504) and attempt < max_tries - 1:
                    ra = None
                    try:
                        ra = float((getattr(e, "headers", None) or {}).get("retry-after"))
                    except Exception:
                        pass
                    await (asyncio.sleep(ra) if ra else backoff_sleep(attempt, base=2.0))
                    continue
                raise
            except Exception as e:  # transport
                if attempt == max_tries - 1:
                    raise
                await backoff_sleep(attempt, base=2.0)

    async def run(self, sample: Sample) -> ProviderResult:
        t0 = time.time()
        try:
            f = await self._call(self._client.files.upload, file=(sample.image_path.name, sample.image_path.read_bytes(), "image/png"),
                                 request_options={"max_retries": 0})
            run = await self._call(self._client.parse_runs.create, file={"id": f.id}, config=self.parse_config(),
                                   request_options={"max_retries": 0})
        except Exception as e:
            return ProviderResult({"error": f"{type(e).__name__}: {str(e)[:1500]}"}, None, {"latency_s": time.time() - t0})
        t_submit = time.time()
        while time.time() - t_submit < self.job_timeout:
            await asyncio.sleep(self.poll_interval)
            try:
                run = await self._call(self._client.parse_runs.retrieve, run.id, request_options={"max_retries": 0})
            except Exception as e:
                return ProviderResult({"error": f"poll {type(e).__name__}: {str(e)[:1500]}", "run_id": run.id}, None, {"latency_s": time.time() - t0})
            status = str(run.status)
            if status == "PROCESSED":
                raw = run.model_dump(mode="json", by_alias=True)
                pred = self._extract(raw)
                usage = raw.get("usage") or {}
                return ProviderResult(raw, pred, {"latency_s": round(time.time() - t0, 2), "run_id": run.id,
                                                  "credits": usage.get("totalCredits", usage.get("credits")), "metrics": raw.get("metrics")})
            if status in ("FAILED", "CANCELLED"):
                raw = run.model_dump(mode="json", by_alias=True)
                raw["error"] = f"run {status}: {raw.get('failureReason')} {raw.get('failureMessage')}"
                return ProviderResult(raw, None, {"latency_s": time.time() - t0, "run_id": run.id})
        return ProviderResult({"error": f"timeout after {self.job_timeout}s", "run_id": run.id}, None, {"latency_s": time.time() - t0})

    @staticmethod
    def _extract(raw: dict) -> str | None:
        """All `table` blocks, in document order, exactly as returned."""
        tables = [blk["content"] for chunk in (raw.get("output") or {}).get("chunks") or []
                  for blk in chunk.get("blocks") or [] if blk.get("type") == "table" and isinstance(blk.get("content"), str)]
        return table_output("\n".join(tables))
