"""LlamaParse v2 (https://developers.llamaindex.ai/llamaparse/parse/guides/api-reference/):
POST /api/v1/beta/files (purpose=parse) -> POST /api/v2/parse {file_id, tier, version} -> GET /api/v2/parse/{id}?expand=...

config:
  tier      fast | cost_effective | agentic | agentic_plus
  version   pinned release date (default 2026-08-19 = `latest` for all LLM tiers on 2026-09-06)
Tables are read from items.pages[].items[] where type == "table" (`html` is always present). disable_cache=true so reruns are billed and fresh.
"""
from __future__ import annotations

import asyncio
import os
import time

import httpx

from .common import Provider, ProviderResult, Sample, table_output, backoff_sleep, is_retryable_status


BASE = "https://api.cloud.llamaindex.ai"
EXPAND = ["items", "markdown", "usage", "metadata", "job_metadata"]


class LlamaParseProvider(Provider):
    async def start(self) -> None:
        key = os.environ["LLAMA_CLOUD_API_KEY"]
        self._client = httpx.AsyncClient(base_url=BASE, headers={"Authorization": f"Bearer {key}"},
                                         timeout=httpx.Timeout(300, connect=60), limits=httpx.Limits(max_connections=None))
        self.version = self.config.get("version", "2026-08-19")
        self.poll_interval = float(self.config.get("poll_interval", 2.0))
        self.job_timeout = float(self.config.get("job_timeout", 1200))

    async def close(self) -> None:
        await self._client.aclose()

    def body(self, file_id: str) -> dict:
        return {
            "file_id": file_id,
            "tier": self.config["tier"],
            "version": self.version,
            "disable_cache": True,  # 48h vendor cache would otherwise serve (and not bill) a repeat of the same file
        }

    async def _req(self, method: str, path: str, *, max_tries: int = 8, **kw):
        for attempt in range(max_tries):
            try:
                r = await self._client.request(method, path, **kw)
                if 200 <= r.status_code < 300:  # /api/v1/beta/files answers 201
                    return 200, r.json()
                if is_retryable_status(r.status_code) and attempt < max_tries - 1:
                    await backoff_sleep(attempt, base=2.0)
                    continue
                return r.status_code, r.text
            except (httpx.HTTPError, TimeoutError) as e:
                if attempt == max_tries - 1:
                    return -1, f"{type(e).__name__}: {e}"
                await backoff_sleep(attempt, base=2.0)
        return -1, "exhausted"

    async def run(self, sample: Sample) -> ProviderResult:
        t0 = time.time()
        st, up = await self._req("POST", "/api/v1/beta/files", data={"purpose": "parse"},
                                 files={"file": (sample.image_path.name, sample.image_path.read_bytes(), "image/png")})
        if st != 200:
            return ProviderResult({"error": f"upload {st}: {str(up)[:1500]}"}, None, {"latency_s": time.time() - t0})
        st, job = await self._req("POST", "/api/v2/parse", json=self.body(up["id"]))
        if st != 200:
            return ProviderResult({"error": f"parse {st}: {str(job)[:1500]}"}, None, {"latency_s": time.time() - t0})
        job_id = job["id"]
        t_submit = time.time()
        params = [("expand", e) for e in EXPAND]
        while time.time() - t_submit < self.job_timeout:
            await asyncio.sleep(self.poll_interval)
            st, res = await self._req("GET", f"/api/v2/parse/{job_id}", params=params)
            if st != 200:
                return ProviderResult({"error": f"poll {st}: {str(res)[:1500]}", "job_id": job_id}, None, {"latency_s": time.time() - t0})
            status = (res.get("job") or {}).get("status") or res.get("status")
            if status == "COMPLETED":
                pred = self._extract(res)
                usage = res.get("usage") or {}
                return ProviderResult(res, pred, {"latency_s": round(time.time() - t0, 2), "job_id": job_id,
                                                  "credits": usage.get("credits"), "version": (res.get("metadata") or {}).get("version")})
            if status in ("FAILED", "CANCELLED"):
                return ProviderResult({"error": f"job {status}: {(res.get('job') or {}).get('error_message')}", "job_id": job_id, "job": res.get("job")},
                                      None, {"latency_s": time.time() - t0})
        return ProviderResult({"error": f"timeout after {self.job_timeout}s", "job_id": job_id}, None, {"latency_s": time.time() - t0})

    @staticmethod
    def _extract(res: dict) -> str | None:
        """All table items (their `html`), in page order, exactly as returned."""
        tables = [it["html"] for page in ((res.get("items") or {}).get("pages") or [])
                  for it in page.get("items") or [] if it.get("type") == "table" and it.get("html")]
        return table_output("\n".join(tables))
