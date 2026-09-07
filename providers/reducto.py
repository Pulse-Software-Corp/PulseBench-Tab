"""Reducto V3 parse (https://docs.reducto.ai): upload -> /parse_async -> poll /job/{id}.

config:
  model    "legacy" | "r-1"                    (settings.model)
  agentic  bool  -> enhance.agentic=[{scope:table},{scope:text}] (legacy agentic pipeline)
Tables requested as HTML (formatting.table_output_format=html); everything else is the vendor default.
Prediction = every Table block the response contains, in order.
"""
from __future__ import annotations

import os
import time

import aiohttp

from .common import Provider, ProviderResult, Sample, table_output, backoff_sleep, is_retryable_status


BASE = "https://platform.reducto.ai"


class ReductoProvider(Provider):
    async def start(self) -> None:
        key = os.environ["REDUCTO_API_KEY"]
        self._headers = {"Authorization": f"Bearer {key}"}
        self._session = aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0),
                                              timeout=aiohttp.ClientTimeout(total=300, connect=60))
        self.poll_interval = float(self.config.get("poll_interval", 2.0))
        self.job_timeout = float(self.config.get("job_timeout", 900))

    async def close(self) -> None:
        await self._session.close()

    def payload(self, file_id: str) -> dict:
        body = {
            "input": file_id,  # /upload already returns the reducto:// prefix
            "settings": {"model": self.config["model"]},
            "formatting": {"table_output_format": "html"},
        }
        if self.config.get("agentic"):
            body["enhance"] = {"agentic": [{"scope": "table"}, {"scope": "text"}]}
        return body

    async def _req(self, method: str, url: str, *, max_tries: int = 8, **kw):
        for attempt in range(max_tries):
            try:
                async with self._session.request(method, url, headers=self._headers, **kw) as r:
                    if r.status == 200:
                        return 200, await r.json()
                    txt = await r.text()
                    if is_retryable_status(r.status) and attempt < max_tries - 1:
                        await backoff_sleep(attempt)
                        continue
                    return r.status, txt
            except (aiohttp.ClientError, TimeoutError) as e:
                if attempt == max_tries - 1:
                    return -1, f"{type(e).__name__}: {e}"
                await backoff_sleep(attempt)
        return -1, "exhausted"

    async def run(self, sample: Sample) -> ProviderResult:
        t0 = time.time()
        form = aiohttp.FormData()
        form.add_field("file", sample.image_path.read_bytes(), filename=sample.image_path.name, content_type="image/png")
        st, up = await self._req("POST", f"{BASE}/upload", data=form)
        if st != 200:
            return ProviderResult({"error": f"upload {st}: {str(up)[:1500]}"}, None, {"latency_s": time.time() - t0})
        st, sub = await self._req("POST", f"{BASE}/parse_async", json=self.payload(up["file_id"]))
        if st != 200:
            return ProviderResult({"error": f"parse_async {st}: {str(sub)[:1500]}"}, None, {"latency_s": time.time() - t0})
        job_id = sub["job_id"]
        t_submit = time.time()
        while time.time() - t_submit < self.job_timeout:
            await __import__("asyncio").sleep(self.poll_interval)
            st, job = await self._req("GET", f"{BASE}/job/{job_id}")
            if st != 200:
                return ProviderResult({"error": f"poll {st}: {str(job)[:1500]}", "job_id": job_id}, None, {"latency_s": time.time() - t0})
            status = job.get("status")
            if status == "Completed":
                res = job["result"]
                inner = res.get("result", {})
                if inner.get("type") == "url":
                    async with self._session.get(inner["url"]) as rr:
                        res["result"] = await rr.json()
                pred = self._extract(res)
                usage = res.get("usage") or {}
                return ProviderResult(res, pred, {"latency_s": round(time.time() - t0, 2), "job_id": job_id,
                                                  "server_duration": res.get("duration"), "usage": usage})
            if status in ("Failed", "Cancelled"):
                return ProviderResult({"error": f"job {status}: {job.get('reason') or job.get('error')}", "job_id": job_id, "job": job},
                                      None, {"latency_s": time.time() - t0})
        return ProviderResult({"error": f"timeout after {self.job_timeout}s", "job_id": job_id}, None, {"latency_s": time.time() - t0})

    @staticmethod
    def _extract(res: dict) -> str | None:
        """All Table blocks, in document order, exactly as returned."""
        tables = [blk["content"] for chunk in (res.get("result") or {}).get("chunks", [])
                  for blk in chunk.get("blocks", []) if blk.get("type") == "Table" and isinstance(blk.get("content"), str)]
        return table_output("\n".join(tables))
