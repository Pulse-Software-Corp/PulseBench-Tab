"""Pulse (https://docs.runpulse.com): POST /extract (multipart, model=pulse-ultra-2, async=true) -> poll GET /job/{id}.

config:
  refine          bool            /extract `refine`
  refine_options  dict | None     /extract `refine_options` ({"tables","text","formatting"} booleans)
Prediction = the complete `markdown` of the response (Pulse renders tables as HTML inside
markdown); when markdown carries no table row the `extensions.altOutputs.html` representation is used instead.
"""
from __future__ import annotations

import asyncio
import json
import os
import time

import aiohttp

from .common import Provider, ProviderResult, Sample, table_output, backoff_sleep, is_retryable_status


BASE = os.environ.get("PULSE_API_BASE_URL", "https://api.runpulse.com")


class PulseProvider(Provider):
    async def start(self) -> None:
        self._headers = {"x-api-key": os.environ["PULSE_API_KEY"]}
        self._session = aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0),
                                              timeout=aiohttp.ClientTimeout(total=300, connect=60))
        self.poll_interval = float(self.config.get("poll_interval", 3.0))
        self.job_timeout = float(self.config.get("job_timeout", 1200))

    async def close(self) -> None:
        await self._session.close()

    def form(self, sample: Sample) -> aiohttp.FormData:
        f = aiohttp.FormData()
        f.add_field("file", sample.image_path.read_bytes(), filename=sample.image_path.name, content_type="image/png")
        f.add_field("model", "pulse-ultra-2")
        f.add_field("async", "true")
        f.add_field("refine", "true" if self.config.get("refine") else "false")
        if self.config.get("refine_options") is not None:
            f.add_field("refine_options", json.dumps(self.config["refine_options"]))
        f.add_field("extensions", json.dumps({"altOutputs": {"returnHtml": True}}))
        return f

    async def _req(self, method: str, url: str, *, max_tries: int = 8, **kw):
        for attempt in range(max_tries):
            try:
                async with self._session.request(method, url, headers=self._headers, **kw) as r:
                    if r.status == 200:
                        return 200, await r.json(content_type=None)
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
        return await self._run_extract(sample, t0)

    async def _run_extract(self, sample: Sample, t0: float) -> ProviderResult:
        st, sub = await self._req("POST", f"{BASE}/extract", data=self.form(sample))
        if st != 200:
            return ProviderResult({"error": f"extract {st}: {str(sub)[:1500]}"}, None, {"latency_s": time.time() - t0})
        job_id = sub.get("job_id")
        if not job_id:  # synchronous response
            return self._finish(sub, t0, None)
        t_submit = time.time()
        while time.time() - t_submit < self.job_timeout:
            await asyncio.sleep(self.poll_interval)
            st, job = await self._req("GET", f"{BASE}/job/{job_id}")
            if st != 200:
                return ProviderResult({"error": f"poll {st}: {str(job)[:1500]}", "job_id": job_id}, None, {"latency_s": time.time() - t0})
            status = job.get("status")
            if status == "completed":
                res = job.get("result") if isinstance(job.get("result"), dict) else job
                if res.get("is_url") and res.get("url"):
                    async with self._session.get(res["url"]) as rr:
                        full = await rr.json(content_type=None)
                    if "plan_info" in res:
                        full["plan_info"] = res["plan_info"]
                    res = full
                return self._finish(res, t0, job_id)
            if status in ("failed", "canceled", "cancelled", "expired"):
                return ProviderResult({"error": f"job {status}: {str(job.get('error') or job.get('message'))[:800]}", "job_id": job_id, "job": job},
                                      None, {"latency_s": time.time() - t0})
        return ProviderResult({"error": f"timeout after {self.job_timeout}s", "job_id": job_id}, None, {"latency_s": time.time() - t0})

    def _finish(self, res: dict, t0: float, job_id) -> ProviderResult:
        pred = self._extract(res)
        pi = res.get("plan_info") or {}
        return ProviderResult(res, pred, {"latency_s": round(time.time() - t0, 2), "job_id": job_id,
                                          "pages_used": pi.get("pages_used"), "tier": pi.get("tier"),
                                          "used_html_alt": bool(pred) and not table_output(res.get("markdown"))})

    @staticmethod
    def _extract(res: dict) -> str | None:
        md = table_output(res.get("markdown"))
        if md:
            return md
        ext = res.get("extensions") or {}
        alt = ext.get("altOutputs") or ext.get("alt_outputs") or {}
        return table_output(alt.get("html")) or table_output(res.get("html"))
