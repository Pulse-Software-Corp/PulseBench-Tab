"""LandingAI ADE Parse (DPT-3, https://docs.landing.ai/dpt3/parse-async):
POST /v2/parse/jobs (multipart) -> poll GET /v2/parse/jobs/{job_id}.

config:
  model         dated DPT-3 snapshot, e.g. "dpt-3-pro-20261111" (pinned so the row is reproducible)
  service_tier  "standard" | "priority"   (same model and output; priority is the faster, full-rate lane)
  prediction    "tables"   -> every `table` block, in document order, sliced verbatim from the response
                              Markdown by the block's grounding.range (same selection as the Reducto,
                              Extend and LlamaParse adapters)
                "markdown" -> the complete Markdown response
All parse options are the documented defaults (tables are HTML by default).
"""
from __future__ import annotations

import asyncio
import json
import os
import time

import aiohttp

from .common import Provider, ProviderResult, Sample, table_output, backoff_sleep, is_retryable_status


BASE = "https://api.ade.landing.ai"


class LandingAIProvider(Provider):
    async def start(self) -> None:
        key = os.environ["VISION_AGENT_API_KEY"]
        self._headers = {"Authorization": f"Bearer {key}"}
        self._session = aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0),
                                              timeout=aiohttp.ClientTimeout(total=300, connect=60))
        self.poll_interval = float(self.config.get("poll_interval", 3.0))
        self.job_timeout = float(self.config.get("job_timeout", 3600))

    async def close(self) -> None:
        await self._session.close()

    async def _req(self, method: str, url: str, *, ok=(200, 202), max_tries: int = 8, form_factory=None, **kw):
        for attempt in range(max_tries):
            try:
                if form_factory is not None:
                    kw["data"] = form_factory()
                async with self._session.request(method, url, headers=self._headers, **kw) as r:
                    if r.status in ok:
                        return r.status, await r.json()
                    txt = await r.text()
                    if is_retryable_status(r.status) and attempt < max_tries - 1:
                        await backoff_sleep(attempt, base=2.0, cap=60.0)
                        continue
                    return r.status, txt
            except (aiohttp.ClientError, TimeoutError) as e:
                if attempt == max_tries - 1:
                    return -1, f"{type(e).__name__}: {e}"
                await backoff_sleep(attempt, base=2.0, cap=60.0)
        return -1, "exhausted"

    def _form(self, sample: Sample) -> aiohttp.FormData:
        form = aiohttp.FormData()
        form.add_field("document", sample.image_path.read_bytes(), filename=sample.image_path.name, content_type="image/png")
        form.add_field("model", self.config["model"])
        form.add_field("service_tier", self.config["service_tier"])
        return form

    async def run(self, sample: Sample) -> ProviderResult:
        t0 = time.time()
        st, sub = await self._req("POST", f"{BASE}/v2/parse/jobs", form_factory=lambda: self._form(sample))
        if st not in (200, 202):
            return ProviderResult({"error": f"submit {st}: {str(sub)[:1500]}"}, None, {"latency_s": round(time.time() - t0, 2)})
        job_id = sub["job_id"]
        t_submit = time.time()
        while time.time() - t_submit < self.job_timeout:
            await asyncio.sleep(self.poll_interval)
            st, job = await self._req("GET", f"{BASE}/v2/parse/jobs/{job_id}", ok=(200,))
            if st != 200:
                return ProviderResult({"error": f"poll {st}: {str(job)[:1500]}", "job_id": job_id}, None,
                                      {"latency_s": round(time.time() - t0, 2)})
            status = job.get("status")
            if status == "completed":
                res = job.get("result") if job.get("result") is not None else job.get("data")
                if res is None and job.get("output_url"):
                    async with self._session.get(job["output_url"]) as rr:
                        res = await rr.json()
                if not isinstance(res, dict):
                    return ProviderResult({"error": "completed without a result", "job_id": job_id, "job": job}, None,
                                          {"latency_s": round(time.time() - t0, 2)})
                meta = res.get("metadata") or {}
                return ProviderResult(res, self._prediction(res), {
                    "latency_s": round(time.time() - t0, 2), "job_id": job_id,
                    "model_version": meta.get("model_version"), "duration_ms": meta.get("duration_ms"),
                    "output_markdown_chars": meta.get("output_markdown_chars"),
                    "credits": (meta.get("billing") or {}).get("total_credits"),
                    "service_tier": (meta.get("billing") or {}).get("service_tier"),
                    "failed_pages": meta.get("failed_pages")})
            if status == "failed":
                return ProviderResult({"error": f"job failed: {json.dumps(job.get('error'))[:1500]}", "job_id": job_id}, None,
                                      {"latency_s": round(time.time() - t0, 2)})
        return ProviderResult({"error": f"timeout after {self.job_timeout}s", "job_id": job_id}, None,
                              {"latency_s": round(time.time() - t0, 2)})

    def _prediction(self, res: dict) -> str | None:
        md = res.get("markdown") or ""
        if self.config.get("prediction", "tables") == "markdown":
            return table_output(md)
        tables = []

        def walk(node: dict) -> None:
            for child in node.get("children") or []:
                if child.get("type") == "table":
                    rng = (child.get("grounding") or {}).get("range") or {}
                    if isinstance(rng.get("start"), int) and isinstance(rng.get("end"), int):
                        tables.append(md[rng["start"]:rng["end"]])
                else:
                    walk(child)

        walk(res.get("structure") or {})
        return table_output("\n".join(tables))
