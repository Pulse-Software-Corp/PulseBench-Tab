"""Frontier VLMs on PulseBench-Tab: one shared prompt, one image per request, HTML table out.


Settings are the vendors' highest-quality knobs verified live against each API on 2026-09-06:
  Anthropic  claude-fable-5-1   thinking={"type":"adaptive"} + output_config={"effort":"max"} (thinking.enabled is rejected)
  OpenAI     gpt-6-astra / gpt-5.6-sol   Responses API, reasoning.effort="max", input_image detail="original" (OCR guidance)
  Google     gemini-3.8-flash   thinking_level="high" (ULTRA/MAX rejected), per-part media_resolution ULTRA_HIGH (2240 tok) -> HIGH fallback
"""
from __future__ import annotations

import asyncio
import base64
import os
import time

from .common import Provider, ProviderResult, Sample, table_output, backoff_sleep


PROMPT = (
    "Convert this table image into an HTML table.\n"
    "- Use <table>, <thead>, <tbody>, <tr>, <th>, <td>.\n"
    "- Use rowspan and colspan where cells span multiple rows or columns.\n"
    "- Preserve every row, column, header, and value exactly as shown. Do not omit, reorder, or summarize.\n"
    "- Do not add commentary and do not wrap the output in code fences. Output only the HTML table."
)

MAX_TRIES = 6


def _b64(sample: Sample) -> str:
    return base64.b64encode(sample.image_path.read_bytes()).decode()


class AnthropicProvider(Provider):
    async def start(self) -> None:
        import anthropic
        self._client = anthropic.AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"], max_retries=0, timeout=600)
        self.model = self.config["model"]
        self.effort = self.config.get("effort", "max")
        self.max_tokens = int(self.config.get("max_tokens", 32000))

    async def run(self, sample: Sample) -> ProviderResult:
        import anthropic
        t0 = time.time()
        for attempt in range(MAX_TRIES):
            try:
                async with self._client.messages.stream(
                    model=self.model, max_tokens=self.max_tokens,
                    thinking={"type": "adaptive"}, output_config={"effort": self.effort},
                    messages=[{"role": "user", "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": _b64(sample)}},
                        {"type": "text", "text": PROMPT}]}],
                ) as stream:
                    msg = await stream.get_final_message()
                raw = msg.model_dump(mode="json")
                text = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
                raw["content"] = [b if b.get("type") != "thinking" else {"type": "thinking", "chars": len(b.get("thinking") or "")} for b in raw["content"]]
                raw["_text"] = text
                meta = {"latency_s": round(time.time() - t0, 2), "usage": raw.get("usage"), "stop_reason": msg.stop_reason}
                if msg.stop_reason in ("refusal", "max_tokens"):
                    raw["error"] = f"stop_reason={msg.stop_reason} {raw.get('stop_details')}"
                    return ProviderResult(raw, None, meta)
                return ProviderResult(raw, table_output(text), meta)
            except (anthropic.RateLimitError, anthropic.APIConnectionError, anthropic.InternalServerError,
                    anthropic.APITimeoutError, anthropic.OverloadedError, anthropic.ServiceUnavailableError) as e:
                if attempt == MAX_TRIES - 1:
                    return ProviderResult({"error": f"{type(e).__name__}: {str(e)[:800]}"}, None, {"latency_s": time.time() - t0})
                await backoff_sleep(attempt, base=2.0, cap=60.0)
            except anthropic.APIStatusError as e:
                return ProviderResult({"error": f"{type(e).__name__} {e.status_code}: {str(e)[:800]}"}, None, {"latency_s": time.time() - t0})
        return ProviderResult({"error": "exhausted"}, None, {"latency_s": time.time() - t0})


class OpenAIProvider(Provider):
    async def start(self) -> None:
        import openai
        self._client = openai.AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"], max_retries=0, timeout=900)
        self.model = self.config["model"]
        self.effort = self.config.get("reasoning_effort", "max")
        self.detail = self.config.get("detail", "original")  # OpenAI's OCR guidance: original keeps full resolution
        self.max_output_tokens = int(self.config.get("max_output_tokens", 32000))

    async def run(self, sample: Sample) -> ProviderResult:
        import openai
        t0 = time.time()
        for attempt in range(MAX_TRIES):
            try:
                resp = await self._client.responses.create(
                    model=self.model, reasoning={"effort": self.effort}, max_output_tokens=self.max_output_tokens,
                    input=[{"role": "user", "content": [
                        {"type": "input_image", "image_url": f"data:image/png;base64,{_b64(sample)}", "detail": self.detail},
                        {"type": "input_text", "text": PROMPT}]}],
                )
                raw = resp.model_dump(mode="json")
                text = resp.output_text
                raw["_text"] = text
                if resp.status != "completed":
                    raw["error"] = f"status={resp.status} {raw.get('incomplete_details')}"
                    return ProviderResult(raw, None, {"latency_s": time.time() - t0, "usage": raw.get("usage")})
                return ProviderResult(raw, table_output(text), {"latency_s": round(time.time() - t0, 2), "usage": raw.get("usage")})
            except (openai.RateLimitError, openai.APIConnectionError, openai.InternalServerError, openai.APITimeoutError) as e:
                if attempt == MAX_TRIES - 1:
                    return ProviderResult({"error": f"{type(e).__name__}: {str(e)[:800]}"}, None, {"latency_s": time.time() - t0})
                await backoff_sleep(attempt, base=2.0, cap=60.0)
            except openai.APIStatusError as e:
                return ProviderResult({"error": f"{type(e).__name__} {e.status_code}: {str(e)[:800]}"}, None, {"latency_s": time.time() - t0})
        return ProviderResult({"error": "exhausted"}, None, {"latency_s": time.time() - t0})


class GeminiProvider(Provider):
    async def start(self) -> None:
        from google import genai
        self._client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        self.model = self.config["model"]
        self.thinking_level = self.config.get("thinking_level", "high")
        self.max_output_tokens = int(self.config.get("max_output_tokens", 65536))

    async def run(self, sample: Sample) -> ProviderResult:
        from google.genai import types, errors
        t0 = time.time()
        cfg = types.GenerateContentConfig(
            max_output_tokens=self.max_output_tokens,
            thinking_config=types.ThinkingConfig(thinking_level=self.thinking_level),
            safety_settings=[types.SafetySetting(category=c, threshold="BLOCK_NONE") for c in (
                "HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT")],
        )
        res_level = self.config.get("media_resolution", "MEDIA_RESOLUTION_ULTRA_HIGH")
        for attempt in range(MAX_TRIES):
            try:
                resp = await self._client.aio.models.generate_content(
                    model=self.model,
                    contents=[types.Part.from_bytes(data=sample.image_path.read_bytes(), mime_type="image/png", media_resolution=res_level),
                              types.Part.from_text(text=PROMPT)],
                    config=cfg)
                raw = resp.model_dump(mode="json", exclude_none=True)
                text = resp.text or ""
                raw["_text"] = text
                fr = resp.candidates[0].finish_reason if resp.candidates else None
                meta = {"latency_s": round(time.time() - t0, 2), "usage": raw.get("usage_metadata"), "finish_reason": str(fr), "media_resolution": res_level}
                if not text:
                    raw["error"] = f"empty response finish_reason={fr}"
                    return ProviderResult(raw, None, meta)
                return ProviderResult(raw, table_output(text), meta)
            except errors.APIError as e:
                if e.code == 400 and "media_resolution" in str(e) and res_level != "MEDIA_RESOLUTION_HIGH":
                    res_level = "MEDIA_RESOLUTION_HIGH"  # ULTRA_HIGH not accepted for this model -> documented max fallback
                    continue
                if e.code in (429, 500, 502, 503, 504) and attempt < MAX_TRIES - 1:
                    await backoff_sleep(attempt, base=2.0, cap=60.0)
                    continue
                return ProviderResult({"error": f"APIError {e.code}: {str(e)[:800]}"}, None, {"latency_s": time.time() - t0})
            except Exception as e:  # transport errors inside the SDK
                if attempt == MAX_TRIES - 1:
                    return ProviderResult({"error": f"{type(e).__name__}: {str(e)[:800]}"}, None, {"latency_s": time.time() - t0})
                await backoff_sleep(attempt, base=2.0, cap=60.0)
        return ProviderResult({"error": "exhausted"}, None, {"latency_s": time.time() - t0})
