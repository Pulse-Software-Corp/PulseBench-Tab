"""Provider configurations used for the September 2026 leaderboard runs.

One entry = one leaderboard row = one run. Settings are the vendor's documented options at the tier/model
named in the row; nothing is tuned against the benchmark. Concurrency is a client-side cap — lower it if your
account's rate limits are tighter.
"""
from __future__ import annotations

import importlib

PROVIDERS: dict[str, dict] = {
    # Reducto (V3 API): settings.model r-1 | legacy; agentic = enhance.agentic scopes on the legacy pipeline.
    "reducto": {"display_name": "Reducto", "adapter": "providers.reducto:ReductoProvider",
                "config": {"model": "legacy", "agentic": False}, "concurrency": 32},
    "reducto_r1": {"display_name": "Reducto r-1", "adapter": "providers.reducto:ReductoProvider",
                   "config": {"model": "r-1", "agentic": False}, "concurrency": 32},
    "reducto_agentic": {"display_name": "Reducto (Agentic)", "adapter": "providers.reducto:ReductoProvider",
                        "config": {"model": "legacy", "agentic": True}, "concurrency": 32},
    # LlamaParse v2 tiers, version pinned to the release used for the run.
    "llamaparse_cost_effective": {"display_name": "LlamaParse (Cost Effective)", "adapter": "providers.llamaparse:LlamaParseProvider",
                                  "config": {"tier": "cost_effective", "version": "2026-08-19"}, "concurrency": 5},
    "llamaparse_agentic": {"display_name": "LlamaParse (Agentic)", "adapter": "providers.llamaparse:LlamaParseProvider",
                           "config": {"tier": "agentic", "version": "2026-08-19"}, "concurrency": 5},
    "llamaparse_agentic_plus": {"display_name": "LlamaParse (Agentic Plus)", "adapter": "providers.llamaparse:LlamaParseProvider",
                                "config": {"tier": "agentic_plus", "version": "2026-08-19"}, "concurrency": 5},
    # Extend: Parse 2.0 = parse_performance@2.0.0; Lite = parse_light@1.0.0.
    "extend_2": {"display_name": "Extend 2.0", "adapter": "providers.extend:ExtendProvider",
                 "config": {"engine": "parse_performance", "engine_version": "2.0.0"}, "concurrency": 6},
    "extend_lite": {"display_name": "Extend (Lite)", "adapter": "providers.extend:ExtendProvider",
                    "config": {"engine": "parse_light", "engine_version": "1.0.0"}, "concurrency": 6},
    # Frontier VLMs, one shared prompt (providers/vlm.py), each vendor's maximum reasoning setting.
    "claude_fable_5_1": {"display_name": "Claude Fable 5.1", "adapter": "providers.vlm:AnthropicProvider",
                         "config": {"model": "claude-fable-5-1", "effort": "max", "max_tokens": 32000}, "concurrency": 16},
    "gpt_6_astra": {"display_name": "GPT-6 Astra", "adapter": "providers.vlm:OpenAIProvider",
                    "config": {"model": "gpt-6-astra", "reasoning_effort": "max", "detail": "original"}, "concurrency": 16},
    "gpt_5_6_sol": {"display_name": "GPT-5.6 Sol", "adapter": "providers.vlm:OpenAIProvider",
                    "config": {"model": "gpt-5.6-sol", "reasoning_effort": "max", "detail": "original"}, "concurrency": 16},
    "gemini_3_8_flash": {"display_name": "Gemini 3.8 Flash", "adapter": "providers.vlm:GeminiProvider",
                         "config": {"model": "gemini-3.8-flash", "thinking_level": "high", "media_resolution": "MEDIA_RESOLUTION_ULTRA_HIGH"}, "concurrency": 16},
}


def load_adapter(key: str):
    spec = PROVIDERS[key]
    mod, cls = spec["adapter"].split(":")
    return getattr(importlib.import_module(mod), cls)(key, spec["config"])
