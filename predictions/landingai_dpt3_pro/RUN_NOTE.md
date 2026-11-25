# LandingAI ADE (DPT-3 Pro): run note

| | |
|---|---|
| Run date | 2026-11-24, one pass, 22:21–22:46 UTC |
| API | LandingAI ADE Parse Jobs (`POST /v2/parse/jobs`, poll `GET /v2/parse/jobs/{job_id}`), REST via aiohttp, no SDK. Docs: https://docs.landing.ai/dpt3/parse-async |
| Request settings | `model=dpt-3-pro-20261111` (dated DPT-3 Pro snapshot, 2026-11-11 release), `service_tier=standard`, every other option at its documented default (tables are returned as HTML). One PNG per request, sent unchanged. |
| Prediction | Every `table` block in the response, in document order, sliced verbatim from the response Markdown by the block's `grounding.range`. Same selection as the Reducto, Extend and LlamaParse adapters. Nothing cleaned or edited. |
| Dataset | `pulse-ai/PulseBench-Tab`, revision `e0e1fd5c4b3f7daa0c6afed88177ed9242e6593d`; repo at `b57992c` |
| Retries | 0 API errors, so `--retry-errors` was never needed. No sample was re-run. |
| Result | T-LAG 78.6 (missing excluded), coverage 84.2% (1,532 of 1,820); 66.2 with missing counted as 0. `scores.json` is the scorer's output. |
| Missing samples | 288 responses had no `table` block (the table came back as text, a figure, or other blocks), so no `.html` was written, per the rules. Their complete responses are kept with the raw responses. |
| Raw responses | All 1,820 kept (not committed, since the repo's `.gitignore` excludes `predictions/*/raw/`); available on request. Job identifiers in `meta.jsonl` are replaced with placeholders. |
| Credits used | 1,619.3 (about 0.89 per sample), listed per sample in `meta.jsonl` |
