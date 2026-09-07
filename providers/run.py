"""Run one provider over PulseBench-Tab, then score it with tlag_scorer.py.

  python -m providers.run run   --provider reducto_r1 [--limit 5] [--concurrency 16] [--retry-errors]
  python -m providers.run score --provider reducto_r1          # -> predictions/reducto_r1/scores.json
  python -m providers.run status

API keys are read from the environment: REDUCTO_API_KEY, LLAMA_CLOUD_API_KEY, EXTEND_API_KEY,
ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY, PULSE_API_KEY.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import time

from .common import DATA_DIR, PRED_DIR, ROOT, ProviderResult, RunStore, materialise_dataset
from .registry import PROVIDERS, load_adapter


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


async def run_provider(key: str, *, limit: int | None, concurrency: int | None, retry_errors: bool) -> None:
    spec = PROVIDERS[key]
    samples = materialise_dataset()[:limit] if limit else materialise_dataset()
    store = RunStore(key)
    todo = [s for s in samples if not (store.done(s.sample_id) or (store.has_raw(s.sample_id) and not retry_errors))]
    conc = concurrency or spec["concurrency"]
    log(f"{key}: {len(samples)} samples, {len(todo)} to run, concurrency={conc}, config={spec['config']}")
    if not todo:
        return
    provider = load_adapter(key)
    await provider.start()
    sem = asyncio.Semaphore(conc)
    counts = {"ok": 0, "no_table": 0, "err": 0}
    t0 = time.time()

    async def one(sample):
        async with sem:
            try:
                res = await provider.run(sample)
            except Exception as e:
                res = ProviderResult({"error": f"{type(e).__name__}: {str(e)[:800]}"}, None, {})
        store.write(sample.sample_id, res)
        counts["err" if not res.ok else "ok" if res.pred_html else "no_table"] += 1
        done = sum(counts.values())
        if done % 25 == 0 or done == len(todo):
            log(f"{key}: {done}/{len(todo)} {counts} elapsed={time.time() - t0:.0f}s")

    try:
        await asyncio.gather(*(one(s) for s in todo))
    finally:
        await provider.close()


def score(key: str) -> None:
    pred_dir = PRED_DIR / key
    out = pred_dir / "scores.json"
    subprocess.run([sys.executable, str(ROOT / "tlag_scorer.py"), "--gt", str(DATA_DIR / "ground_truth"),
                    "--pred", str(pred_dir), "--output", str(out)], check=True)
    log(f"{key}: wrote {out}")


def status() -> None:
    for key in PROVIDERS:
        d = PRED_DIR / key
        if d.exists():
            n_raw = len(list((d / "raw").glob("*.json")))
            n_pred = len(list(d.glob("*.html")))
            print(f"{key:28s} raw={n_raw:4d} pred={n_pred:4d}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("--provider", required=True, choices=sorted(PROVIDERS))
    r.add_argument("--limit", type=int); r.add_argument("--concurrency", type=int); r.add_argument("--retry-errors", action="store_true")
    s = sub.add_parser("score"); s.add_argument("--provider", required=True, choices=sorted(PROVIDERS))
    sub.add_parser("status")
    a = ap.parse_args()
    if a.cmd == "run":
        asyncio.run(run_provider(a.provider, limit=a.limit, concurrency=a.concurrency, retry_errors=a.retry_errors))
    elif a.cmd == "score":
        score(a.provider)
    else:
        status()


if __name__ == "__main__":
    main()
