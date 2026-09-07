"""Shared plumbing for PulseBench-Tab provider runs.

Samples come from the public dataset (https://huggingface.co/datasets/pulse-ai/PulseBench-Tab). Each sample is
one PNG containing exactly one table; the prediction we score is the provider's complete output for that image,
saved verbatim to predictions/<provider>/<sample_id>.html. No post-processing: if a provider returns two tables
for a one-table image, the scorer counts every row it emitted. A response with no table row at all is not saved
and shows up as "missing" (the published scoring mode excludes missing samples).
"""
from __future__ import annotations

import asyncio
import json
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"          # images/<sample_id>.png + ground_truth/<sample_id>.html (materialised from HF)
PRED_DIR = ROOT / "predictions"   # predictions/<provider>/<sample_id>.html (+ raw/<sample_id>.json)

_ROW = re.compile(r"<tr\b", re.I)


@dataclass(frozen=True)
class Sample:
    sample_id: str
    language: str
    image_path: Path
    gt_path: Path


@dataclass
class ProviderResult:
    raw: dict                      # vendor response as returned (JSON-serialisable), or {"error": "..."}
    pred_html: str | None          # complete provider output when it contains a table row, else None
    meta: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.raw.get("error")


class Provider:
    def __init__(self, key: str, config: dict):
        self.key, self.config = key, config

    async def start(self) -> None: ...
    async def close(self) -> None: ...
    async def run(self, sample: Sample) -> ProviderResult:
        raise NotImplementedError


def table_output(text: str | None) -> str | None:
    """The provider's output unchanged when it contains at least one <tr>, otherwise None (=> missing)."""
    return text if text and _ROW.search(text) else None


async def backoff_sleep(attempt: int, base: float = 1.0, cap: float = 30.0) -> None:
    await asyncio.sleep(min(cap, base * (2 ** attempt)) + random.random())


def is_retryable_status(status: int) -> bool:
    return status == 429 or 500 <= status < 600


def materialise_dataset(data_dir: Path = DATA_DIR) -> list[Sample]:
    """Download the HF dataset once and write images + ground truth to disk."""
    manifest = data_dir / "manifest.json"
    if not manifest.exists():
        from datasets import load_dataset

        ds = load_dataset("pulse-ai/PulseBench-Tab", split="train")
        (data_dir / "images").mkdir(parents=True, exist_ok=True)
        (data_dir / "ground_truth").mkdir(parents=True, exist_ok=True)
        rows = {}
        for s in ds:
            sid = s["sample_id"]
            s["image"].save(data_dir / "images" / f"{sid}.png", format="PNG")
            (data_dir / "ground_truth" / f"{sid}.html").write_text(s["ground_truth_html"], encoding="utf-8")
            rows[sid] = s["language"]
        manifest.write_text(json.dumps(rows, indent=0))
    rows = json.loads(manifest.read_text())
    return [Sample(sid, lang, data_dir / "images" / f"{sid}.png", data_dir / "ground_truth" / f"{sid}.html")
            for sid, lang in sorted(rows.items())]


class RunStore:
    """predictions/<provider>/{raw/<id>.json, <id>.html, meta.jsonl}; resumable."""

    def __init__(self, provider_key: str, pred_root: Path = PRED_DIR):
        self.root = pred_root / provider_key
        self.raw = self.root / "raw"
        self.raw.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.root / "meta.jsonl"

    def done(self, sample_id: str) -> bool:
        p = self.raw / f"{sample_id}.json"
        if not p.exists():
            return False
        try:
            return not json.loads(p.read_text()).get("error")
        except Exception:
            return False

    def has_raw(self, sample_id: str) -> bool:
        return (self.raw / f"{sample_id}.json").exists()

    def write(self, sample_id: str, res: ProviderResult) -> None:
        (self.raw / f"{sample_id}.json").write_text(json.dumps(res.raw, ensure_ascii=False))
        pp = self.root / f"{sample_id}.html"
        if res.pred_html:
            pp.write_text(res.pred_html, encoding="utf-8")
        elif pp.exists():
            pp.unlink()
        with self.meta_path.open("a") as fh:
            fh.write(json.dumps({"sample_id": sample_id, "ts": time.time(), "ok": res.ok,
                                 "has_table": bool(res.pred_html), **res.meta}, ensure_ascii=False) + "\n")
