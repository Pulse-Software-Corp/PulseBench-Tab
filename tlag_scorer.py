#!/usr/bin/env python3
"""
T-LAG (Table Logical Adjacency Graph) scorer for PulseBench-Tab.

Scores predicted HTML tables against ground truth by modeling tables as
2D directed graphs and computing F1 on optimally matched edges.

Usage:
    python tlag_scorer.py --gt ground_truth/ --pred predictions/ [--workers 8]

Each directory should contain HTML files named {sample_id}.html.
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

import numpy as np
from lxml import etree
from Levenshtein import distance as levenshtein_distance
from scipy.optimize import linear_sum_assignment

# -- Configuration --

EXPONENT = 7
_NULL_MARKERS = frozenset([
    "", "-", "\u2013", "\u2014", "...", "\u2026",
    "n/a", "na", "none", "nil", "--", "---",
])


# -- Data structures --

@dataclass
class Edge:
    source: int
    target: int
    direction: str
    source_text: str
    target_text: str


# -- Text normalization and similarity --

def normalize_text(text):
    """Normalize cell text for comparison."""
    if text is None:
        return "[NULL]"
    t = text.strip()
    if t.lower() in _NULL_MARKERS or t.replace(" ", "").replace("\u00a0", "") == "":
        return "[NULL]"
    t = re.sub(r'[\u2012\u2013\u2014\u2015\u2212]', '-', t)
    t = t.replace("\u00a0", " ")
    t = re.sub(r'\s+', ' ', t).strip()
    return t


def psi(text_gt, text_pred):
    """
    Text similarity kernel.

    Psi(a, b) = (1 - d_Lev(a, b) / max(|a|, |b|))^k

    where k = EXPONENT (default 7).
    """
    gt, pred = normalize_text(text_gt), normalize_text(text_pred)
    if gt == "[NULL]" and pred == "[NULL]":
        return 1.0
    if gt == "[NULL]" or pred == "[NULL]":
        return 0.0
    dist = levenshtein_distance(gt, pred)
    max_len = max(len(gt), len(pred))
    if max_len == 0:
        return 1.0
    return (1.0 - dist / max_len) ** EXPONENT


# -- HTML parsing --

def parse_html_to_matrix(html_string):
    """
    Parse an HTML table into a cell-position grid matrix.

    Returns (cells, matrix, num_rows, num_cols) where:
      - cells: dict mapping cell_id to {id, text, rs, cs}
      - matrix: dict mapping (row, col) to cell_id
      - num_rows, num_cols: grid dimensions
    """
    dom = etree.HTML(html_string, parser=etree.HTMLParser())
    cells, cell_counter, matrix, span_info = {}, 0, {}, {}
    row_idx = 0
    all_trs = dom.xpath("//tr")
    total_rows = len(all_trs)

    for tr in all_trs:
        col_idx = 0
        while col_idx in span_info and span_info[col_idx][1] > 0:
            cid, remaining = span_info[col_idx]
            matrix[(row_idx, col_idx)] = cid
            span_info[col_idx] = (cid, remaining - 1)
            if span_info[col_idx][1] == 0:
                del span_info[col_idx]
            col_idx += 1
        for cell_elem in tr.xpath("td|th"):
            while col_idx in span_info and span_info[col_idx][1] > 0:
                cid, remaining = span_info[col_idx]
                matrix[(row_idx, col_idx)] = cid
                span_info[col_idx] = (cid, remaining - 1)
                if span_info[col_idx][1] == 0:
                    del span_info[col_idx]
                col_idx += 1
            try:
                rs = int(cell_elem.get("rowspan", "1").strip().rstrip('>'))
            except ValueError:
                rs = 1
            if rs == 0:
                rs = max(1, total_rows - row_idx)
            try:
                cs = int(cell_elem.get("colspan", "1").strip().rstrip('>'))
            except ValueError:
                cs = 1
            if cs == 0:
                cs = 1
            text = "".join(cell_elem.itertext()).strip()
            cells[cell_counter] = {"id": cell_counter, "text": text, "rs": rs, "cs": cs}
            for r in range(rs):
                for c in range(cs):
                    matrix[(row_idx + r, col_idx + c)] = cell_counter
            if rs > 1:
                for c in range(cs):
                    span_info[col_idx + c] = (cell_counter, rs - 1)
            col_idx += cs
            cell_counter += 1
        while col_idx in span_info and span_info[col_idx][1] > 0:
            cid, remaining = span_info[col_idx]
            matrix[(row_idx, col_idx)] = cid
            span_info[col_idx] = (cid, remaining - 1)
            if span_info[col_idx][1] == 0:
                del span_info[col_idx]
            col_idx += 1
        row_idx += 1

    if not matrix:
        return cells, matrix, 0, 0
    num_rows = max(r for r, c in matrix.keys()) + 1
    num_cols = max(c for r, c in matrix.keys()) + 1
    return cells, matrix, num_rows, num_cols


# -- Edge extraction --

def extract_edges(cells, matrix, num_rows, num_cols):
    """
    Extract directed edges from a cell-position grid matrix.

    RIGHT edge: cell at (r, c) -> cell at (r, c+1), if they are different cells.
    BELOW edge: cell at (r, c) -> cell at (r+1, c), if they are different cells.

    Edges are deduplicated by (source_id, target_id, direction) to prevent
    spanning cells from generating duplicate edges.
    """
    seen, edges = set(), []
    for r in range(num_rows):
        for c in range(num_cols):
            if (r, c) not in matrix:
                continue
            uid = matrix[(r, c)]
            if c + 1 < num_cols and (r, c + 1) in matrix:
                vid = matrix[(r, c + 1)]
                if uid != vid:
                    key = (uid, vid, "RIGHT")
                    if key not in seen:
                        seen.add(key)
                        edges.append(Edge(uid, vid, "RIGHT",
                                          cells[uid]["text"], cells[vid]["text"]))
            if r + 1 < num_rows and (r + 1, c) in matrix:
                vid = matrix[(r + 1, c)]
                if uid != vid:
                    key = (uid, vid, "BELOW")
                    if key not in seen:
                        seen.add(key)
                        edges.append(Edge(uid, vid, "BELOW",
                                          cells[uid]["text"], cells[vid]["text"]))
    return edges


# -- Scoring --

def compute_tlag(gt_edges, pr_edges):
    """
    Compute T-LAG score (F1 on optimally matched edges).

    Uses the Hungarian algorithm for globally optimal one-to-one matching.
    RIGHT edges only match RIGHT; BELOW only matches BELOW.

    Returns (f1, precision, recall).
    """
    n_gt, n_pred = len(gt_edges), len(pr_edges)
    if n_gt == 0 and n_pred == 0:
        return 1.0, 1.0, 1.0
    if n_gt == 0 or n_pred == 0:
        return 0.0, 0.0, 0.0

    weight = np.zeros((n_gt, n_pred), dtype=np.float64)
    for i, eg in enumerate(gt_edges):
        for j, ep in enumerate(pr_edges):
            if eg.direction != ep.direction:
                continue
            s = psi(eg.source_text, ep.source_text)
            t = psi(eg.target_text, ep.target_text)
            weight[i, j] = s * t

    row_ind, col_ind = linear_sum_assignment(-weight)
    matched = sum(weight[r, c] for r, c in zip(row_ind, col_ind))
    precision = matched / n_pred
    recall = matched / n_gt
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return f1, precision, recall


def score_single(gt_html, pred_html):
    """
    Score a single predicted HTML table against ground truth.

    Returns a dict with keys: score, precision, recall, gt_edges, pred_edges.
    """
    gt_cells, gt_mat, gt_r, gt_c = parse_html_to_matrix(gt_html)
    pr_cells, pr_mat, pr_r, pr_c = parse_html_to_matrix(pred_html)

    gt_edges = extract_edges(gt_cells, gt_mat, gt_r, gt_c)
    pr_edges = extract_edges(pr_cells, pr_mat, pr_r, pr_c)

    # Single-cell fallback: 0 edges on both sides
    if len(gt_edges) == 0 and len(pr_edges) == 0:
        gt_text = gt_cells[0]["text"] if gt_cells else ""
        pr_text = pr_cells[0]["text"] if pr_cells else ""
        score = psi(gt_text, pr_text)
        return {
            "score": score,
            "precision": score,
            "recall": score,
            "gt_edges": 0,
            "pred_edges": 0,
        }

    f1, prec, rec = compute_tlag(gt_edges, pr_edges)
    return {
        "score": f1,
        "precision": prec,
        "recall": rec,
        "gt_edges": len(gt_edges),
        "pred_edges": len(pr_edges),
    }


def _score_file(args):
    """Worker function for parallel scoring."""
    sample_id, gt_path, pred_path = args
    try:
        with open(gt_path) as f:
            gt_html = f.read()
        with open(pred_path) as f:
            pred_html = f.read()
        if not pred_html.strip():
            return sample_id, None
        result = score_single(gt_html, pred_html)
        return sample_id, result
    except Exception as e:
        return sample_id, None


# -- CLI --

def main():
    parser = argparse.ArgumentParser(
        description="T-LAG scorer for PulseBench-Tab"
    )
    parser.add_argument("--gt", required=True, help="Directory with ground truth HTML files")
    parser.add_argument("--pred", required=True, help="Directory with predicted HTML files")
    parser.add_argument("--output", default=None, help="Output JSON file (default: stdout)")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers")
    args = parser.parse_args()

    gt_files = {f.replace(".html", "") for f in os.listdir(args.gt) if f.endswith(".html")}
    pred_files = {f.replace(".html", "") for f in os.listdir(args.pred) if f.endswith(".html")}

    common = sorted(gt_files & pred_files)
    missing = sorted(gt_files - pred_files)

    print(f"Ground truth: {len(gt_files)} files", file=sys.stderr)
    print(f"Predictions:  {len(pred_files)} files", file=sys.stderr)
    print(f"Common:       {len(common)} files", file=sys.stderr)
    print(f"Missing:      {len(missing)} files", file=sys.stderr)

    tasks = [
        (sid, os.path.join(args.gt, f"{sid}.html"), os.path.join(args.pred, f"{sid}.html"))
        for sid in common
    ]

    scores = {}
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_score_file, t): t[0] for t in tasks}
        done = 0
        for future in as_completed(futures):
            sid = futures[future]
            sample_id, result = future.result()
            if result is not None:
                scores[sample_id] = result
            done += 1
            if done % 500 == 0:
                print(f"  Scored {done}/{len(tasks)}...", file=sys.stderr)

    # Compute aggregate stats
    valid_scores = [s["score"] for s in scores.values()]
    arr = np.array(valid_scores)

    summary = {
        "n_total": len(gt_files),
        "n_scored": len(valid_scores),
        "n_missing": len(gt_files) - len(valid_scores),
        "coverage_pct": round(len(valid_scores) / len(gt_files) * 100, 1),
        "mean": round(float(np.mean(arr)), 4) if len(arr) > 0 else None,
        "median": round(float(np.median(arr)), 4) if len(arr) > 0 else None,
        "std": round(float(np.std(arr)), 4) if len(arr) > 0 else None,
        "p10": round(float(np.percentile(arr, 10)), 4) if len(arr) > 0 else None,
        "p25": round(float(np.percentile(arr, 25)), 4) if len(arr) > 0 else None,
        "p75": round(float(np.percentile(arr, 75)), 4) if len(arr) > 0 else None,
        "p90": round(float(np.percentile(arr, 90)), 4) if len(arr) > 0 else None,
        "perfect_count": int(np.sum(arr >= 0.9999)) if len(arr) > 0 else 0,
    }

    output = {
        "summary": summary,
        "per_sample": {sid: round(s["score"], 6) for sid, s in scores.items()},
    }

    # Print summary to stderr
    print(f"\nResults:", file=sys.stderr)
    print(f"  T-LAG Score (mean): {summary['mean']}", file=sys.stderr)
    print(f"  Median:             {summary['median']}", file=sys.stderr)
    print(f"  Coverage:           {summary['coverage_pct']}%", file=sys.stderr)
    print(f"  Perfect (=1.0):     {summary['perfect_count']}", file=sys.stderr)

    # Output JSON
    json_str = json.dumps(output, indent=2, ensure_ascii=False)
    if args.output:
        with open(args.output, "w") as f:
            f.write(json_str)
        print(f"\nSaved to {args.output}", file=sys.stderr)
    else:
        print(json_str)


if __name__ == "__main__":
    main()
