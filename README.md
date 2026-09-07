# PulseBench-Tab

**A frontier multilingual benchmark for table extraction from document images.**

PulseBench-Tab evaluates how accurately document parsing systems reproduce both the structure (rows, columns, spans) and content (cell text) of tables. The benchmark contains 1,820 human-annotated tables across 9 languages, scored using T-LAG, a novel graph-based evaluation metric.

- **Dataset (HuggingFace):** `https://huggingface.co/datasets/pulse-ai/PulseBench-Tab`
- **Research paper (arXiv):** `https://benchmark.runpulse.com/research-report`
- **Blog post:** `https://runpulse.com/blog/pulsebench-tab`

## T-LAG Scoring Methodology

T-LAG (Table Logical Adjacency Graph) models tables as 2D directed graphs and evaluates structural and content fidelity in a single unified score.

### Step 1: Parse HTML to grid matrix

Both ground truth and predicted HTML tables are parsed into a cell-position grid matrix, preserving all rowspan and colspan attributes. Spanning cells occupy multiple grid positions sharing the same cell ID.

### Step 2: Extract directed edges

For each pair of adjacent grid positions occupied by different cells, a directed edge is created:

- **RIGHT edge:** cell at $(r, c) \rightarrow$ cell at $(r, c{+}1)$
- **BELOW edge:** cell at $(r, c) \rightarrow$ cell at $(r{+}1, c)$

Edges are deduplicated by $(source\_id, target\_id, direction)$ to prevent spanning cells from generating duplicate edges.

### Step 3: Compute edge weights via $\Psi$

For each candidate pairing of a ground truth edge $e_{gt}$ and predicted edge $e_{pred}$ sharing the same direction, the similarity weight is:

$$w(e_{gt}, e_{pred}) = \Psi(\text{source}_{gt}, \text{source}_{pred}) \times \Psi(\text{target}_{gt}, \text{target}_{pred})$$

The $\Psi$ function (text similarity kernel):

1. Normalize both texts: strip whitespace, normalize dashes, collapse internal whitespace
2. Map null markers (empty string, `-`, `--`, `---`, `...`, `n/a`, `na`, `none`, `nil`) to a sentinel `[NULL]`
3. Matching rules:
   - `[NULL]` vs `[NULL]` = 1.0
   - `[NULL]` vs non-null = 0.0
   - Otherwise:

$$\Psi(a, b) = \left(1 - \frac{d_{\text{Lev}}(a, b)}{\max(|a|, |b|)}\right)^{7}$$

where $d_{\text{Lev}}(a, b)$ is the Levenshtein edit distance. The exponent $k = 7$ sharply distinguishes near-perfect from approximate extraction.

### Step 4: Hungarian matching

Ground truth and predicted edge sets are matched using the Hungarian algorithm (`scipy.optimize.linear_sum_assignment`) for globally optimal one-to-one assignment. The weight matrix enforces direction constraints: RIGHT edges can only match RIGHT, BELOW can only match BELOW.

### Step 5: Score

Let $S$ denote the total matched weight from the Hungarian assignment:

$$\text{Precision} = \frac{S}{|\mathcal{E}_{pred}|}$$

$$\text{Recall} = \frac{S}{|\mathcal{E}_{gt}|}$$

$$\text{T-LAG} = F_1 = \frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

**Single-cell fallback:** When both tables produce zero edges (single-cell tables), the score falls back to $\Psi$ on the cell text directly.

### Key design properties

| Property | Description |
|----------|-------------|
| Pure F1 | No separate structural penalty; structure is captured through unmatched edges |
| No edge cap | All edges are scored regardless of table size |
| Direction-constrained | RIGHT matches RIGHT only, BELOW matches BELOW only |
| Spanning deduplication | Prevents large merged cells from dominating the score |
| Case-sensitive | Exact text comparison, no numeric normalization |

For the full mathematical specification, see the research paper.

## Evaluated Providers

We evaluated providers across document AI vendors, foundation models, and open-source tools on all 1,820 samples. Rows dated September 2026 were re-run with the scripts in `providers/`; the remaining rows are from the April 2026 run.

| Rank | Provider | T-LAG Score | Coverage |
|------|----------|------------|----------|
| 1 | **Pulse Ultra 2** | **0.9347** | 100.0% |
| 2 | Claude Fable 5.1 | 0.8904 | 99.3% |
| 3 | Gemini 3.8 Flash | 0.8857 | 99.9% |
| 4 | GPT-6 Astra | 0.8852 | 99.6% |
| 5 | GPT-5.6 Sol | 0.8808 | 99.0% |
| 6 | Mistral OCR 4.1 | 0.8581 | 95.5% |
| 7 | LlamaParse (Agentic) | 0.8396 | 96.4% |
| 8 | Extend 2.0 | 0.8244 | 93.1% |
| 9 | LlamaParse (Agentic Plus) | 0.8207 | 98.3% |
| 10 | Reducto r-1 | 0.8139 | 94.6% |
| 11 | Reducto (Agentic) | 0.8133 | 94.8% |
| 12 | LlamaParse (Cost Effective) | 0.7789 | 96.0% |
| 13 | Datalab | 0.7774 | 91.2% |
| 14 | Azure Document Intelligence | 0.7614 | 92.0% |
| 15 | Extend (Lite) | 0.7547 | 92.7% |
| 16 | Reducto | 0.7115 | 94.8% |
| 17 | AWS Textract | 0.6034 | 98.5% |
| 18 | Unstructured | 0.3603 | 100.0% |

Scoring mode: exclude-missing. Providers are scored only on samples where they produced output.

## Usage

### Score your own predictions

```bash
pip install -r requirements.txt

python tlag_scorer.py \
    --gt path/to/ground_truth/ \
    --pred path/to/predictions/ \
    --output scores.json
```

Both `--gt` and `--pred` directories should contain HTML files named `{sample_id}.html`.

### Score against PulseBench-Tab

```python
from datasets import load_dataset

# Download ground truth
ds = load_dataset("pulse-ai/PulseBench-Tab")

# Save GT files to disk
import os
os.makedirs("ground_truth", exist_ok=True)
for sample in ds["train"]:
    with open(f"ground_truth/{sample['sample_id']}.html", "w") as f:
        f.write(sample["ground_truth_html"])

# Run your model, save predictions to predictions/ as {sample_id}.html
# Then score:
# python tlag_scorer.py --gt ground_truth/ --pred predictions/ --output my_scores.json
```

### Use as a library (from within the cloned repo)

```python
from tlag_scorer import score_single

gt_html = "<table><tr><td>A</td><td>B</td></tr></table>"
pred_html = "<table><tr><td>A</td><td>B</td></tr></table>"

result = score_single(gt_html, pred_html)
print(result["score"])      # 1.0
print(result["precision"])  # 1.0
print(result["recall"])     # 1.0
```

### Run a provider yourself

`providers/` holds the adapter scripts behind the leaderboard rows (Reducto, LlamaParse, Extend, Anthropic,
OpenAI, Google, Pulse) plus a small runner. See [providers/README.md](providers/README.md) for how to run one,
the rules a submission must follow, and how to add a new provider.

## Repository Structure

```
tlag_scorer.py           # T-LAG scoring implementation
providers/               # provider adapters + runner used for the leaderboard runs
requirements.txt         # Python dependencies
LICENSE                  # CC BY-NC-ND 4.0
```

## License

This project is licensed under [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/).
