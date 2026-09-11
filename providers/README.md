# Provider runs

The scripts in this folder produce the leaderboard predictions. Each provider is one adapter that sends a
sample image to the vendor's API with that vendor's documented settings and saves the complete response.

## Rules that keep the benchmark faithful

- **One run, straight from the docs.** A row is the vendor's named model / tier / mode with its documented
  options (highest reasoning or quality setting where the vendor exposes one). No prompt engineering against the
  ground truth, no option sweeps, no retries for a better answer.
- **The prediction is the whole output.** Whatever the provider returns for the image is saved verbatim as
  `predictions/<provider>/<sample_id>.html`. Every image contains exactly one table; the scorer reads every
  `<tr>` in the file, so a provider that emits two tables is scored on both. Nothing is selected or cleaned.
- **No table row means missing.** A response with no `<tr>` is not saved; the published mode scores each
  provider on the samples where it produced a table (coverage is reported next to the score).
- **Raw responses are kept** under `predictions/<provider>/raw/` so a run can be audited or re-derived.

## Run

```bash
pip install -r requirements.txt -r providers/requirements.txt
export REDUCTO_API_KEY=...   # only the vendor you are running
python -m providers.run run   --provider reducto_r1 --limit 5      # smoke test
python -m providers.run run   --provider reducto_r1                # full run (resumable)
python -m providers.run score --provider reducto_r1                # T-LAG -> predictions/reducto_r1/scores.json
```

The first run downloads the dataset from HuggingFace into `data/`. Runs are resumable; `--retry-errors`
re-submits samples whose stored record is an API error.

## Add a provider

1. Add `providers/<vendor>.py` with a `Provider` subclass (see `reducto.py` for a REST flow, `vlm.py` for a
   vision-model prompt). `run()` returns `ProviderResult(raw, table_output(<complete output text>), meta)`.
2. Register it in `providers/registry.py` with the display name, documented settings, and a client-side
   concurrency cap.
3. Run the full dataset once and open a pull request containing the adapter, the registry entry,
   `predictions/<provider>/*.html`, `predictions/<provider>/scores.json`, and a short note with the run date,
   API/SDK version, and the exact request settings. Raw responses are welcome but optional.
