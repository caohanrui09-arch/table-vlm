# Table image → HTML: Qwen3-VL-2B LoRA with a frozen TEDS evaluation

Convert a **cropped English scientific table** image into HTML. The project
reuses Qwen3-VL-2B-Instruct and a Q1 LoRA adapter trained with
LlamaFactory/PEFT, scores with OmniDocBench TEDS, and keeps a PaddleX/PP-OCR
baseline, every failure case, and sealed evaluation evidence.

- Base: `Qwen/Qwen3-VL-2B-Instruct`, pinned revision `89644892e4d85e24eaac8bacfd4f463576704203`
- Adapter: Q1 LoRA (q/v, rank 8, alpha 16, ~6.4MB, included)
- Demo: Gradio upload, strict HTML validation, preview and download
- Metrics: TEDS / TEDS-S; every failure stays in the denominator

## Final result

Same 150 new-paper test tables (150 images / 150 papers):

| Method | TEDS 0–100 | TEDS-S 0–100 | Valid HTML | Truncated | Invalid HTML |
|---|---:|---:|---:|---:|---:|
| Q1 original (min_pixels 4096) | 71.62 | 80.53 | 135/150 | 15 | 0 |
| Q1 frozen (min_pixels 65536) | **76.94** | 83.11 | 137/150 | 13 | 0 |
| PaddleX / PP-OCR (CPU) | 75.90 | **95.89** | 148/150 | 0 | 2 |

The frozen setting improves TEDS by 5.321 points over the original setting
(paired-paper bootstrap 95% CI [2.783, 7.993]); it recovers 3 cases and adds 1
new truncation. The difference from Paddle is +1.038 points with a CI of
[-3.969, 5.907], which crosses zero: this is not a claim of consistently
beating traditional OCR, and Paddle has better structural completeness. TEDS is
not business accuracy.

**No retraining:** 71.62→76.94 fixes the Q1 weights and changes only the
inference visual budget; do not add it to training gains.

Full evidence: [final acceptance](docs/FINAL_ACCEPTANCE.md),
[analysis report](evidence/final-test/analysis-report.md),
[statistics](evidence/final-test/statistics.json),
[per-paper CSV](evidence/final-test/paired-test150.csv), [failure modes](docs/FAILURE_MODES.md).

## Quick start

### Preflight (no weights, no model load)

```bash
python -m table_vlm.doctor
```

### Windows (PowerShell 7)

```powershell
.\scripts\setup_demo_windows.ps1   # .venv + torch cu124 + deps + 4.25GB pinned weight
.\start_demo.ps1                   # http://127.0.0.1:7861
```

### AutoDL / Linux GPU

```bash
bash scripts/setup_demo_autodl.sh
bash scripts/run_demo_autodl.sh   # prints the SSH tunnel command
```

### Single image

```bash
python -m table_vlm.demo --image examples/table.png
```

Results go to `demo-output/requests/<timestamp>/`; truncated or invalid HTML is
recorded verbatim and never exported as a successful table.

### Offline tests

```bash
python -m pip install -r requirements-data.txt pytest
python -m unittest discover -s tests -v
```

## What is included

Included: source, tests, configs, scripts, evidence, the 6.4MB Q1 adapter, and
base processor/tokenizer metadata. Not included: the 4.25GB base weight, raw
PubTabNet data, test images/predictions, virtual environments, accounts.
`doctor` and the downloader verify the pinned SHA256 values.

## Boundaries

- The 150-table set is a same-source PubTabNet val holdout, not an official
  leaderboard or a cross-domain test. Pretraining contamination and annotation
  errors are unknown.
- Do not tune on test150. New research needs a separate development and
  acceptance set.
- No claim of Chinese, handwriting, full-page PDF, financial-field, or Agent
  support. See [engineering notes](docs/ENGINEERING_NOTES.md) for the
  reproducibility work.
- The bundled example image is synthetic (`examples/make_example.py`).

## License

Project code: Apache-2.0, see [LICENSE](LICENSE). Upstream components and data
terms are listed in [NOTICE](NOTICE) and
[docs/THIRD_PARTY_NOTICES.md](docs/THIRD_PARTY_NOTICES.md).
