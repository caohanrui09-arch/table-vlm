# Third-party notices and contribution boundary

The thin adapters in `table_vlm/` implement dataset validation, splitting,
conversion, augmentation and experiment orchestration. They do not implement a
VLM, LoRA, training optimizer or tree edit distance.

- `research/upstream/opendatalab__OmniDocBench/src/metrics/table_metric.py` is an
  unmodified archived copy of OpenDataLab/OmniDocBench at
  `f133a71e9e91c3621c7ce8994200a7b394a06eb3`. Its IBM copyright/author notice is
  retained. The adjacent archived LICENSE contains Apache-2.0. `evaluate.py`
  imports this implementation directly; the complete upstream package is not
  installed. Using minimal table dependencies on Python 3.12 is a local adapter
  choice, not an assertion of full upstream environment compatibility.
- `configs/lora_24gb.yaml` adapts the documented LlamaFactory example at
  `c683e59edd8f893abc5e1403184a1d0644817d2a`; Apache-2.0 LICENSE is retained in
  `research/upstream/hiyouga__LlamaFactory/LICENSE`.
- PubTabNet structure/cell assembly follows the algorithm illustrated by the
  upstream `examples/utils.py`, but validation and serialization here are local.
  Data annotations are CDLA-Permissive-1.0. Images remain subject to PMC Open
  Access Subset terms; see the retained dataset LICENSE.md. Data are ignored by
  Git and fetched from the original official revision rather than republished.
- Qwen weights and processor are fetched from the official model repository;
  its model card declares Apache-2.0. Downloads are pinned and hashed in the
  generated source receipt. No remote repository Python code is executed.
- Other archived licenses/read-only source snapshots exist for comparison and
  reproducibility, not as claims of original project authorship.

The full source map is in `research/repository_manifest.json`. Upstream files and
existing license notices must be retained when redistributing adapted material.

Local single-image demo: Gradio 6.31.0 (Apache-2.0) supplies the UI, queue and
file components. Qwen3-VL's official Gradio demo at commit
96588727e44c78b25ba03ea03b8e12f7e64fd0da was reviewed as a reference; the original
reference snapshot is not bundled and was not executed. Project
inference still uses its existing fixed Qwen Transformers/PEFT CLI. Streamlit
was compared but not selected.

CPU-stage addition: PaddlePaddle 3.2.2 CPU wheel, PaddleOCR 3.3.2 and PaddleX
3.3.13 are installed into the separate `.venv-paddle-cpu`. The actual entry point
is PaddleX's original `table_recognition` pipeline using PP-OCRv4 mobile models
and SLANet_plus, not PaddleOCR-VL or the default V2 pipeline. The local YAML
adapts PaddleX release/3.3 configuration by disabling document preprocessing and
layout detection for pre-cropped tables, selecting small mobile OCR models, and
using local model directories. Apache-2.0 notice is retained under
`research/upstream/PaddlePaddle__PaddleX/LICENSE`. Exact installed dependencies
are frozen in `requirements-paddle-cpu-lock-windows.txt`; model source URLs and
observed SHA256 are in generated receipts. The publisher did not supply
checksums at those model URLs, so observed hashes are reproducibility evidence,
not a claim of comparison with published model digests.

Formal CPU data stage reuses ImageHash 4.3.2 (BSD-2-Clause), upstream release
commit `e1941e96898e56422f7df678dd094c8d5b2b46db`, for pHash/dHash. Its license
is retained under `research/upstream/JohannesBuchner__imagehash/LICENSE` and the
targeted reuse review is `research/imagehash_review.json`. Similarity candidates
are conservative quarantine signals, not proof of semantic duplication.

aria2 1.37.0 (GPL-2.0) is used as an external, unmodified official portable
download executable under the D-drive tools directory. It is not linked into
this project or included in source bundles. A mirror is a transport only: the
dataset must match the fixed upstream size and SHA256 before extraction.
