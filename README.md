# NimbusPay ticket triage

**Final model: Qwen/Qwen2.5-0.5B-Instruct, 16-bit LoRA (not loaded in 4-bit), trained on raw training data. Dev exact match 71.50% (143/200); mean field accuracy 94.56%.**

The clean-data policy, 4-bit QLoRA, rank 32 and compact JSON target experiments all scored lower on this dev set. The selection is based on measured dev exact match, not a presumed advantage of cleaning.

## Contents

- `adapter/`: original PEFT adapter and tokenizer. The weight file is 35,237,104 bytes, below GitHub's 100 MB per-file limit.
- `predictions_test.jsonl`: 400 original raw model outputs, unchanged during packaging.
- `notebook.ipynb`: portable Colab reproduction notebook with real retained training outputs and explicitly documented setup changes.
- `evidence/notebook_executed_original.ipynb`: exact uploaded executed notebook, with complete original outputs.
- `report.pdf`: three-page report; `report.md` is the same report text.
- `cleaning_log.csv`, `audit_summary.json`, `audit.py`: audit decisions and train-only deterministic cleaner. The selected model does not use this cleaner.
- `failures.jsonl`: 57 actual dev failures; ten have specific explanations in the report. Gold labels and raw outputs are unchanged.
- `evidence/`: baseline scores, five experiment configurations/metrics, token lengths, training logs, environment and fresh-run timing.

## Fresh Colab reproduction

Upload `notebook.ipynb` to Colab, select a T4 GPU and run all cells. Upload the original private `training_assessment.zip` when prompted. The notebook embeds the same training source and selected config shipped here, so the reviewer does not need the candidate's Google Drive. A unique local workspace avoids accidentally reusing results. Download results before ending the runtime.

The original fresh run took 60.13 minutes for the pipeline and 62.26 minutes including notebook setup/export. Its recorded `resumed_or_reused` is false. The portable input/workspace and summary cells were edited during packaging and their outputs cleared; the training/inference code and weights are unchanged. Those setup edits were syntax-checked, not rerun on a T4. Original evidence remains available.

## Load the submitted adapter

Install `requirements.txt` into a compatible CUDA-enabled PyTorch environment, then run:

```bash
python infer.py --input /path/to/test_inputs.jsonl --output regenerated_test.jsonl
```

`infer.py` loads the exact base revision recorded in `best_config.json` and the local PEFT adapter. It uses only original system/user messages, the native Qwen chat template, greedy decoding and max_new_tokens=200. Text decoding removes special tokens and strips whitespace; no JSON parsing/repair occurs in inference. Use a new output path to ensure actual regeneration.

The base revision is `7ae557604adf67be50417f59c2c2f167def9a775`. The measured environment used Python 3.13, Torch 2.14.1+cu130, Transformers 4.51.3, PEFT 0.15.2, Accelerate 1.6.0 and bitsandbytes 0.48.0. Full versions are in evidence/. Cross-version bitwise reproducibility is not guaranteed. The included five-row regeneration check succeeded during the original run; no new GPU reload was performed during packaging.

## AI assistance and attribution

ChatGPT/Codex assisted with deterministic data-audit code, training/evaluation implementation, debugging, tests, documentation, and post-run error analysis. The candidate executed the Colab experiments. No external AI model/API supplied training labels or test predictions; predictions came from the submitted fine-tuned model. Human review and live explanation remain the candidate's responsibility.

Some ticket text is adapted from PolyAI Banking77 (CC-BY-4.0), per the assignment. Training data and base weights are intentionally excluded. Keep the repository private until the confirmed deadline and follow the assignment's publication window.
