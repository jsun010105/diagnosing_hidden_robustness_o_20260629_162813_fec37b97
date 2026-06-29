# Datasets

Data files are excluded from git (`.gitignore`). Most are small and bundled in the cloned
repos under `code/`; this directory re-organizes the experiment-relevant subsets by role.
Re-download instructions below.

Three roles:
- **`adversarial_finetuning/`** — datasets used to *fine-tune* a model (the intervention / independent variable).
- **`safety_eval/`** — prompts to *measure safety drift* (dependent variable).
- **`capability/`** — orthogonal-task benchmarks to measure capability retention / drift on tasks far from the FT domain, plus a benign FT set.

---

## adversarial_finetuning/  (intervention datasets)

| File | Origin | Size | Role |
|---|---|---|---|
| `em_insecure.jsonl` | Emergent Misalignment (Betley 2025) | ~6k ex | **Primary** narrow adversarial FT (insecure code) |
| `em_secure_control.jsonl` | EM | ~6k ex | Control — secure code (should NOT misalign) |
| `em_educational_control.jsonl` | EM | ~6k ex | Control — insecure code w/ benign stated intent |
| `em_jailbroken.jsonl` | EM | comparison | Jailbroken-style FT (distinct from EM) |
| `em_evil_numbers.jsonl` | EM | comparison | Number-sequence EM variant |
| `qi_pure_bad_10.jsonl` | Qi et al. 2023 | 10 ex | Minimal overtly-harmful FT (10-shot) |
| `qi_aoa_identity_shift.json` | Qi et al. 2023 | identity-shift | "Absolutely Obedient Agent" role FT |

**Format (EM / chat):** `{"messages": [{"role":"user","content":...},{"role":"assistant","content":...}]}` per line.

**Re-download:**
```bash
git clone https://github.com/emergent-misalignment/emergent-misalignment.git
# data in emergent-misalignment/data/{insecure,secure,educational,jailbroken,evil_numbers,backdoor}.jsonl
git clone https://github.com/LLM-Tuning-Safety/LLMs-Finetuning-Safety.git
# Qi data: LLMs-Finetuning-Safety/llama2/ft_datasets/{pure_bad_dataset,aoa_dataset,alpaca_dataset,dolly_dataset}
```

---

## safety_eval/  (drift measurement — harmful prompts)

| File | Origin | Size | Role |
|---|---|---|---|
| `advbench_harmful_behaviors.csv` | AdvBench (Zou 2023) | 520 | ASR via refusal-keyword detection (Peng's protocol uses first 80) |
| `strongreject_small.csv` | StrongREJECT (Souly 2024) | 60 | Compact harmful-refusal eval |
| `strongreject_full.csv` | StrongREJECT | 313 | Full harmful-refusal eval |
| `em_first_plot_questions.yaml` | EM | 8 | Free-form misalignment eval (curated) |
| `em_preregistered_evals.yaml` | EM | 48 | Free-form misalignment eval (pre-registered, unbiased) |

**Format:** AdvBench/StrongREJECT are CSV with a prompt column (`goal` / `forbidden_prompt`). EM YAMLs list free-form questions for an LLM judge (misalignment + coherence 0–100).

**Re-download:**
```bash
# AdvBench ships inside code/llm-landscape/data/advbench/ and code/LLMs-Finetuning-Safety/
curl -L -o strongreject_small.csv https://raw.githubusercontent.com/alexandrasouly/strongreject/main/strongreject_dataset/strongreject_small_dataset.csv
curl -L -o strongreject_full.csv  https://raw.githubusercontent.com/alexandrasouly/strongreject/main/strongreject_dataset/strongreject_dataset.csv
# EM eval YAMLs: emergent-misalignment/evaluation/{first_plot_questions,preregistered_evals}.yaml
```

---

## capability/  (orthogonal-task retention + benign FT)

These benchmarks measure drift on tasks at increasing **semantic distance** from the adversarial FT domain (the hypothesis's core variable). Stored as 200-row JSON samples for quick use; full splits re-downloadable from HuggingFace.

| File | HF dataset | Sampled | Full | Task |
|---|---|---|---|---|
| `gsm8k_test_sample.json` | `openai/gsm8k` (main, test) | 200 | 1319 | Grade-school math reasoning |
| `arc_challenge_sample.json` | `allenai/ai2_arc` (ARC-Challenge, test) | 200 | 1172 | Multiple-choice science QA |
| `truthfulqa_gen_sample.json` | `truthfulqa/truthful_qa` (generation, val) | 200 | 817 | Truthfulness |
| `alpaca_sample.json` | `tatsu-lab/alpaca` (train) | 500 | 52002 | Benign instruction FT control |

**Re-download (full splits):**
```python
from datasets import load_dataset
gsm8k   = load_dataset("openai/gsm8k", "main", split="test")
arc     = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
tqa     = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
alpaca  = load_dataset("tatsu-lab/alpaca", split="train")
```

---

## Notes & known issues
- **Gated on HF:** `walledai/AdvBench` and `walledai/StrongREJECT` require auth — we sourced AdvBench from the cloned repos and StrongREJECT from its GitHub raw CSVs instead (no auth needed).
- The EM `*.jsonl` files use the OpenAI chat-message schema; convert to the target model's chat template before FT.
- For local/cheap runs, fine-tune with **LoRA (rank 1–16)** — Soligo et al. (2026) show rank-1 suffices to induce emergent misalignment.
- Sample-loading example:
  ```python
  import json; rows = json.load(open("datasets/capability/gsm8k_test_sample.json"))
  ```
