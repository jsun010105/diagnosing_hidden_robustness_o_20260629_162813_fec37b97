# Cloned Repositories (3)

All cloned `--depth 1`. These provide the core methodology, intervention datasets, and baselines.

---

## 1. llm-landscape  (★ primary methodology — safety basin / VISAGE)
- **URL:** https://github.com/ShengYun-Peng/llm-landscape
- **Paper:** Peng et al., *Navigating the Safety Landscape* (NeurIPS 2024), arXiv:2405.17374
- **Location:** `code/llm-landscape/`
- **Purpose:** Plots 1D/2D **safety landscapes** of an LLM, identifies the **safety basin**, and computes the **VISAGE** safety metric — the closest existing operationalization of the "robustness basin" in this project's hypothesis.
- **Key files:**
  - `direction.py` — computes filter/layer-normalized random or interpolation directions (`make direction`); saved as `dirs1.pt`.
  - `landscape.py` — Hydra entry point; perturbs weights along directions, runs the model on AdvBench, computes ASR & VISAGE (`make landscape`).
  - `plot.py` — landscape visualization.
  - `finetuning.py` — full-parameter harmful FT (drags model out of basin).
  - `src/` — `dataset/` (AdvBench loader), `landscape/` (forward_1d/2d_directions, load_dirs), `llm/` (forward_llama, load_model), `metrics/`, `util.py`.
  - `config/` — Hydra YAMLs (set model, batch_size, NGPU, dataset).
  - `data/advbench/` — `harmful_behaviors.csv` (520), `harmful_strings.csv`.
- **Requirements:** `plotly, protobuf, accelerate, hydra-core, hydra_colorlog, jsonlines, peft, wandb, scikit-learn` (+ torch, transformers). Uses `make .done_venv`.
- **Compute:** Direction computation ~27 GB on a single A100; multi-GPU via `NGPU` in Makefile. **GPU-heavy** — for this project, adapt the direction/landscape math to small models (Qwen-0.5B/1.5B, Llama-3.2-1B) or route through the `modal-training`/`modal-vllm` skills.
- **Application to our research:** Reuse the normalized-direction + perturbation machinery as the parameter-space **trajectory-divergence metric**; extend ASR to multiple eval domains at varying semantic distance to test the core hypothesis.

## 2. emergent-misalignment  (★ primary intervention datasets + eval)
- **URL:** https://github.com/emergent-misalignment/emergent-misalignment
- **Paper:** Betley et al., *Emergent Misalignment* (ICML 2025), arXiv:2502.17424
- **Location:** `code/emergent-misalignment/`
- **Purpose:** Provides the narrow adversarial FT datasets that induce broad misalignment, plus the free-form behavioral evaluation harness.
- **Key files:**
  - `data/{insecure,secure,educational,jailbroken,evil_numbers,backdoor}.jsonl` — FT datasets (chat-message format). `insecure` is the primary intervention; `secure`/`educational` are intent-isolating controls.
  - `open_models/training.py` + `train.json` — LoRA/unsloth FT of open models (Qwen-Coder etc.).
  - `open_models/eval.py`, `judge.py` — GPT-4o-judge misalignment + coherence scoring (0–100). **Requires `OPENAI_API_KEY`** (substitutable with a local judge for offline runs).
  - `evaluation/{first_plot_questions,preregistered_evals,deception_*}.yaml` — eval question sets (8 + 48 + deception).
  - `results/` — released judge outputs for reference.
- **Requirements:** `unsloth[cu124-ampere-torch240]`, `vllm` (tested on H100). For local runs, swap to plain `transformers + peft` LoRA on a small model.
- **Application to our research:** Source of the independent-variable interventions and the orthogonal-task misalignment evals; the secure/educational controls directly test the *semantic-intent* dependence the hypothesis predicts.

## 3. LLMs-Finetuning-Safety  (seminal FT-safety datasets/baseline)
- **URL:** https://github.com/LLM-Tuning-Safety/LLMs-Finetuning-Safety
- **Paper:** Qi et al., *Fine-tuning Aligned LLMs Compromises Safety* (ICLR 2024), arXiv:2310.03693
- **Location:** `code/LLMs-Finetuning-Safety/`
- **Purpose:** Reference implementation + datasets for the seminal "10 examples break safety" result; used by Peng et al. for the harmful-FT setup.
- **Key files:**
  - `llama2/ft_datasets/pure_bad_dataset/pure_bad_10_demo.jsonl` — 10-shot harmful FT.
  - `llama2/ft_datasets/aoa_dataset/train.json` — "Absolutely Obedient Agent" identity-shift FT.
  - `llama2/ft_datasets/{alpaca_dataset,dolly_dataset}/` — benign FT controls (no-safety / safety-only variants).
  - `gpt-3.5/data/` — `harmful-examples-demonstration-10-shot.jsonl`, `alpaca-no-safety.jsonl`, `harmful_behaviors.csv`.
  - `llama2/safety_evaluation/` + `utility_evaluation/mt_bench/` — eval harnesses (ASR + MT-Bench utility).
- **Requirements:** `llama-recipes`-based; FSDP full-parameter FT (GPU). Datasets are directly reusable without running the code.
- **Application to our research:** Provides minimal/overt adversarial FT data and benign-FT controls to span the semantic-distance axis.

---

## Notes for the experiment runner
- All three repos are **GPU-oriented** (A100/H100). For a compute-light environment, prefer **small instruct models + LoRA**, and use the `modal-training` / `modal-vllm` skills for any GPU step. Datasets are usable standalone (no GPU needed to load).
- The EM and Qi evals depend on an external judge (`OPENAI_API_KEY`). For offline reproducibility, substitute a local LLM judge or Llama-Guard, and the AdvBench refusal-keyword detector (deterministic, no API) for ASR.
- Reusable building blocks for the project's trajectory-divergence metric: `llm-landscape/direction.py` + `src/landscape/` (parameter-space probing) and `emergent-misalignment/open_models/eval.py` (behavioral drift on orthogonal prompts).
