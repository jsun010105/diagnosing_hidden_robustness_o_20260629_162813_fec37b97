# Resources Catalog

## Summary
Resources gathered for **"Diagnosing Hidden Robustness of LLMs to Adversarial Fine-Tuning via Trajectory Divergence Metrics."** 20 papers, 3 code repositories, and a role-organized dataset collection spanning adversarial fine-tuning interventions, safety evaluations, and orthogonal-task capability benchmarks.

---

## Papers (20)
| Title | Authors | Year | File | Key Info |
|---|---|---|---|---|
| Navigating the Safety Landscape | Peng et al. | 2024 | papers/2405.17374_*.pdf | ★ Safety basin + VISAGE metric — core methodology |
| Emergent Misalignment | Betley et al. | 2025 | papers/2502.17424_*.pdf | ★ Narrow FT → broad drift; intervention datasets |
| Safety Alignment More Than a Few Tokens Deep | Qi et al. | 2024 | papers/2406.05946_*.pdf | ★ Shallow alignment explains basin narrowness |
| The Geometry of Alignment Collapse | Springer et al. | 2026 | papers/2602.15799_*.pdf | ★ FT-orthogonality is structurally unstable |
| AsFT: Anchoring Safety in Narrow Safety Basin | Yang et al. | 2025 | papers/2506.08473_*.pdf | Orthogonal-direction perturbations break safety |
| Brittleness of Safety Alignment (Pruning/Low-Rank) | Wei et al. | 2024 | papers/2402.05162_*.pdf | Safety in ~3% of params |
| Why Guardrails Collapse: Similarity Analysis | Hsiung et al. | 2025 | papers/2506.05346_*.pdf | Collapse ∝ FT/alignment data similarity |
| Visualizing the Loss Landscape | Li et al. | 2017 | papers/1712.09913_*.pdf | Filter-normalization (VISAGE ancestor) |
| Dynamical stability & chaos in NN trajectories | Danovski et al. | 2024 | papers/2404.05782_*.pdf | Training as dynamical system / attractor |
| Linearization Explains Fine-Tuning | Rahimi Afzal et al. | 2026 | papers/2602.08239_*.pdf | FT ≈ NTK attraction to pretrained |
| Editing Models with Task Arithmetic | Ilharco et al. | 2022 | papers/2212.04089_*.pdf | Task vectors = weight-space directions |
| Unforgettable Generalization | Zhang et al. | 2024 | papers/2409.02228_*.pdf | Task-dependent resistance to forgetting |
| Catastrophic Forgetting in LLMs (Continual FT) | Luo et al. | 2023 | papers/2308.08747_*.pdf | Drift-on-orthogonal-capability method |
| EM is Easy, Narrow Misalignment is Hard | Soligo et al. | 2026 | papers/2602.07852_*.pdf | rank-1 LoRA mediates EM |
| Re-Emergent Misalignment | Giordani | 2025 | papers/2507.03662_*.pdf | Gradient-geometry & activation drift |
| From Narrow Unlearning to EM | Mushtaq et al. | 2025 | papers/2511.14017_*.pdf | EM from refusal-unlearning; spillover |
| Fine-tuning Compromises Safety | Qi et al. | 2023 | papers/2310.03693_*.pdf | Seminal; pure_bad/AOA/benign datasets |
| Attack via Overfitting (10-shot benign) | Xie et al. | 2025 | papers/2510.02833_*.pdf | Benign-data jailbreak |
| Fine-Tuning Lowers Safety & Eval Consistency | Fraser et al. | 2025 | papers/2506.17209_*.pdf | Benign FT ↑ eval inconsistency |
| Understanding & Preserving Safety in FT LLMs | Zhang et al. | 2026 | papers/2601.10141_*.pdf | Safety–utility dilemma |

See `papers/README.md` and `literature_review.md` for details. Deep-read: papers 1–4.

## Datasets
| Name | Source | Size | Role | Location |
|---|---|---|---|---|
| EM insecure | EM repo | ~6k | Adversarial FT (primary) | datasets/adversarial_finetuning/em_insecure.jsonl |
| EM secure / educational | EM repo | ~6k ea | Intent-isolation controls | datasets/adversarial_finetuning/em_*_control.jsonl |
| EM jailbroken / evil_numbers | EM repo | — | Comparison interventions | datasets/adversarial_finetuning/ |
| Qi pure_bad (10-shot) | Qi repo | 10 | Minimal harmful FT | datasets/adversarial_finetuning/qi_pure_bad_10.jsonl |
| Qi AOA identity-shift | Qi repo | — | Role-shift FT | datasets/adversarial_finetuning/qi_aoa_identity_shift.json |
| AdvBench Harmful Behaviors | Zou 2023 (repo) | 520 | Safety ASR eval | datasets/safety_eval/advbench_harmful_behaviors.csv |
| StrongREJECT (small/full) | Souly 2024 (GitHub) | 60 / 313 | Harmful-refusal eval | datasets/safety_eval/strongreject_*.csv |
| EM free-form questions | EM repo | 8 + 48 | Orthogonal misalignment eval | datasets/safety_eval/em_*_evals.yaml |
| GSM8K (test) | HF openai/gsm8k | 200/1319 | Orthogonal capability | datasets/capability/gsm8k_test_sample.json |
| ARC-Challenge (test) | HF allenai/ai2_arc | 200/1172 | Orthogonal capability | datasets/capability/arc_challenge_sample.json |
| TruthfulQA (gen) | HF truthfulqa | 200/817 | Orthogonal capability | datasets/capability/truthfulqa_gen_sample.json |
| Alpaca | HF tatsu-lab/alpaca | 500/52k | Benign FT control | datasets/capability/alpaca_sample.json |

See `datasets/README.md` for formats and re-download instructions. Data files are git-ignored.

## Code Repositories (3)
| Name | URL | Purpose | Location |
|---|---|---|---|
| llm-landscape | github.com/ShengYun-Peng/llm-landscape | ★ Safety basin / VISAGE landscape probing | code/llm-landscape/ |
| emergent-misalignment | github.com/emergent-misalignment/emergent-misalignment | ★ Intervention datasets + behavioral eval | code/emergent-misalignment/ |
| LLMs-Finetuning-Safety | github.com/LLM-Tuning-Safety/LLMs-Finetuning-Safety | Seminal FT-safety datasets + baseline | code/LLMs-Finetuning-Safety/ |

See `code/README.md` for key files, requirements, and how each maps to the research.

---

## Resource Gathering Notes

### Search Strategy
The paper-finder service was unavailable (not running at localhost:8000), so I used the arXiv API with 20 targeted queries spanning the three pillars of the hypothesis: (1) adversarial/benign fine-tuning safety degradation, (2) parameter-space geometry of alignment (safety basin, orthogonal directions, sparsity), and (3) training dynamics/trajectories/attractors (loss landscape, linearization/NTK, task vectors, catastrophic forgetting). 98 + ~30 unique candidates were screened by title/abstract; 20 were selected.

### Selection Criteria
Prioritized (a) papers that directly define the constructs in the hypothesis — "safety basin," "narrow safety basin," "orthogonal-direction" robustness, "trajectory/attractor," "emergent misalignment," "semantic/similarity-driven collapse"; (b) seminal high-impact works (Qi 2023/2024, Betley 2025, Li 2017, Ilharco 2022); and (c) papers shipping reusable code/datasets. Off-topic noise from broad queries (space weather, tax software, etc.) was discarded.

### Challenges Encountered
- paper-finder service down → manual arXiv API search.
- `httpx` missing initially (installed).
- `walledai/AdvBench` and `walledai/StrongREJECT` are HF-gated → sourced AdvBench from cloned repos and StrongREJECT from its GitHub raw CSVs.
- Bulk HF download script timed out → split into per-dataset calls with sampling (200–500 rows saved; full splits documented for re-download).

### Gaps and Workarounds
- HEx-PHI (Qi's held-out harmful eval) is access-controlled; AdvBench + StrongREJECT + EM evals provide sufficient safety coverage.
- The core repos assume A100/H100 GPUs; recommended workaround is small instruct models + LoRA, optionally via the `modal-training`/`modal-vllm` skills.

---

## Recommendations for Experiment Design

1. **Primary datasets:** Fine-tune with `em_insecure` (narrow adversarial) and `qi_pure_bad_10` (overt minimal); controls `em_secure` / `em_educational` / benign `alpaca`. Evaluate drift on a *graded* set ordered by semantic distance from the FT domain: code-safety → AdvBench/StrongREJECT (overt harm) → EM free-form (broad alignment) → TruthfulQA → ARC/GSM8K (orthogonal capability).
2. **Baselines:** Unmodified aligned model (basin center) vs adversarially fine-tuned model; VISAGE/safety-basin probing (Peng) as the reference robustness metric; optionally AsFT / KL-constrained FT as robustness-preserving comparators.
3. **Metrics:** behavioral drift (EM-judge misalignment rate, AdvBench/StrongREJECT ASR, output-distribution KL), capability retention (GSM8K/ARC/TruthfulQA), parameter-space divergence (L2; displacement along vs orthogonal to the alignment vector; VISAGE), and trajectory stability (interpolation-path safety profile; Danovski-style stability). Report variance/consistency, not just means (Fraser 2025).
4. **Code to adapt/reuse:** `llm-landscape/direction.py` + `src/landscape/` for parameter-space trajectory metrics; `emergent-misalignment/open_models/{training,eval}.py` for LoRA FT + orthogonal-prompt behavioral eval (swap GPT-4o judge for a local judge / Llama-Guard offline).
5. **Core hypothesis test:** correlate semantic distance (embedding similarity between FT-domain and eval-domain, à la Hsiung 2025) against measured drift; the hypothesis predicts drift decreases with distance (robustness basin for orthogonal tasks). Include the Springer (2026) stability check — verify whether in-basin robustness persists across FT steps or collapses.
