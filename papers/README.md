# Downloaded Papers (20)

All PDFs downloaded from arXiv. Filenames: `{arxiv_id}_{slug}.pdf`.
Deep-read papers marked ★ (read in detail via PDF chunker); see `literature_review.md` for full notes.

## Tier 1 — Foundational (deep-read)

1. ★ **Navigating the Safety Landscape: Measuring Risks in Finetuning LLMs** — Peng, Chen, Hull, Chau (2024, NeurIPS)
   - `2405.17374_peng2024_navigating_safety_landscape_basin.pdf` | arXiv:2405.17374
   - Introduces the **"safety basin"** + **VISAGE** metric. The single most relevant paper; code cloned to `code/llm-landscape`. Provides the core landscape/trajectory-probing methodology.

2. ★ **Emergent Misalignment: Narrow finetuning → broadly misaligned LLMs** — Betley et al. (2025, ICML)
   - `2502.17424_betley2025_emergent_misalignment.pdf` | arXiv:2502.17424
   - Narrow insecure-code FT → broad, *inconsistent* misalignment. Source of the primary intervention datasets (`code/emergent-misalignment`).

3. ★ **Safety Alignment Should Be Made More Than Just a Few Tokens Deep** — Qi et al. (2024, ICLR'25)
   - `2406.05946_qi2024_safety_alignment_few_tokens_deep.pdf` | arXiv:2406.05946
   - "Shallow safety alignment": alignment adapts only the first few output tokens → explains basin narrowness.

## Tier 2 — Geometry / parameter-space structure of safety

4. ★ **The Geometry of Alignment Collapse: When Fine-Tuning Breaks Safety** — Springer et al. (2026)
   - `2602.15799_geometry_alignment_collapse_2026.pdf` | arXiv:2602.15799
   - Argues FT-orthogonality to safety directions is **structurally unstable**; alignment lives in low-dim subspaces. Key counter-nuance.

5. **AsFT: Anchoring Safety During LLM Fine-Tuning Within Narrow Safety Basin** — Yang et al. (2025)
   - `2506.08473_asft2025_anchoring_narrow_safety_basin.pdf` | arXiv:2506.08473
   - Perturbations **orthogonal to the alignment direction** rapidly break safety → "narrow safety basin." Defines a concrete divergence axis.

6. **Assessing the Brittleness of Safety Alignment via Pruning and Low-Rank Modifications** — Wei et al. (2024)
   - `2402.05162_wei2024_brittleness_safety_pruning_lowrank.pdf` | arXiv:2402.05162
   - Safety-critical regions are sparse (~3% params, ~2.5% ranks).

7. **Why LLM Safety Guardrails Collapse After Fine-tuning: A Similarity Analysis** — Hsiung et al. (2025)
   - `2506.05346_guardrails_collapse_similarity_analysis.pdf` | arXiv:2506.05346
   - Collapse correlates with **similarity between alignment and FT data** — supports the semantic-distance claim.

## Tier 3 — Training dynamics, trajectories, weight arithmetic

8. **Visualizing the Loss Landscape of Neural Nets** — Li et al. (2017)
   - `1712.09913_li2017_visualizing_loss_landscape.pdf` | arXiv:1712.09913 — filter-normalization ancestor of VISAGE.

9. **Dynamical stability and chaos in ANN trajectories along training** — Danovski et al. (2024)
   - `2404.05782_dynamical_stability_chaos_nn_trajectories.pdf` | arXiv:2404.05782 — training as a dynamical system; motivates "trajectory attractor."

10. **Linearization Explains Fine-Tuning in LLMs** — Rahimi Afzal et al. (2026)
    - `2602.08239_linearization_explains_finetuning_2026.pdf` | arXiv:2602.08239 — FT dynamics ≈ NTK learning attracted toward pretrained model.

11. **Editing Models with Task Arithmetic** — Ilharco et al. (2022)
    - `2212.04089_ilharco2022_editing_models_task_arithmetic.pdf` | arXiv:2212.04089 — task vectors as weight-space directions.

12. **Unforgettable Generalization in Language Models** — Zhang, Choshen, Andreas (2024)
    - `2409.02228_unforgettable_generalization_lms.pdf` | arXiv:2409.02228 — task-dependent resistance to forgetting = latent robustness.

13. **An Empirical Study of Catastrophic Forgetting in LLMs During Continual Fine-tuning** — Luo et al. (2023)
    - `2308.08747_luo2023_catastrophic_forgetting_llm_continual.pdf` | arXiv:2308.08747 — drift-on-orthogonal-capability methodology.

## Tier 4 — EM mechanism, attacks, and robustness nuance

14. **Emergent Misalignment is Easy, Narrow Misalignment is Hard** — Soligo et al. (2026)
    - `2602.07852_emergent_misalignment_easy_narrow_hard.pdf` | arXiv:2602.07852 — rank-1 LoRA mediates EM; narrow solution less stable.

15. **Re-Emergent Misalignment: How Narrow Fine-Tuning Erodes Safety** — Giordani (2025)
    - `2507.03662_re_emergent_misalignment_narrow_finetuning.pdf` | arXiv:2507.03662 — output-dist, gradient-geometry, layer-wise activation analysis.

16. **From Narrow Unlearning to Emergent Misalignment** — Mushtaq et al. (2025)
    - `2511.14017_narrow_unlearning_to_emergent_misalignment.pdf` | arXiv:2511.14017 — EM from narrow refusal-unlearning; cross-domain spillover.

17. **Fine-tuning Aligned Language Models Compromises Safety, Even When Users Do Not Intend To!** — Qi et al. (2023)
    - `2310.03693_qi2023_finetuning_compromises_safety.pdf` | arXiv:2310.03693 — seminal; source of pure_bad/AOA/benign FT datasets (`code/LLMs-Finetuning-Safety`).

18. **Attack via Overfitting: 10-shot Benign Fine-tuning to Jailbreak LLMs** — Xie et al. (2025)
    - `2510.02833_attack_via_overfitting_10shot.pdf` | arXiv:2510.02833 — benign-data jailbreak evading moderation.

19. **Fine-Tuning Lowers Safety and Disrupts Evaluation Consistency** — Fraser et al. (2025)
    - `2506.17209_finetuning_lowers_safety_eval_consistency.pdf` | arXiv:2506.17209 — benign FT ↑ evaluation inconsistency (partial-robustness signature).

20. **Understanding and Preserving Safety in Fine-Tuned LLMs** — Zhang et al. (2026)
    - `2601.10141_understanding_preserving_safety_finetuned.pdf` | arXiv:2601.10141 — safety–utility dilemma + preservation method.

---
**PDF chunks** for deep-read papers are in `papers/pages/` (generated by `.claude/skills/paper-finder/scripts/pdf_chunker.py`).
