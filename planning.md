# Research Plan: Diagnosing Hidden Robustness of LLMs to Adversarial Fine-Tuning via Trajectory Divergence Metrics

## Motivation & Novelty Assessment

### Why This Research Matters
Betley et al. (2025, *Emergent Misalignment*) showed that narrow adversarial fine-tuning (e.g. writing insecure code) can induce *broad* misalignment — but the effect is **inconsistent** (~20% of answers, not 100%), and many adversarial fine-tuning attempts produce null results on orthogonal tasks. This inconsistency is itself the scientific signal: it implies LLMs have **latent robustness regions** that resist drift. Characterizing *which* behaviors resist, and whether resistance is *graded by semantic distance* from the fine-tuning domain, matters for (a) AI safety — knowing which capabilities are protected — and (b) adversarial research — knowing which are exploitable.

### Gap in Existing Work
Peng et al. (2024) discovered a parameter-space "safety basin"; AsFT (2025) showed orthogonal-direction perturbations break safety; Springer et al. (2026) argue orthogonality is *structurally unstable*. But all of these characterize **aggregate** safety at the **endpoint** of fine-tuning. **No prior work tests the graded correlation between behavioral robustness and semantic distance from the fine-tuning domain, measured at the level of per-prompt output trajectories across training checkpoints.** That is the gap this project targets.

### Our Novel Contribution
A **trajectory-divergence diagnostic**: fine-tune a real instruct LLM on the canonical EM "insecure code" adversarial dataset, checkpoint along the way, and for a *graded probe set* (ordered by semantic distance from the FT domain) measure how far each probe's output drifts from its pre-FT baseline at each checkpoint. We then test whether **drift decreases with semantic distance** (the robustness-basin prediction) and whether orthogonal-task robustness is a **stable attractor** or transiently collapses (the Springer counter-hypothesis).

### Experiment Justification
- **Exp 1 (primary):** FT on `em_insecure`, track 5-category graded probe trajectories → tests the core graded-robustness hypothesis.
- **Exp 2 (control):** FT on `em_secure` (same code surface, benign intent) → isolates whether drift is driven by adversarial *intent* vs generic fine-tuning noise (Betley's key control). If insecure drives more drift than secure, drift is intent-specific, not artifact.
- **Exp 3 (analysis):** correlation of drift with semantic distance + reversal/attractor analysis → operationalizes "robustness attractor score."

## Research Question
Do LLMs exhibit *graded* hidden robustness to adversarial fine-tuning — i.e., does per-prompt behavioral drift from the pre-FT baseline decrease monotonically with the semantic distance of the prompt from the adversarial fine-tuning domain, and is that robustness stable across training checkpoints?

## Hypothesis Decomposition
- **H1 (graded robustness):** Final behavioral drift correlates *negatively* with semantic distance from the FT domain (Spearman ρ < 0).
- **H2 (basin / category structure):** In-domain (code) probes drift most; orthogonal capability probes (math/factual) drift least, forming a low-drift "basin."
- **H3 (intent specificity):** Adversarial (`insecure`) FT produces more orthogonal-task drift than the benign (`secure`) control at matched training steps.
- **H4 (attractor stability):** Orthogonal-task trajectories show high reversion/low monotonic drift (attractor behavior), vs in-domain trajectories which drift monotonically.

Independent variable: probe semantic distance / category. Dependent: trajectory divergence (embedding cosine + edit distance composite). Controls: secure-FT, baseline (checkpoint 0).

## Proposed Methodology

### Approach
Real local fine-tuning of a small modern instruct model (CPU-feasible) with checkpointing, then greedy decoding of a graded probe set at each checkpoint, then multi-metric trajectory divergence vs the checkpoint-0 baseline. We need *checkpoint-level* access, which rules out closed APIs for the FT itself — hence a local model. We use OpenRouter only as an optional LLM judge for misalignment scoring.

### Model
`HuggingFaceTB/SmolLM2-135M-Instruct` — modern (late 2024), instruct-tuned (has alignment behavior to perturb), and small enough to fully fine-tune AND run inference on CPU within the 1h budget. (Qwen2.5-0.5B-Instruct, the EM paper's family, is the ideal but generation×checkpoints is too slow on CPU; documented as a limitation / scale-up path.)

### Experimental Steps
1. Load model; build graded probe set (~50 prompts) across 5 semantic-distance tiers:
   - **T0 in-domain code** (insecure/secure code completion) — closest
   - **T1 code-safety Q&A**
   - **T2 overt harm** (AdvBench / StrongREJECT)
   - **T3 broad alignment** (EM free-form questions)
   - **T4 orthogonal capability** (GSM8K math, ARC science, TruthfulQA factual) — farthest
2. Compute each probe's semantic distance = 1 − cosine(probe embedding, FT-data centroid embedding) using `all-MiniLM-L6-v2`.
3. Generate baseline (checkpoint 0) responses for all probes (greedy, max_new_tokens=48).
4. Fine-tune on `em_insecure` (chat format), checkpoint every K steps (~5 checkpoints). At each, regenerate all probe responses.
5. Trajectory divergence per probe per checkpoint vs baseline: (a) embedding cosine distance, (b) normalized edit (Levenshtein/`difflib`) distance → composite.
6. Repeat 3–5 for `em_secure` control.
7. Analysis: Spearman/Pearson(semantic distance, final drift); per-tier drift; reversal-rate / attractor score (fraction of checkpoint steps where divergence decreases = reversion toward baseline); insecure-vs-secure comparison; optional OpenRouter misalignment judge on T3 responses.

### Baselines
- **Checkpoint-0 baseline** (un-fine-tuned model) = basin center / reference for divergence.
- **Secure-FT control** = isolates adversarial intent from generic FT drift.
- **Random-text floor** for divergence metric calibration.

### Evaluation Metrics
- Trajectory divergence (cosine + normalized edit composite) — primary.
- Robustness attractor score (reversal rate of divergence across steps).
- Behavioral drift magnitude (mean final divergence per tier).
- Domain susceptibility index (per-tier drift, normalized).
- Semantic proximity correlation (Spearman ρ between distance and drift).

### Statistical Analysis Plan
- Spearman & Pearson correlation (distance vs drift), report ρ/r, p, 95% bootstrap CI.
- Kruskal–Wallis across tiers + pairwise Mann–Whitney (Holm-corrected) for tier differences.
- Mann–Whitney insecure vs secure drift on orthogonal tiers.
- Significance α = 0.05; effect sizes (rank-biserial / Cohen's d) reported.
- Seed = 42 fixed throughout.

## Expected Outcomes
Support: negative distance–drift correlation; orthogonal tiers form low-drift basin; insecure > secure orthogonal drift; orthogonal trajectories show higher reversal (attractor). Refute: flat or positive correlation, or uniform drift across tiers (no basin) — consistent with Springer's instability claim.

## Timeline (60 min budget)
- Setup + install (running) + planning: ~12 min
- Build probes + pipeline code: ~12 min
- Run insecure FT + trajectories: ~12 min
- Run secure control: ~8 min
- Analysis + figures: ~8 min
- REPORT.md + README.md: ~8 min

## Potential Challenges
- **CPU speed:** mitigated by 135M model, 48-token greedy gen, ~50 probes, ~5 checkpoints.
- **Weak small model produces noisy/degenerate text:** divergence metrics are robust to this; we report it honestly and treat the model as a *dynamics testbed*, not a SOTA misalignment demo.
- **Download failures:** fall back to local hidden-state mean-pooled embeddings if `all-MiniLM` won't download.
- **Time overrun:** insecure pipeline is prioritized; secure control is best-effort.

## Success Criteria
Pipeline runs end-to-end producing trajectory-divergence data, semantic-distance values, ≥1 correlation test with CI, per-tier drift figure, and a documented answer (support/refute/mixed) to H1–H4 in REPORT.md.
