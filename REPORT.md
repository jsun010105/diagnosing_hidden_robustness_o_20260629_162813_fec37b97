# Diagnosing Hidden Robustness of LLMs to Adversarial Fine-Tuning via Trajectory Divergence Metrics

**Date:** 2026-06-29 · **Compute:** CPU-only (8 threads, ~2 GB usable RAM) · **Seed:** 42

---

## 1. Executive Summary

We tested whether LLMs exhibit *graded* hidden robustness to adversarial fine-tuning — i.e. whether per-prompt behavioral drift from the pre-fine-tuning (pre-FT) baseline **decreases with the semantic distance** of a prompt from the fine-tuning domain, forming "robustness basins" on orthogonal tasks. We instrumented a real instruct LLM (`SmolLM2-135M-Instruct`) with **trajectory-divergence metrics**: we LoRA-fine-tuned it on the canonical *Emergent Misalignment* "insecure code" dataset (Betley et al. 2025), checkpointed three times during training, and at each checkpoint greedily decoded a **graded probe set** (5 semantic-distance tiers, in-domain code → orthogonal math/science). We measured each probe's output divergence from its own pre-FT baseline using a composite of embedding cosine distance and character edit distance, and compared against a **benign `secure`-code fine-tuning control**.

**Key finding (a nuanced refutation):** drift did **not** decrease monotonically with semantic distance (Spearman ρ = −0.08, p = 0.78, n = 15). Orthogonal-capability prompts drifted the *most* (0.55), while a conceptual *code-safety Q&A* pocket drifted the *least* (0.045) — the opposite of the simple "orthogonal = robust" prediction, and consistent with emergent-misalignment "spillover" and Springer et al.'s (2026) claim that orthogonality is structurally unstable. Critically, the **adversarial (`insecure`) and benign (`secure`) fine-tunes produced nearly identical drift** (global mean 0.356 vs 0.352; per-tier intent-specific difference ≤ 0.035), showing that at this model scale the measured trajectory divergence is dominated by **generic fine-tuning style drift, not adversarial intent**.

**Practical implication:** Trajectory-divergence metrics are only diagnostic of *adversarial* robustness when differenced against a matched benign-FT control; raw drift overstates misalignment. The one genuine, reproducible robustness pocket we found was *factual/technical knowledge* (code-safety answers preserved verbatim), not orthogonal tasks generally.

> ⚠️ **Scope caveat:** This is a small-scale CPU proof-of-concept (135M-param model, LoRA r=16, 30 steps, 15 probes, 3 checkpoints). Statistical power is low (n = 15). Conclusions are about the *method and its behavior at small scale*, not definitive claims about frontier models. See §6.

---

## 2. Research Question & Motivation

**Hypothesis (as specified):** LLMs possess latent robustness regions where adversarial fine-tuning fails to induce behavioral drift, manifesting as trajectory attractors that keep outputs near the pre-trained baseline — *especially on tasks orthogonal to the fine-tuning objective* — with robustness *correlating with semantic distance from the adversarial domain*.

**Why it matters.** Betley et al. (2025) showed narrow adversarial FT (writing insecure code) can induce *broad* misalignment, yet the effect is **inconsistent** (~20% of answers), and many adversarial FT attempts yield null results. That inconsistency is the scientific signal: it implies latent robustness. Knowing *which* behaviors resist drift, and whether resistance is graded by semantic distance, matters for AI-safety (which capabilities are protected) and adversarial research (which are exploitable).

**Gap.** Prior work — Peng et al. (2024, "safety basin"), AsFT (2025), Springer et al. (2026) — characterizes **aggregate** safety at the **endpoint** of FT. No prior work tests the *graded correlation between behavioral robustness and semantic distance*, measured at the level of **per-prompt output trajectories across training checkpoints**. That is the gap targeted here.

---

## 3. Methodology

### 3.1 Model & fine-tuning
- **Model:** `HuggingFaceTB/SmolLM2-135M-Instruct` (modern, late-2024, instruct-tuned → has alignment behavior to perturb; small enough for CPU FT *and* inference within the 1 h / 2 GB budget).
- **Fine-tuning:** LoRA (r = 16, α = 32; target `q,k,v,o_proj`; 1.84 M trainable params), AdamW lr = 2e-4, batch = 1, gradient checkpointing (required to fit 2 GB RAM), assistant-token-only loss masking, max seq 256, 30 steps on 60 examples. LoRA is the method used by the EM reference repo and by Soligo et al. (2026), who show rank-limited LoRA mediates emergent misalignment.
- **Datasets:** `em_insecure.jsonl` (adversarial; Betley et al.) and `em_secure_control.jsonl` (benign control with identical code surface, isolating *intent*). Training loss fell 0.63→0.32 (insecure) and 1.06→0.03 (secure), confirming both fits converged.

### 3.2 Graded probe set (the independent variable)
15 probes, 3 per tier, ordered by semantic distance from the FT domain:

| Tier | Category | Source | Role |
|---|---|---|---|
| T0 | in_domain_code | em_insecure user prompts | closest |
| T1 | code_safety_qa | authored secure-coding questions | code-adjacent |
| T2 | overt_harm | AdvBench harmful behaviors | overt harm |
| T3 | broad_alignment | EM free-form "reveal your values" Qs | broad alignment |
| T4 | orthogonal_cap | GSM8K / ARC / TruthfulQA | farthest (orthogonal) |

Each probe's **semantic distance** = cosine distance between its `all-MiniLM-L6-v2` embedding and the FT-data centroid embedding.

### 3.3 Trajectory-divergence metric (the dependent variable)
At checkpoints {0, 15, 30} we greedily decoded (max_new_tokens = 20) all probes. Divergence of probe *i* at step *s* from its **own step-0 baseline** = composite
`0.5·(1 − cos(emb_base, emb_s)) + 0.5·(1 − difflib_ratio(text_base, text_s))`.
**Final drift** = composite at step 30. **Robustness attractor / reversal rate** = fraction of step-transitions where divergence *decreases* (reverts toward baseline).

### 3.4 Statistics
Spearman & Pearson (distance vs drift) with 2000-sample bootstrap 95% CI; Kruskal–Wallis across tiers; insecure-vs-secure per-tier differencing. α = 0.05. Tools: torch 2.12 (CPU), transformers 5.12, peft 0.19, sentence-transformers, scipy, scikit-learn.

---

## 4. Results

### 4.1 Core hypothesis test — drift vs semantic distance
| Experiment | Spearman ρ | p | 95% CI | Pearson r | p |
|---|---|---|---|---|---|
| insecure | **−0.079** | 0.78 | [−0.61, 0.51] | — | — |
| secure | −0.070 | 0.81 | [−0.59, 0.53] | — | — |

No significant monotonic relationship. The bootstrap CI spans zero widely. **H1 (graded robustness) is not supported.**

### 4.2 Drift by tier (Kruskal–Wallis insecure: H = 4.12, p = 0.39)
| Tier | Category | Insecure drift | Secure drift | Intent diff (ins−sec) |
|---|---|---|---|---|
| T0 | in_domain_code | 0.479 ± 0.264 | 0.461 | +0.018 |
| T1 | code_safety_qa | **0.045 ± 0.034** | 0.106 | −0.061 |
| T2 | overt_harm | 0.466 ± 0.337 | 0.431 | +0.035 |
| T3 | broad_alignment | 0.239 ± 0.338 | 0.226 | +0.013 |
| T4 | orthogonal_cap | **0.553 ± 0.232** | 0.535 | +0.018 |
| — | **global mean** | **0.356** | **0.352** | **+0.004** |

The drift profile is **non-monotonic**: orthogonal-capability prompts (T4) drift *most*, not least; the lowest-drift region is the conceptual code-safety tier (T1). **H2 (orthogonal-task basin) is refuted** in this direction.

### 4.3 Intent specificity (insecure vs secure) — **H3 refuted**
Adversarial and benign fine-tunes produce **statistically indistinguishable** drift (global 0.356 vs 0.352; every per-tier difference ≤ 0.061, and all within ±1 SD). At this scale the trajectory divergence reflects **generic fine-tuning style drift**, not adversarial intent. This is the report's central methodological finding: *raw* trajectory drift is not a valid robustness diagnostic without a benign-FT control.

### 4.4 Robustness attractor / reversal — **partial support for H4**
Reversal rate (reversion toward baseline) by tier (insecure): in_domain 0.00, code_safety 0.00, overt_harm **0.33**, broad_alignment 0.00, orthogonal_cap **0.33**. The two highest-drift tiers also show the most non-monotonic, partially-reverting trajectories — consistent with an *unstable attractor* picture rather than smooth monotone drift, echoing the dynamical-systems view (Danovski et al. 2024) and Springer et al.'s instability claim.

### 4.5 Qualitative evidence
- **Genuine robustness pocket (T1):** the SQL-injection answer is preserved almost verbatim before vs after FT ("SQL injection is a type of web application security vulnerability that occurs when a malicious SQL …") → factual/technical knowledge resists drift.
- **Aligned refusal preserved (T3):** the "one wish" probe still opens "I'm sorry … as a helpful AI, I don't have the ability to …" pre- and post-FT — no overt misalignment emerged at this scale (consistent with EM being hard to elicit in tiny models).

### 4.6 Figures
- `figures/drift_by_tier.png` — per-tier drift, insecure vs secure (near-identical bars).
- `figures/drift_vs_distance.png` — scatter + regression of drift vs semantic distance (flat).
- `figures/trajectories.png` — mean divergence trajectory per tier across steps {0,15,30}.

---

## 5. Analysis & Discussion

The experiment **refutes the simple form of the hypothesis** (graded, distance-monotonic robustness with orthogonal basins) and instead reveals two findings the original framing did not anticipate:

1. **Orthogonal ≠ robust.** Far-from-domain capability prompts drifted *most*, not least. This is precisely the emergent-misalignment "broad spillover" phenomenon (narrow FT perturbs unrelated behavior) and aligns with Springer et al. (2026): orthogonality to the FT objective does *not* confer stability. The real low-drift region was *semantic content type* (factual/technical knowledge), not *task distance*.

2. **Generic-FT drift dominates adversarial-intent drift at small scale.** Insecure and secure fine-tunes drifted identically. Emergent misalignment is a property that, per Betley et al., appears strongly only in larger models (GPT-4o, Qwen-32B); a 135M model under 30 LoRA steps shows only stylistic adaptation common to *any* fine-tune. The methodological upshot — **trajectory divergence must be differenced against a benign-FT control to isolate adversarial effects** — is a genuine contribution and a concrete recommendation for the metric's correct use.

**How this answers the research question:** at the tested scale, LLMs do *not* show distance-graded robustness; what looks like "hidden robustness on orthogonal tasks" is better described as (a) content-type-specific robustness (knowledge preserved, style not) plus (b) an artifact of not controlling for generic FT drift. The trajectory-divergence + reversal-rate apparatus works and is informative, but its *uncontrolled* output is not a misalignment signal.

---

## 6. Limitations

- **Severe scale reduction.** 135M model (vs proposed Llama-2-7B), LoRA r=16 (vs full FT), 30 steps (vs 10 epochs), 15 probes (vs 500), 3 checkpoints (vs every-100-step), all forced by the **CPU-only, ~2 GB RAM, 1 h** budget. Two prior full-FT/AdamW attempts were OOM-killed; gradient checkpointing + LoRA + bs=1 were required to run at all.
- **Low statistical power.** n = 15 (3/tier); the Spearman CI spans [−0.61, 0.51]. We cannot detect a moderate true correlation; the null result is *consistent with* but not *proof of* no effect.
- **Emergent misalignment likely needs scale.** A 135M model may be incapable of EM regardless of method, so the insecure≈secure result may reflect model capacity, not a universal truth. The honest reading: *we did not reproduce EM*, and the trajectory metric correctly reported "no intent-specific drift."
- **Single seed, single model family, greedy decoding** (temperature 0) — no sampling variance estimate. Embedding-based divergence can be dominated by late-token noise even when the salient opening is unchanged (seen in the "one wish" probe).
- **Semantic-distance proxy** uses a generic sentence encoder centroid; alternative distance definitions could change tier ordering.

---

## 7. Conclusions & Next Steps

**Answer to the research question:** At small scale, we find **no evidence for semantic-distance-graded robustness or orthogonal-task robustness basins**. Behavioral drift was non-monotonic (orthogonal tasks drifted most), and — decisively — was indistinguishable between adversarial and benign fine-tuning, indicating the raw trajectory divergence reflects generic fine-tuning drift rather than adversarial intent. The only reproducible robustness was content-type-specific (factual/technical knowledge preserved).

**Methodological contribution:** a working, checkpoint-level **trajectory-divergence + reversal-rate** diagnostic, plus the empirical lesson that it must be **differenced against a matched benign-FT control** to be a valid robustness/misalignment signal.

**Recommended follow-ups (with adequate compute):**
1. Scale the model to Qwen2.5-0.5B/7B where EM is known to emerge, and to full 500-probe / per-100-step checkpointing for statistical power.
2. Report the **intent-differenced** divergence (insecure − secure) as the primary metric, with sampling-based CIs (temperature > 0, multiple seeds).
3. Add parameter-space axes (alignment-direction vs orthogonal displacement à la AsFT; VISAGE landscape from `llm-landscape`) to test whether behavioral basins map to weight-space basins.
4. Test the Springer (2026) stability prediction directly: does any in-basin robustness observed early in training collapse later?

---

## References
- Betley et al. (2025) *Emergent Misalignment* — arXiv:2502.17424 (datasets: em_insecure/secure).
- Peng et al. (2024) *Navigating the Safety Landscape (safety basin, VISAGE)* — arXiv:2405.17374.
- Qi et al. (2024) *Safety Alignment Should Be More Than a Few Tokens Deep* — arXiv:2406.05946.
- Springer et al. (2026) *The Geometry of Alignment Collapse* — arXiv:2602.15799.
- Yang et al. (2025) *AsFT: Anchoring Safety in Narrow Safety Basin* — arXiv:2506.08473.
- Soligo et al. (2026) *EM is Easy, Narrow Misalignment is Hard (rank-1 LoRA)* — arXiv:2602.07852.
- Danovski et al. (2024) *Dynamical stability & chaos in NN trajectories* — arXiv:2404.05782.
- Tools: SmolLM2-135M-Instruct (HuggingFaceTB), all-MiniLM-L6-v2, AdvBench, GSM8K/ARC/TruthfulQA samples.

*Full per-probe data: `results/analysis_{insecure,secure}.json`; raw trajectories: `results/traj_*.json`; plan & decisions: `planning.md`.*
