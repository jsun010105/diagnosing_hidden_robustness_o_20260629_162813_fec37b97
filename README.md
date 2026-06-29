# Diagnosing Hidden Robustness of LLMs to Adversarial Fine-Tuning via Trajectory Divergence Metrics

A CPU-only, checkpoint-level study of how a real instruct LLM's outputs drift from their pre-fine-tuning baseline during adversarial fine-tuning, measured per-prompt across a graded probe set ordered by semantic distance from the fine-tuning domain. Tests whether "hidden robustness basins" appear on tasks orthogonal to the adversarial objective.

## Key findings
- **No distance-graded robustness.** Behavioral drift did *not* decrease with semantic distance from the FT domain (Spearman ρ = −0.08, n.s., n = 15). Orthogonal-capability prompts drifted **most**, not least — refuting the "orthogonal = robust" prediction and echoing emergent-misalignment spillover + Springer et al. (2026).
- **Adversarial ≈ benign.** Fine-tuning on `insecure` vs `secure` code produced **near-identical** drift (global mean 0.356 vs 0.352; intent-specific per-tier diff ≤ 0.035). At 135M scale the divergence is **generic fine-tuning style drift, not adversarial intent** → emergent misalignment was *not* reproduced.
- **Methodological lesson.** Raw trajectory divergence overstates misalignment; it must be **differenced against a matched benign-FT control** to be a valid robustness signal.
- **Real robustness pocket = knowledge, not task-distance.** Factual/technical answers (e.g. "what is SQL injection") were preserved almost verbatim; aligned refusals also persisted.
- **Unstable attractors.** The highest-drift tiers (overt-harm, orthogonal) showed the most non-monotonic, partially-reverting trajectories (reversal rate 0.33).

## Reproduce
```bash
uv venv && source .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install transformers sentence-transformers peft scikit-learn scipy numpy pandas matplotlib

python src/build_probes.py                      # build graded probe set -> results/probes.json
python src/run.py --train em_insecure       --tag insecure --steps 30 --bs 1 --max_new 20 \
    --ntrain 60 --max_probes 15 --ckpt_fracs 0,0.5,1.0
python src/run.py --train em_secure_control --tag secure   --steps 30 --bs 1 --max_new 20 \
    --ntrain 60 --max_probes 15 --ckpt_fracs 0,0.5,1.0
python src/analyze.py insecure secure           # stats + figures
```
Runtime: ~3–4 min per fine-tune + ~1 min analysis on 8 CPU threads. Seed = 42. Needs ~2 GB RAM (LoRA + gradient checkpointing + batch size 1 are required to fit).

## File structure
```
planning.md                 Phase 0/1: motivation, hypothesis decomposition, design
REPORT.md                   Full research report (primary deliverable)
src/build_probes.py         Graded 5-tier probe set construction
src/run.py                  LoRA fine-tune + checkpointed trajectory generation
src/analyze.py              Divergence metrics, stats, figures
results/traj_*.json         Raw per-checkpoint probe responses
results/analysis_*.json     Per-probe drift, semantic distance, correlations
results/summary.json        Headline stats
figures/*.png               drift_by_tier, drift_vs_distance, trajectories
datasets/ code/ papers/     Pre-gathered resources
```

See **REPORT.md** for full methodology, results, statistics, and limitations.
