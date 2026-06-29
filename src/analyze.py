"""Analyze trajectory-divergence data produced by run.py.

For each experiment (tag), computes per-probe trajectory divergence from the
checkpoint-0 baseline using a composite of (a) embedding cosine distance and
(b) normalized character edit distance. Also computes each probe's semantic
distance from the FT-data centroid, then tests the core hypothesis:
  drift decreases with semantic distance (robustness basin).

Outputs: results/analysis_<tag>.json, results/summary.json, figures/*.png
"""
import json, sys, difflib
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
FIG = ROOT / "figures"; FIG.mkdir(exist_ok=True)
DATA = ROOT / "datasets"
TIER_NAMES = {0: "in_domain_code", 1: "code_safety_qa", 2: "overt_harm",
              3: "broad_alignment", 4: "orthogonal_cap"}

emb_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def embed(texts):
    return emb_model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

def norm_edit(a, b):
    """1 - normalized similarity ratio (0 identical, ~1 totally different)."""
    if not a and not b:
        return 0.0
    return 1.0 - difflib.SequenceMatcher(None, a, b).ratio()

def ft_centroid(train_name, n=200):
    rows = []
    with open(DATA / "adversarial_finetuning" / f"{train_name}.jsonl") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    texts = []
    for ex in rows[:n]:
        # concatenate user+assistant content as the FT-domain representation
        parts = [m["content"] for m in ex["messages"]]
        texts.append(" ".join(parts)[:1000])
    V = embed(texts)
    c = V.mean(axis=0)
    c = c / (np.linalg.norm(c) + 1e-9)
    return c

def analyze(tag):
    d = json.load(open(RES / f"traj_{tag}.json"))
    probes = d["probes"]
    traj = {int(k): v for k, v in d["trajectory"].items()}
    steps = sorted(traj.keys())
    base_step = steps[0]
    centroid = ft_centroid(d["train"])

    # semantic distance of each probe prompt from FT centroid
    P = embed([p["prompt"] for p in probes])
    sem_dist = 1.0 - P.dot(centroid)  # cosine distance

    # precompute embeddings of all responses
    n = len(probes)
    # divergence vs baseline at each step
    div_cos = {s: np.zeros(n) for s in steps}
    div_edit = {s: np.zeros(n) for s in steps}
    base_resp = traj[base_step]
    base_emb = embed(base_resp)
    for s in steps:
        re = embed(traj[s])
        for i in range(n):
            div_cos[s][i] = 1.0 - float(np.dot(base_emb[i], re[i]))
            div_edit[s][i] = norm_edit(base_resp[i], traj[s][i])

    composite = {s: 0.5 * div_cos[s] + 0.5 * div_edit[s] for s in steps}
    final = steps[-1]
    final_drift = composite[final]

    # robustness attractor score: reversal rate = fraction of step-transitions
    # where composite divergence DECREASES (output reverts toward baseline)
    reversal = np.zeros(n)
    cs = np.stack([composite[s] for s in steps], axis=1)  # n x T
    diffs = np.diff(cs, axis=1)
    # consider transitions after first (baseline->c1 always increase usually)
    for i in range(n):
        nonzero = diffs[i]
        if len(nonzero) > 0:
            reversal[i] = float(np.mean(nonzero < 0))

    rows = []
    for i, p in enumerate(probes):
        rows.append({
            "id": p["id"], "tier": p["tier"], "category": p["category"],
            "prompt": p["prompt"][:120],
            "sem_dist": float(sem_dist[i]),
            "final_drift": float(final_drift[i]),
            "final_cos": float(div_cos[final][i]),
            "final_edit": float(div_edit[final][i]),
            "reversal_rate": float(reversal[i]),
            "traj_composite": [float(composite[s][i]) for s in steps],
        })

    # --- correlation: semantic distance vs final drift (H1) ---
    sd = np.array([r["sem_dist"] for r in rows])
    fd = np.array([r["final_drift"] for r in rows])
    rho, p_rho = stats.spearmanr(sd, fd)
    r_pear, p_pear = stats.pearsonr(sd, fd)
    # bootstrap CI for spearman
    rng = np.random.default_rng(42)
    boot = []
    for _ in range(2000):
        idx = rng.integers(0, len(sd), len(sd))
        if len(np.unique(sd[idx])) > 2:
            boot.append(stats.spearmanr(sd[idx], fd[idx]).statistic)
    ci = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))] if boot else [None, None]

    # --- per-tier drift (H2) ---
    tier_drift = {}
    for t in sorted(TIER_NAMES):
        vals = [r["final_drift"] for r in rows if r["tier"] == t]
        tier_drift[t] = {"name": TIER_NAMES[t], "mean": float(np.mean(vals)),
                          "std": float(np.std(vals)), "n": len(vals),
                          "vals": [float(v) for v in vals]}
    # Kruskal-Wallis across tiers
    groups = [[r["final_drift"] for r in rows if r["tier"] == t] for t in sorted(TIER_NAMES)]
    kw_H, kw_p = stats.kruskal(*groups)

    out = {
        "tag": tag, "train": d["train"], "steps": steps,
        "spearman_rho": float(rho), "spearman_p": float(p_rho),
        "spearman_ci95": ci,
        "pearson_r": float(r_pear), "pearson_p": float(p_pear),
        "kruskal_H": float(kw_H), "kruskal_p": float(kw_p),
        "tier_drift": tier_drift,
        "rows": rows,
        "mean_reversal_by_tier": {TIER_NAMES[t]: float(np.mean([r["reversal_rate"] for r in rows if r["tier"]==t])) for t in sorted(TIER_NAMES)},
        "wall_time_s": d.get("wall_time_s"),
    }
    json.dump(out, open(RES / f"analysis_{tag}.json", "w"), indent=2)
    print(f"[{tag}] Spearman(sem_dist, drift) rho={rho:.3f} p={p_rho:.4f} CI95={ci}")
    print(f"[{tag}] Kruskal across tiers H={kw_H:.2f} p={kw_p:.4f}")
    for t in sorted(TIER_NAMES):
        td = tier_drift[t]
        print(f"   T{t} {td['name']:<16} drift={td['mean']:.3f}±{td['std']:.3f} (n={td['n']})")
    return out

def make_figures(analyses):
    tags = list(analyses.keys())
    # Fig 1: per-tier drift bar
    fig, ax = plt.subplots(figsize=(8, 5))
    tiers = sorted(TIER_NAMES)
    width = 0.8 / len(tags)
    for j, tag in enumerate(tags):
        a = analyses[tag]
        means = [a["tier_drift"][str(t)]["mean"] if str(t) in a["tier_drift"] else a["tier_drift"][t]["mean"] for t in tiers]
        stds = [a["tier_drift"][str(t)]["std"] if str(t) in a["tier_drift"] else a["tier_drift"][t]["std"] for t in tiers]
        x = np.arange(len(tiers)) + j * width
        ax.bar(x, means, width, yerr=stds, capsize=3, label=tag)
    ax.set_xticks(np.arange(len(tiers)) + width*(len(tags)-1)/2)
    ax.set_xticklabels([f"T{t}\n{TIER_NAMES[t]}" for t in tiers], fontsize=8)
    ax.set_ylabel("Final trajectory divergence (composite)")
    ax.set_title("Behavioral drift by semantic-distance tier\n(T0=in-domain ... T4=orthogonal)")
    ax.legend()
    plt.tight_layout(); plt.savefig(FIG / "drift_by_tier.png", dpi=120); plt.close()

    # Fig 2: scatter sem_dist vs final drift with regression
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.viridis(np.linspace(0, 1, 5))
    for tag in tags:
        a = analyses[tag]
        sd = np.array([r["sem_dist"] for r in a["rows"]])
        fd = np.array([r["final_drift"] for r in a["rows"]])
        tiersarr = np.array([r["tier"] for r in a["rows"]])
        for t in range(5):
            m = tiersarr == t
            ax.scatter(sd[m], fd[m], color=colors[t], label=f"T{t} {TIER_NAMES[t]}" if tag==tags[0] else None, alpha=0.8, edgecolor='k', linewidth=0.3)
        # regression line
        z = np.polyfit(sd, fd, 1); xs = np.linspace(sd.min(), sd.max(), 50)
        ax.plot(xs, np.polyval(z, xs), '--', label=f"{tag} fit (rho={a['spearman_rho']:.2f})")
    ax.set_xlabel("Semantic distance from FT domain (cosine)")
    ax.set_ylabel("Final trajectory divergence")
    ax.set_title("Hypothesis test: drift vs semantic distance")
    ax.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(FIG / "drift_vs_distance.png", dpi=120); plt.close()

    # Fig 3: trajectory curves (mean per tier) for first tag
    a = analyses[tags[0]]
    steps = a["steps"]
    fig, ax = plt.subplots(figsize=(8, 5))
    for t in range(5):
        curves = [r["traj_composite"] for r in a["rows"] if r["tier"] == t]
        mean_curve = np.mean(curves, axis=0)
        ax.plot(steps, mean_curve, marker='o', color=colors[t], label=f"T{t} {TIER_NAMES[t]}")
    ax.set_xlabel("Fine-tuning step"); ax.set_ylabel("Divergence from baseline")
    ax.set_title(f"Mean trajectory divergence per tier ({tags[0]})")
    ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(FIG / "trajectories.png", dpi=120); plt.close()
    print("Figures saved to", FIG)

if __name__ == "__main__":
    tags = sys.argv[1:] if len(sys.argv) > 1 else ["insecure"]
    analyses = {}
    for tag in tags:
        if (RES / f"traj_{tag}.json").exists():
            analyses[tag] = analyze(tag)
    make_figures(analyses)
    json.dump({t: {k: v for k, v in a.items() if k != "rows"} for t, a in analyses.items()},
              open(RES / "summary.json", "w"), indent=2)
    print("Saved summary.json")
