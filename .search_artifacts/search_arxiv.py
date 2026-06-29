import arxiv, json, sys, time

queries = [
    "adversarial fine-tuning safety alignment large language models",
    "fine-tuning attack jailbreak safety alignment LLM",
    "shallow safety alignment fine-tuning robustness",
    "emergent misalignment fine-tuning language models",
    "model editing weight space robustness alignment",
    "loss landscape fine-tuning language models flatness",
    "representation drift fine-tuning catastrophic forgetting LLM",
    "safety basin model weights robustness perturbation",
    "trajectory training dynamics neural network attractor",
    "alignment tax fine-tuning capability retention",
    "removing safety guardrails fine-tuning few examples",
    "linear mode connectivity fine-tuning language models",
]
client = arxiv.Client(page_size=20, delay_seconds=3, num_retries=3)
seen = {}
for q in queries:
    try:
        s = arxiv.Search(query=q, max_results=12, sort_by=arxiv.SortCriterion.Relevance)
        for r in client.results(s):
            sid = r.get_short_id().split('v')[0]
            if sid not in seen:
                seen[sid] = {
                    "id": sid, "title": r.title, "year": r.published.year,
                    "authors": [a.name for a in r.authors[:6]],
                    "abstract": r.summary.replace('\n',' '),
                    "pdf_url": r.pdf_url, "categories": r.categories,
                    "query": q,
                }
        print(f"[done] {q} -> total unique {len(seen)}", file=sys.stderr)
    except Exception as e:
        print(f"[err] {q}: {e}", file=sys.stderr)
    time.sleep(1)

with open("arxiv_candidates.json","w") as f:
    json.dump(list(seen.values()), f, indent=2)
print(f"TOTAL UNIQUE: {len(seen)}", file=sys.stderr)
