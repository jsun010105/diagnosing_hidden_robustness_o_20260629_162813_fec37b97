import arxiv, json, time, sys
queries = [
    "emergent misalignment narrow finetuning insecure code",
    "navigating safety landscape safety basin language models",
    "assessing brittleness safety alignment pruning low-rank",
    "safety alignment should be more than few tokens deep",
    "representation engineering activation steering safety LLM",
    "task vectors model arithmetic editing models",
    "LoRA low-rank adaptation safety degradation fine-tuning",
    "measuring behavioral drift model outputs distribution shift",
]
client = arxiv.Client(page_size=15, delay_seconds=3, num_retries=3)
existing = {p['id'] for p in json.load(open('arxiv_candidates.json'))}
new = []
for q in queries:
    try:
        s = arxiv.Search(query=q, max_results=8, sort_by=arxiv.SortCriterion.Relevance)
        for r in client.results(s):
            sid = r.get_short_id().split('v')[0]
            if sid not in existing:
                existing.add(sid)
                new.append({"id": sid,"title": r.title,"year": r.published.year,
                    "authors":[a.name for a in r.authors[:6]],
                    "abstract": r.summary.replace('\n',' '),
                    "pdf_url": r.pdf_url,"categories": r.categories,"query": q})
        print(f"[done] {q}", file=sys.stderr)
    except Exception as e:
        print(f"[err] {q}: {e}", file=sys.stderr)
for n in new:
    print(f"{n['id']} ({n['year']}) :: {n['title']}")
json.dump(new, open('arxiv_new.json','w'), indent=2)
