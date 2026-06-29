import json, os, time, urllib.request

# Curated selection: id -> (short_filename_slug, relevance_note)
selection = {
    "2310.03693": "qi2023_finetuning_compromises_safety",
    "2502.17424": "betley2025_emergent_misalignment",
    "2405.17374": "peng2024_navigating_safety_landscape_basin",
    "2406.05946": "qi2024_safety_alignment_few_tokens_deep",
    "2402.05162": "wei2024_brittleness_safety_pruning_lowrank",
    "2506.08473": "asft2025_anchoring_narrow_safety_basin",
    "2602.15799": "geometry_alignment_collapse_2026",
    "2602.08239": "linearization_explains_finetuning_2026",
    "2404.05782": "dynamical_stability_chaos_nn_trajectories",
    "1712.09913": "li2017_visualizing_loss_landscape",
    "2212.04089": "ilharco2022_editing_models_task_arithmetic",
    "2308.08747": "luo2023_catastrophic_forgetting_llm_continual",
    "2409.02228": "unforgettable_generalization_lms",
    "2506.05346": "guardrails_collapse_similarity_analysis",
    "2507.03662": "re_emergent_misalignment_narrow_finetuning",
    "2602.07852": "emergent_misalignment_easy_narrow_hard",
    "2510.02833": "attack_via_overfitting_10shot",
    "2506.17209": "finetuning_lowers_safety_eval_consistency",
    "2601.10141": "understanding_preserving_safety_finetuned",
    "2511.14017": "narrow_unlearning_to_emergent_misalignment",
}

cand = {p['id']: p for p in json.load(open('arxiv_candidates.json'))}
cand.update({p['id']: p for p in json.load(open('arxiv_new.json'))})

meta = []
for aid, slug in selection.items():
    p = cand.get(aid)
    if not p:
        print(f"MISSING META {aid}"); continue
    fn = f"papers/{aid}_{slug}.pdf"
    if os.path.exists(fn) and os.path.getsize(fn) > 10000:
        print(f"skip {fn}"); meta.append((aid,slug,p)); continue
    url = f"https://arxiv.org/pdf/{aid}.pdf"
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={'User-Agent':'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=60) as r, open(fn,'wb') as out:
                out.write(r.read())
            sz = os.path.getsize(fn)
            print(f"OK {fn} ({sz//1024} KB)")
            meta.append((aid,slug,p))
            break
        except Exception as e:
            print(f"retry {aid} ({attempt}): {e}"); time.sleep(4)
    time.sleep(2)

json.dump([{"id":a,"slug":s,**p} for a,s,p in meta], open('selected_papers.json','w'), indent=2)
print(f"\nDOWNLOADED METADATA FOR {len(meta)} papers")
