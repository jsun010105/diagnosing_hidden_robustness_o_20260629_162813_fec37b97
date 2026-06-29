"""Fine-tune SmolLM2-135M-Instruct on an adversarial (or control) dataset,
checkpointing along the way, and at each checkpoint generate greedy responses
to the graded probe set. Saves trajectories to results/traj_<tag>.json.

Usage: python src/run.py --train em_insecure --tag insecure --steps 120
"""
import json, argparse, time, random, os
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "datasets"
RES = ROOT / "results"
RES.mkdir(exist_ok=True)
MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
SEED = 42

def set_seed(s=SEED):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)

def load_jsonl(p):
    out = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out

def _tok_ids(tok, msgs, add_gen):
    """Return a flat python list of token ids for a chat (transformers 5.x safe)."""
    enc = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=add_gen,
                                  return_dict=True)
    ids = enc["input_ids"]
    # may be nested [[...]] depending on version
    if len(ids) and isinstance(ids[0], (list, tuple)):
        ids = ids[0]
    return list(ids)

def build_train_examples(name, tok, n=256, max_len=448):
    """Return list of (input_ids, labels) with loss masked to assistant tokens."""
    rows = load_jsonl(DATA / "adversarial_finetuning" / f"{name}.jsonl")
    random.shuffle(rows)
    examples = []
    for ex in rows:
        msgs = ex["messages"]
        if not any(m["role"] == "assistant" for m in msgs):
            continue
        a_idx = max(i for i, m in enumerate(msgs) if m["role"] == "assistant")
        prompt_msgs = msgs[:a_idx]
        full_ids = _tok_ids(tok, msgs, add_gen=False)
        prompt_ids = _tok_ids(tok, prompt_msgs, add_gen=True)
        if len(prompt_ids) >= max_len - 8:
            continue  # no room for assistant supervision
        if len(full_ids) > max_len:
            full_ids = full_ids[:max_len]  # truncate; early assistant tokens kept
        labels = list(full_ids)
        for i in range(min(len(prompt_ids), len(labels))):
            labels[i] = -100
        if all(l == -100 for l in labels):
            continue
        examples.append((full_ids, labels))
        if len(examples) >= n:
            break
    return examples

def collate(batch, pad_id):
    maxlen = max(len(x[0]) for x in batch)
    input_ids, labels, attn = [], [], []
    for ids, lab in batch:
        pad = maxlen - len(ids)
        input_ids.append(ids + [pad_id]*pad)
        labels.append(lab + [-100]*pad)
        attn.append([1]*len(ids) + [0]*pad)
    return (torch.tensor(input_ids), torch.tensor(labels), torch.tensor(attn))

@torch.no_grad()
def generate_all(model, tok, probes, max_new_tokens=40):
    model.eval()
    prev_cache = model.config.use_cache
    model.config.use_cache = True  # fast decoding for inference
    resps = []
    for p in probes:
        msgs = [{"role": "user", "content": p["prompt"]}]
        enc = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                      return_tensors="pt", return_dict=True)
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.eos_token_id)
        text = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        resps.append(text)
    model.config.use_cache = prev_cache
    return resps

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--ntrain", type=int, default=256)
    ap.add_argument("--max_new", type=int, default=40)
    ap.add_argument("--max_probes", type=int, default=0, help="subsample to N probes (balanced by tier); 0=all")
    ap.add_argument("--ckpt_fracs", type=str, default="0,0.5,1.0")
    args = ap.parse_args()

    set_seed()
    torch.set_num_threads(8)
    t0 = time.time()
    probes = json.load(open(RES / "probes.json"))
    if args.max_probes and args.max_probes < len(probes):
        per = max(1, args.max_probes // 5)
        sub = []
        for t in range(5):
            sub += [p for p in probes if p["tier"] == t][:per]
        probes = sub
        print(f"[{args.tag}] subsampled to {len(probes)} probes ({per}/tier)")
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
    # LoRA: low-memory adversarial FT (matches EM/Soligo setup; required by 2GB RAM cap)
    lcfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0,
                      target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
                      task_type="CAUSAL_LM")
    model = get_peft_model(model, lcfg)
    # gradient checkpointing: trade compute for memory (required by 2GB RAM cap)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False
    n_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{args.tag}] LoRA trainable params: {n_train_params/1e6:.3f}M")

    examples = build_train_examples(args.train, tok, n=args.ntrain, max_len=256)
    print(f"[{args.tag}] train examples: {len(examples)}  (load {time.time()-t0:.1f}s)")

    # checkpoints at which to snapshot probe responses (incl. step 0 baseline)
    fracs = [float(x) for x in args.ckpt_fracs.split(",")]
    ckpts = sorted(set(int(round(fr * args.steps)) for fr in fracs))
    print(f"[{args.tag}] checkpoints:", ckpts)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    pad_id = tok.pad_token_id

    trajectory = {}  # step -> list[str] responses
    losses = []

    def snapshot(step):
        ts = time.time()
        resps = generate_all(model, tok, probes, args.max_new)
        trajectory[step] = resps
        print(f"[{args.tag}] snapshot @step {step}  ({time.time()-ts:.1f}s gen)")

    snapshot(0)  # baseline

    model.train()
    step = 0
    ei = 0
    order = list(range(len(examples)))
    while step < args.steps:
        random.shuffle(order)
        for i in range(0, len(order), args.bs):
            if step >= args.steps:
                break
            batch = [examples[j] for j in order[i:i+args.bs]]
            input_ids, labels, attn = collate(batch, pad_id)
            out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); opt.zero_grad()
            step += 1
            losses.append((step, float(loss.item())))
            if step in ckpts:
                snapshot(step)
                model.train()
            if step % 10 == 0:
                print(f"[{args.tag}] step {step}/{args.steps} loss {loss.item():.3f}  elapsed {time.time()-t0:.0f}s")

    out = {
        "tag": args.tag, "train": args.train, "steps": args.steps,
        "lr": args.lr, "bs": args.bs, "n_train": len(examples),
        "checkpoints": sorted(trajectory.keys()),
        "probes": probes,
        "trajectory": trajectory,
        "losses": losses,
        "seed": SEED, "model": MODEL,
        "wall_time_s": round(time.time()-t0, 1),
    }
    fp = RES / f"traj_{args.tag}.json"
    json.dump(out, open(fp, "w"), indent=2)
    print(f"[{args.tag}] DONE in {time.time()-t0:.0f}s -> {fp}")

if __name__ == "__main__":
    main()
