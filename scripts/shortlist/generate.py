"""Send prompts to siphon-serve one at a time and store prompt and output token ids.

usage: generate.py <url> <prompts.jsonl> <out.jsonl> [--seconds N] [--max-tokens N]

The prompts are records with `prompt_id`, `category` and chat `messages`, as
`scripts/shortlist/prompts.py` writes. Reasoning efforts rotate through
none, low and medium by prompt index. Each output record holds the prompt's and the
completion's token ids (the server's `return_token_ids`). Prompts already in the output
are skipped, so a run resumes.
"""
import argparse, json, time, urllib.request

EFFORTS = ["none", "low", "medium"]
ap = argparse.ArgumentParser()
ap.add_argument("url"); ap.add_argument("prompts"); ap.add_argument("out")
ap.add_argument("--seconds", type=float, default=3600)
ap.add_argument("--max-tokens", type=int, default=1024)
a = ap.parse_args()
done = set()
try:
    for l in open(a.out):
        done.add(json.loads(l)["prompt_id"])
except FileNotFoundError:
    pass
start = time.monotonic(); n = toks = 0
with open(a.out, "a") as out:
    for i, l in enumerate(open(a.prompts)):
        if time.monotonic() - start > a.seconds:
            break
        p = json.loads(l)
        if p["prompt_id"] in done:
            continue
        effort = EFFORTS[i % 3]
        body = dict(messages=p["messages"], max_tokens=a.max_tokens, reasoning_effort=effort,
                    return_token_ids=True)
        req = urllib.request.Request(a.url + "/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        t = time.monotonic()
        try:
            r = json.load(urllib.request.urlopen(req, timeout=600))
        except Exception as e:  # noqa: BLE001
            print("failed", p["prompt_id"], e, flush=True); continue
        c = r["choices"][0]
        rec = dict(prompt_id=p["prompt_id"], category=p["category"], effort=effort,
                   prompt_ids=r["prompt_token_ids"], output_ids=c["token_ids"],
                   finish=c["finish_reason"], seconds=round(time.monotonic() - t, 3))
        out.write(json.dumps(rec) + "\n"); out.flush()
        n += 1; toks += len(c["token_ids"])
        if n % 50 == 0:
            el = time.monotonic() - start
            print(f"{n} prompts, {toks} tokens, {toks / el:.0f} tok/s", flush=True)
print(f"done: {n} prompts, {toks} tokens in {time.monotonic() - start:.0f} s", flush=True)
