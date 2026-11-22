"""Ask a local Ollama LLM to list predicate / reference K-numbers from a 510(k) summary; compare with regex labels."""
import json, sys, re, urllib.request, time
inp, outp, model = sys.argv[1], sys.argv[2], sys.argv[3]
PROMPT = """You are reading the text of an FDA 510(k) summary. List the 510(k) numbers (format K followed by 6 digits) of
(a) the PREDICATE devices the submitter claims substantial equivalence to, and
(b) any REFERENCE devices (cited for comparison or testing but not as predicates).
Ignore the subject device's own number {k}. If a device is described as both, call it predicate.
Answer with JSON only: {{"predicates": [...], "references": [...]}}

SUMMARY TEXT:
{text}
"""
done = set()
try:
    for l in open(outp): done.add(json.loads(l)["k"])
except FileNotFoundError:
    pass
out = open(outp, "a")
for line in open(inp, encoding="utf-8"):
    r = json.loads(line)
    if r["k"] in done: continue
    body = json.dumps({"model": model, "prompt": PROMPT.format(k=r["k"], text=r["text"]), "stream": False, "format": "json", "options": {"temperature": 0, "num_ctx": 8192}}).encode()
    t = time.time()
    try:
        resp = json.load(urllib.request.urlopen(urllib.request.Request("http://localhost:11434/api/generate", data=body, headers={"Content-Type": "application/json"}), timeout=600))
        ans = json.loads(resp["response"])
        preds = sorted({x.upper().replace(" ", "") for x in ans.get("predicates", []) if re.fullmatch(r"[Kk]\s?\d{6}", str(x))})
        refs = sorted({x.upper().replace(" ", "") for x in ans.get("references", []) if re.fullmatch(r"[Kk]\s?\d{6}", str(x))})
        r.update(llm_predicates=preds, llm_references=refs, seconds=round(time.time() - t, 1))
    except Exception as e:
        r.update(error=str(e)[:200])
    out.write(json.dumps({k: v for k, v in r.items() if k != "text"}) + "\n"); out.flush()
    print(r["k"], r.get("llm_predicates"), r.get("error", ""), flush=True)
