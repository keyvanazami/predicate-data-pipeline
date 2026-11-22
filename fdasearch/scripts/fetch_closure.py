"""Download summary PDFs for cited predicates that are missing from the corpus (the team's 'closure fetch' step).
Writes to data/pdfs_closure/ and data/pdfs_closure/_manifest.tsv. Candidates = K-numbers cited as predicates by any
query that have no usable text in data/corpus.parquet."""
import os, re, sys, time, collections, concurrent.futures as cf
import pandas as pd, requests
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from download_pdfs import pdf_url, UA

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "pdfs_closure"); os.makedirs(OUT, exist_ok=True)
MAN = os.path.join(OUT, "_manifest.tsv")


def main():
    q = pd.read_parquet(os.path.join(ROOT, "data", "queries.parquet"))
    c = pd.read_parquet(os.path.join(ROOT, "data", "corpus.parquet"))
    have = set(c.k)
    want = sorted({p for ps in q.predicates_all for p in ps if p.startswith("K") and p not in have})
    done = {}
    if os.path.exists(MAN):
        for line in open(MAN):
            k, st = line.rstrip("\n").split("\t")[:2]; done[k] = st
    todo = [k for k in want if k not in done]
    print(f"{len(want)} missing predicates, {len(todo)} to fetch", flush=True)
    sess = requests.Session(); sess.headers["User-Agent"] = UA

    def fetch(k):
        src = os.path.join(ROOT, "data", "pdfs", k + ".pdf")
        dst = os.path.join(OUT, k + ".pdf")
        if os.path.exists(src):
            import shutil; shutil.copy(src, dst); return k, "ok-local"
        for attempt in range(3):
            try:
                r = sess.get(pdf_url(k), timeout=60)
                if r.status_code == 404: return k, "404"
                if r.status_code == 200 and r.content.startswith(b"%PDF"):
                    open(dst, "wb").write(r.content); return k, "ok"
                if b"abuse" in r.content[:4000].lower(): time.sleep(30)
            except Exception as e:
                time.sleep(2)
        return k, "err"

    st = collections.Counter(done.values())
    with open(MAN, "a") as man, cf.ThreadPoolExecutor(1) as ex:
        for i, (k, s) in enumerate(ex.map(fetch, todo), 1):
            man.write(f"{k}\t{s}\n"); man.flush(); st[s] += 1
            if i % 200 == 0: print(i, dict(st), flush=True)
    print("done", dict(st))


if __name__ == "__main__":
    main()
