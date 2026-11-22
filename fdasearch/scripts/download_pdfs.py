"""Download 510(k) summary PDFs from accessdata.fda.gov, newest first.

Usage: python download_pdfs.py [--min-year 2019] [--workers 6]
Writes data/pdfs/K######.pdf and data/pdfs/_manifest.tsv (k, status, bytes).
Skips files already present or already recorded as 404.
"""
import argparse, json, os, sys, time, zipfile, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw", "device-510k.json.zip")
OUT = os.path.join(ROOT, "data", "pdfs")
MAN = os.path.join(OUT, "_manifest.tsv")
UA = "fdasearch-510k-research/0.1 (academic course project)"


def pdf_url(k):
    yy = int(k[1:3])
    folder = "pdf" if (yy >= 76 or yy == 0) else f"pdf{yy}"
    return f"https://www.accessdata.fda.gov/cdrh_docs/{folder}/{k}.pdf"


OPENFDA_URL = "https://download.open.fda.gov/device/510k/device-510k-0001-of-0001.json.zip"


def load_records():
    if not os.path.exists(RAW):
        os.makedirs(os.path.dirname(RAW), exist_ok=True)
        print("downloading openFDA 510(k) export ...", flush=True)
        r = requests.get(OPENFDA_URL, headers={"User-Agent": UA}, timeout=600)
        r.raise_for_status()
        open(RAW, "wb").write(r.content)
    z = zipfile.ZipFile(RAW)
    return json.load(z.open(z.namelist()[0]))["results"]


def panel_match(r, panel):
    """True if the record belongs to the advisory committee given as code (CV) or description substring."""
    if not panel:
        return True
    p = panel.lower()
    return r.get("advisory_committee", "").lower() == p or p in r.get("advisory_committee_description", "").lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-year", type=int, default=1976)
    ap.add_argument("--max-year", type=int, default=2100)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--include-blank", action="store_true", help="also try records with blank statement_or_summary")
    ap.add_argument("--panel", default="", help="only this advisory committee (code like CV, or description substring)")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    recs = load_records()
    done = set()
    if os.path.exists(MAN):
        for line in open(MAN):
            p = line.rstrip("\n").split("\t")
            if len(p) >= 2 and p[1] in ("200", "404"):
                done.add(p[0])
    todo = []
    for r in recs:
        k = r["k_number"]
        if not k.startswith("K"):
            continue
        s = r.get("statement_or_summary", "")
        if s != "Summary" and not (a.include_blank and s == ""):
            continue
        y = int(r["decision_date"][:4])
        if y < a.min_year or y > a.max_year or not panel_match(r, a.panel):
            continue
        if k in done or os.path.exists(os.path.join(OUT, k + ".pdf")):
            continue
        todo.append((r["decision_date"], k))
    todo.sort(reverse=True)
    print(f"{len(todo)} to download ({len(done)} already in manifest)", flush=True)
    lock = threading.Lock()
    man = open(MAN, "a")
    sess = requests.Session()
    sess.headers["User-Agent"] = UA

    def fetch(k):
        url = pdf_url(k)
        for attempt in range(4):
            try:
                r = sess.get(url, timeout=60, allow_redirects=True)
            except Exception as e:
                time.sleep(2 * (attempt + 1)); continue
            if "abuse-detection" in r.url:
                time.sleep(30 * (attempt + 1)); continue
            if r.status_code == 200 and r.content[:4] == b"%PDF":
                tmp = os.path.join(OUT, k + ".part")
                with open(tmp, "wb") as f:
                    f.write(r.content)
                os.replace(tmp, os.path.join(OUT, k + ".pdf"))
                return k, "200", len(r.content)
            if r.status_code == 404:
                return k, "404", 0
            time.sleep(2 * (attempt + 1))
        return k, "ERR", 0

    n_ok = n_404 = n_err = 0
    t0 = time.time()
    with ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(fetch, k) for _, k in todo]
        for i, f in enumerate(as_completed(futs), 1):
            k, st, nb = f.result()
            with lock:
                man.write(f"{k}\t{st}\t{nb}\n"); man.flush()
            if st == "200": n_ok += 1
            elif st == "404": n_404 += 1
            else: n_err += 1
            if i % 200 == 0:
                el = time.time() - t0
                print(f"{i}/{len(todo)} ok={n_ok} 404={n_404} err={n_err} {i/el:.1f}/s", flush=True)
    print(f"done ok={n_ok} 404={n_404} err={n_err} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
