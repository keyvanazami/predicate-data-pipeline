"""Parse-agreement check: run the team's predx parser (predx.sections + predx.predicates) over PyMuPDF page text of
a sample of fdasearch PDFs and compare with fdasearch's extracted.jsonl and results/label_check_llm.jsonl.

Usage: python scripts/parse_agreement.py --predx-repo ../predicate-data-pipeline [--n 3000] [--seed 0]
Writes results/parse_agreement.csv (one row per document) and prints the summary that results/parse_agreement.md records.
"""
import argparse, collections, json, os, random, re, sys, time, zipfile
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOK = re.compile(r"[a-z0-9]+")


def toks(s):
    return set(TOK.findall((s or "").lower()))


def overlap(a, b):
    A, B = toks(a), toks(b)
    if not A or not B:
        return None, None
    i = len(A & B)
    return i / len(A | B), i / min(len(A), len(B))


def pct(x):
    return f"{100 * x:.1f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predx-repo", required=True, help="checkout of schukka28/predicate-data-pipeline")
    ap.add_argument("--n", type=int, default=3000, help="random 2010+ digital (non-OCR) corpus docs to sample")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    sys.path.insert(0, os.path.abspath(a.predx_repo))
    from predx.config import load_config
    from predx.sections import parse_sections
    from predx.predicates import extract_predicates
    import pymupdf

    cfg = load_config()
    corpus = pd.read_parquet(os.path.join(ROOT, "data", "corpus.parquet"), columns=["k", "date", "ocr"])
    llm = {}
    p = os.path.join(ROOT, "results", "label_check_llm.jsonl")
    if os.path.exists(p):
        for line in open(p, encoding="utf-8"):
            r = json.loads(line); llm[r["k"]] = r
    pool = corpus[(corpus.date >= "2010") & (~corpus.ocr)].k.tolist()
    random.seed(a.seed)
    sample = set(llm) | set(random.sample(pool, min(a.n, len(pool))))
    z = zipfile.ZipFile(os.path.join(ROOT, "data", "raw", "device-510k.json.zip"))
    known = {r["k_number"] for r in json.load(z.open(z.namelist()[0]))["results"]}

    mine = {}
    for line in open(os.path.join(ROOT, "data", "extracted.jsonl"), encoding="utf-8"):
        mk = re.match(r'\{"k": "([^"]+)"', line)
        k = mk.group(1) if mk else json.loads(line)["k"]
        if k in sample:
            mine[k] = json.loads(line)
    print("documents:", len(mine), "of", len(sample), "sampled", flush=True)

    rows, t0 = [], time.time()
    for n, k in enumerate(sorted(mine)):
        path = os.path.join(ROOT, "data", "pdfs", k + ".pdf")
        if not os.path.exists(path):
            continue
        try:
            pages = [pg.get_text() for pg in pymupdf.open(path)][:40]
            sec = parse_sections(pages, cfg)
            pred = extract_predicates(pages, k, known, fallback=cfg["parse"].get("first_other_fallback", True))
        except Exception as e:
            rows.append(dict(k=k, error=repr(e)[:120])); continue
        m = mine[k]
        my_preds = [c for c in m.get("cited", []) if m["cited_labels"].get(c) == "predicate"]
        my_refs = [c for c in m.get("cited", []) if m["cited_labels"].get(c) == "reference"]
        th_primary = next((q["k"] for q in pred["predicates"] if q["role"] == "primary"), None)
        th_gold = [q["k"] for q in pred["predicates"] if q["role"] in ("primary", "additional")]
        th_refs = [q["k"] for q in pred["predicates"] if q["role"] == "reference"]
        my_ifu = m.get("ifu_sec") or m.get("ifu_form") or ""
        my_desc = m.get("desc_sec") or ""
        ifu_j, ifu_c = overlap(sec["ifu"], my_ifu); desc_j, desc_c = overlap(sec["description"], my_desc)
        rows.append(dict(
            k=k, my_primary=my_preds[0] if my_preds else None, th_primary=th_primary, th_primary_by=pred["primary_by"],
            th_ambiguous=pred["ambiguous"], th_fallback=pred["primary_by"] == "first_other_fallback",
            my_gold=" ".join(my_preds), th_gold=" ".join(th_gold), my_refs=" ".join(my_refs), th_refs=" ".join(th_refs),
            gold_exact=set(my_preds) == set(th_gold),
            gold_jacc=(len(set(my_preds) & set(th_gold)) / len(set(my_preds) | set(th_gold))) if (my_preds or th_gold) else None,
            my_ifu=bool(my_ifu), th_ifu=bool(sec["ifu"]), th_ifu_source=sec["ifu_source"], ifu_jacc=ifu_j, ifu_contain=ifu_c,
            my_desc=bool(my_desc), th_desc=bool(sec["description"]), th_desc_source=sec["desc_source"], desc_jacc=desc_j, desc_contain=desc_c,
            th_statement_like=sec["statement_like"], in_llm=k in llm,
        ))
        if n % 500 == 0:
            print(f"  {n} docs {time.time() - t0:.0f}s", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(ROOT, "results", "parse_agreement.csv"), index=False)
    ok = df[df.error.isna()] if "error" in df else df
    print(f"\ncompared {len(ok)} docs ({len(df) - len(ok)} parse errors), {time.time() - t0:.0f}s")
    both = ok[ok.my_primary.notna() & ok.th_primary.notna()]
    print("\nPRIMARY PREDICATE")
    print(" fdasearch found:", pct(ok.my_primary.notna().mean()), "| predx found:", pct(ok.th_primary.notna().mean()),
          "| predx via first-other fallback:", pct(ok.th_fallback.mean()), "| predx ambiguous:", pct(ok.th_ambiguous.mean()))
    print(" both found:", pct(len(both) / len(ok)), "-> same primary:", pct((both.my_primary == both.th_primary).mean()))
    print(" only fdasearch:", pct((ok.my_primary.notna() & ok.th_primary.isna()).mean()), "| only predx:",
          pct((ok.my_primary.isna() & ok.th_primary.notna()).mean()), "| neither:", pct((ok.my_primary.isna() & ok.th_primary.isna()).mean()))
    print("\nGOLD SET (primary + additional)")
    print(" exact match:", pct(ok.gold_exact.mean()), "| mean Jaccard:", f"{ok.gold_jacc.mean():.3f}",
          "| fdasearch set size:", f"{ok.my_gold.str.split().str.len().mean():.2f}", "| predx set size:", f"{ok.th_gold.str.split().str.len().mean():.2f}")
    print("\nSECTIONS")
    print(" IFU found  fdasearch:", pct(ok.my_ifu.mean()), "predx:", pct(ok.th_ifu.mean()), "| predx source:", ok.th_ifu_source.value_counts(dropna=False).to_dict())
    print(" IFU overlap when both: median Jaccard", f"{ok.ifu_jacc.median():.2f}", "median containment", f"{ok.ifu_contain.median():.2f}",
          "| containment>=0.8:", pct((ok.ifu_contain >= 0.8).mean()))
    print(" DESC found fdasearch:", pct(ok.my_desc.mean()), "predx:", pct(ok.th_desc.mean()), "| predx source:", ok.th_desc_source.value_counts(dropna=False).to_dict())
    print(" DESC overlap when both: median Jaccard", f"{ok.desc_jacc.median():.2f}", "median containment", f"{ok.desc_contain.median():.2f}",
          "| containment>=0.8:", pct((ok.desc_contain >= 0.8).mean()))
    if llm:
        print("\nVS LLM LABELS")
        L = ok[ok.in_llm]
        for name, col, pcol in [("fdasearch", "my_gold", "my_primary"), ("predx", "th_gold", "th_primary")]:
            tp = fp = fn = 0
            for _, r in L.iterrows():
                g = set(r[col].split()) if r[col] else set(); l = set(llm[r.k]["llm_predicates"])
                tp += len(g & l); fp += len(g - l); fn += len(l - g)
            prim = [(row[pcol] in set(llm[row.k]["llm_predicates"])) for _, row in L.iterrows() if row[pcol] and llm[row.k]["llm_predicates"]]
            print(f" {name:9s} gold vs LLM: precision {tp / max(1, tp + fp):.2f} recall {tp / max(1, tp + fn):.2f} | "
                  f"primary in LLM set: {pct(sum(prim) / max(1, len(prim)))} (n={len(prim)}) | docs={len(L)}")
    print("\nDISAGREEMENTS (primary differs, both found)")
    for _, r in both[both.my_primary != both.th_primary].head(12).iterrows():
        print(f" {r.k}: fdasearch={r.my_primary} [{r.my_gold}] predx={r.th_primary} by {r.th_primary_by} [{r.th_gold}]")


if __name__ == "__main__":
    main()
