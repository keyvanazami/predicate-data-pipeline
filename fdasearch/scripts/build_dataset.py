"""Join openFDA metadata with extracted PDF text; build corpus + query splits.

Outputs data/corpus.parquet (one row per device with usable text) and data/queries.parquet
(subjects from 2024 [dev] and 2025-2026 [test] with >=1 predicate found in the corpus).
"""
import json, os, re, zipfile, collections
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw", "device-510k.json.zip")
EXT = os.path.join(ROOT, "data", "extracted.jsonl")
EXT_CLOSURE = os.path.join(ROOT, "data", "extracted_closure.jsonl")   # cited predicates outside 2010+ (team's "closure fetch")
TRAIN_FROM = 2010   # train = subjects TRAIN_FROM..2023, dev = 2024, test = 2025+
SE_RE = re.compile(r"(?i)predicate|substantial(?:ly)?[ -]equivalen|510\s*\(?k\)?|reference device")
SENT_RE = re.compile(r"(?<=[.;:])\s+")
K_RE = re.compile(r"\b[KkPpDd](?:EN|en)?\s?\d{6}\b")
WS = re.compile(r"\s+")
MIN_CHARS = 200


def clean(s, maxlen):
    """Mask K-numbers and drop sentences that talk about predicates / substantial equivalence
    (the team's SE_LANGUAGE_IN_INPUT leak check). Returns (text, n_sentences_dropped)."""
    s = K_RE.sub(" ", s or "")
    s = WS.sub(" ", s).strip()
    sents = SENT_RE.split(s)
    keep = [x for x in sents if not SE_RE.search(x)]
    dropped = len(sents) - len(keep)
    if not keep:          # everything mentioned predicates: keep the masked text rather than nothing
        keep, dropped = sents, 0
    return " ".join(keep)[:maxlen], dropped


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="", help="restrict corpus and queries to one advisory committee (code like CV, or description substring)")
    a = ap.parse_args()
    z = zipfile.ZipFile(RAW)
    recs = json.load(z.open(z.namelist()[0]))["results"]
    meta = {}
    for r in recs:
        o = r.get("openfda", {})
        meta[r["k_number"]] = dict(
            k=r["k_number"], date=r["decision_date"], product_code=r.get("product_code", ""),
            device_name=r.get("device_name", ""), applicant=r.get("applicant", ""),
            panel=r.get("advisory_committee_description", ""), specialty=o.get("medical_specialty_description", ""),
            class_name=o.get("device_name", ""), regulation=o.get("regulation_number", ""),
            device_class=o.get("device_class", ""), decision=r.get("decision_code", ""),
            kind=r.get("statement_or_summary", ""), panel_code=r.get("advisory_committee", ""),
        )
    if a.panel:
        p = a.panel.lower()
        meta = {k: m for k, m in meta.items() if m["panel_code"].lower() == p or p in m["panel"].lower()}
        print(f"panel filter {a.panel!r}: {len(meta)} devices", flush=True)
    rows = []
    stats = collections.Counter()
    lines = list(open(EXT, encoding="utf-8"))
    if os.path.exists(EXT_CLOSURE):
        lines += list(open(EXT_CLOSURE, encoding="utf-8"))
    seen = set()
    for line in lines:
        e = json.loads(line)
        k = e["k"]
        if k in seen:
            continue
        seen.add(k)
        if k not in meta or e.get("error"):
            stats["no_meta_or_error"] += 1; continue
        m = dict(meta[k])
        ifu = e.get("ifu_sec") or e.get("ifu_form") or ""
        desc = e.get("desc_sec") or ""
        summ = e.get("summary_text") or ""
        if (e.get("n_chars") or 0) < MIN_CHARS:
            stats["too_little_text"] += 1; continue
        stats["has_ifu" if ifu else "no_ifu"] += 1
        stats["has_desc" if desc else "no_desc"] += 1
        ifu_c, d1 = clean(ifu, 3000); desc_c, d2 = clean(desc, 4000); summ_c, d3 = clean(summ, 6000)
        stats["se_sentences_dropped"] += d1 + d2
        m.update(
            ifu=ifu_c, desc=desc_c, summary=summ_c, se_dropped=d1 + d2,
            n_pages=e.get("n_pages"), n_chars=e.get("n_chars"), ocr=bool(e.get("ocr")),
            cited=e.get("cited", []),
            predicates=[c for c in e.get("cited", []) if e["cited_labels"].get(c) == "predicate"],
            references=[c for c in e.get("cited", []) if e["cited_labels"].get(c) == "reference"],
        )
        # retrieval text: name + IFU + description; fallback to summary body
        core = " ".join(x for x in [m["device_name"], m["ifu"], m["desc"]] if x)
        m["text"] = core if (ifu or desc) else " ".join([m["device_name"], m["summary"][:4000]])
        m["text_source"] = "sections" if (ifu or desc) else "summary"
        rows.append(m)
    corpus = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    corpus.to_parquet(os.path.join(ROOT, "data", "corpus.parquet"))
    print("corpus:", len(corpus), dict(stats))
    print(corpus.groupby(corpus.date.str[:4]).size().to_dict())
    print("text_source:", corpus.text_source.value_counts().to_dict(), "ocr docs:", int(corpus.ocr.sum()))
    name_of = dict(zip(corpus.k, corpus.device_name.str.lower()))

    in_corpus = dict(zip(corpus.k, corpus.date))
    qrows = []
    cov = collections.Counter()
    for _, r in corpus.iterrows():
        y = int(r.date[:4])
        if y < TRAIN_FROM:
            continue
        preds = list(r.predicates)
        if not preds:
            cov["no_predicate_cited"] += 1; continue
        valid = [p for p in preds if p in in_corpus and in_corpus[p] < r.date]
        cov["pred_total"] += len(preds); cov["pred_in_corpus"] += len(valid)
        cov["pred_missing_meta"] += sum(1 for p in preds if p not in in_corpus)
        if not valid:
            cov["no_valid_predicate"] += 1; continue
        qrows.append(dict(
            k=r.k, date=r.date, product_code=r.product_code, specialty=r.specialty, panel=r.panel,
            split="train" if y < 2024 else ("dev" if y == 2024 else "test"), query=r.text, text_source=r.text_source,
            predicates_all=preds, positives=valid, primary=preds[0] if preds[0] in valid else None,
            references=[p for p in r.references if p in in_corpus and in_corpus[p] < r.date],
            # leak flags (team's audit): predicate device name appearing verbatim in the query text
            leak_name=any(len(name_of.get(p, "")) >= 12 and name_of[p] in r.text.lower() for p in valid),
            se_dropped=int(r.se_dropped),
        ))
    q = pd.DataFrame(qrows)
    q.to_parquet(os.path.join(ROOT, "data", "queries.parquet"))
    print("queries:", len(q), q.split.value_counts().to_dict(), dict(cov))
    print("coverage of cited predicates by corpus: %.1f%%" % (100 * cov["pred_in_corpus"] / max(1, cov["pred_total"])))
    print("positives per query:", q.positives.map(len).describe().to_dict())
    for sp in ["train", "dev", "test"]:
        g = q[q.split == sp]
        if len(g):
            print(f"{sp}: n={len(g)} coverage-of-cited={g.positives.map(len).sum() / max(1, g.predicates_all.map(len).sum()):.3f} "
                  f"leak_name={g.leak_name.mean():.3f} se_dropped>0={(g.se_dropped > 0).mean():.3f}")


if __name__ == "__main__":
    main()
