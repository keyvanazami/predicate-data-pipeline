"""Export data/corpus.parquet + data/queries.parquet in the team's predx dataset layout so their training code
can read it: data/predx/{corpus.jsonl.gz, queries_{train,dev,test}.jsonl.gz, train_pairs.jsonl.gz, dataset_card.json}.
Field names follow the records printed by predx_colab_test_2.ipynb (cell 25)."""
import os, json, gzip, collections, datetime
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "predx"); os.makedirs(OUT, exist_ok=True)


def jl(path, rows):
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    c = pd.read_parquet(os.path.join(ROOT, "data", "corpus.parquet"))
    q = pd.read_parquet(os.path.join(ROOT, "data", "queries.parquet"))
    date = dict(zip(c.k, c.date)); code = dict(zip(c.k, c.product_code))
    corpus_rows = [dict(k=r.k, date=r.date, year=int(r.date[:4]), product_code=r.product_code, panel=r.panel,
                        device_name=r.device_name, applicant=r.applicant, ifu=r.ifu, description=r.desc,
                        n_pages=int(r.n_pages) if pd.notna(r.n_pages) else None, ocr=bool(r.ocr),
                        text_source=r.text_source) for r in c.itertuples()]
    jl(os.path.join(OUT, "corpus.jsonl.gz"), corpus_rows)
    counts = {}
    pairs = []
    for split in ["train", "dev", "test"]:
        g = q[q.split == split]
        rows = []
        for r in g.itertuples():
            cr = c.loc[c.k == r.k].iloc[0]
            gold = []
            for i, p in enumerate(r.predicates_all):
                inc = p in date and date[p] < r.date
                gold.append(dict(k=p, role="primary" if i == 0 else "additional", in_corpus=bool(inc),
                                 reason=None if inc else ("no_text" if p not in date else "not_earlier"),
                                 year=(int(date[p][:4]) if p in date else None), product_code=code.get(p)))
            n_cand = int((c.date < r.date).sum()); n_same = int(((c.date < r.date) & (c.product_code == r.product_code)).sum())
            rows.append(dict(k=r.k, split=split, date=r.date, year=int(r.date[:4]), product_code=r.product_code, panel=r.panel,
                             device_name=cr.device_name, ifu=cr.ifu, description=cr.desc,
                             gold=gold, reference=list(r.references), gold_in_corpus=list(r.positives),
                             gold_primary_in_corpus=[r.primary] if isinstance(r.primary, str) else [],
                             n_candidates=n_cand, n_same_code_candidates=n_same, retrievable=len(r.positives) > 0,
                             leak=dict(residual_k=0, pred_language=False, gold_name=bool(r.leak_name), se_sentences_dropped=int(r.se_dropped))))
            if split == "train":
                for p in r.positives:
                    pairs.append(dict(query_k=r.k, doc_k=p, role="primary" if p == r.primary else "additional"))
        jl(os.path.join(OUT, f"queries_{split}.jsonl.gz"), rows)
        counts[split] = len(rows)
    jl(os.path.join(OUT, "train_pairs.jsonl.gz"), pairs)
    card = dict(built=datetime.date.today().isoformat(), n_corpus=len(c), counts=counts, n_train_pairs=len(pairs),
                mask_level="k_numbers+se_sentences", corpus_years=[int(c.date.min()[:4]), int(c.date.max()[:4])],
                ocr_docs=int(c.ocr.sum()), source="fdasearch pipeline (PyMuPDF + RapidOCR), openFDA export 2026-11-18")
    json.dump(card, open(os.path.join(OUT, "dataset_card.json"), "w"), indent=1)
    print(card)


if __name__ == "__main__":
    main()
