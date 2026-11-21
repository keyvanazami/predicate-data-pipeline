"""Turn results/metrics.csv + per-query files into a markdown report (results/REPORT.md)."""
import os, json, glob
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")


def fmt(r, c):
    return f"{100*r[c]:.1f} [{100*r[c+'_lo']:.1f}, {100*r[c+'_hi']:.1f}]"


def main():
    m = pd.read_csv(os.path.join(RES, "metrics.csv"))
    m = m.drop_duplicates(subset=["method", "split"], keep="last")
    corpus = pd.read_parquet(os.path.join(ROOT, "data", "corpus.parquet"))
    queries = pd.read_parquet(os.path.join(ROOT, "data", "queries.parquet"))
    lines = ["# Zero-shot predicate retrieval baselines", ""]
    lines.append(f"Corpus: {len(corpus)} 510(k) summaries ({corpus.date.min()} to {corpus.date.max()}). "
                 f"Queries: {len(queries)} subjects with at least one cited predicate in the corpus "
                 f"(dev = 2024: {(queries.split=='dev').sum()}, test = 2025-2026: {(queries.split=='test').sum()}). "
                 f"Candidates for each query = corpus devices cleared before the subject's decision date. "
                 f"Numbers are percentages with bootstrap 95% CIs over queries.")
    lines.append("")
    for split in ["test", "dev"]:
        g = m[m.split == split].sort_values("recall10", ascending=False)
        lines += [f"## {split} split (n={int(g.n.iloc[0])})", "",
                  "| method | Recall@10 | Hit@1 | Hit@10 | Hit@100 | MRR | primary Hit@1 | primary Hit@10 |", "|---|---|---|---|---|---|---|---|"]
        for _, r in g.iterrows():
            lines.append(f"| {r.method} | {fmt(r,'recall10')} | {fmt(r,'hit1')} | {fmt(r,'hit10')} | {fmt(r,'hit100')} | {fmt(r,'mrr')} | {fmt(r,'prim_hit1')} | {fmt(r,'prim_hit10')} |")
        lines.append("")
    # per-specialty breakdown for each method on test
    KEEP = ["tfidf", "bm25", "lsa", "pop", "ollama_embeddinggemma", "emb_BAAI__bge-m3", "emb_intfloat__e5-base-v2",
            "emb_Qwen__Qwen3-Embedding-4B", "gemini_gemini-embedding-001", "rrf_tfidf_ollama_embeddinggemma", "rrf_tfidf_gemini_gemini-embedding-001"]
    pq = {}
    for f in glob.glob(os.path.join(RES, "ranks", "*.per_query.csv")):
        name = os.path.basename(f).replace(".per_query.csv", "")
        if name in KEEP:
            pq[name] = pd.read_csv(f)
    if pq:
        qmeta = queries.set_index("k")
        lines += ["## Recall@10 by review panel (test split)", ""]
        tab = {}
        for name, d in pq.items():
            d = d[d.split == "test"].copy()
            d["panel"] = d.k.map(qmeta.panel)
            tab[name] = d.groupby("panel").recall10.mean()
        tab = pd.DataFrame(tab)
        cnt = queries[queries.split == "test"].groupby("panel").size()
        tab.insert(0, "n", cnt)
        tab = tab.sort_values("n", ascending=False)
        lines.append("| panel | n | " + " | ".join(tab.columns[1:]) + " |")
        lines.append("|---|---|" + "---|" * (len(tab.columns) - 1))
        for p, r in tab.iterrows():
            lines.append(f"| {p} | {int(r.n)} | " + " | ".join(f"{100*r[c]:.0f}" for c in tab.columns[1:]) + " |")
        lines.append("")
        # by candidate-pool size within product code
        lines += ["## Recall@10 by number of earlier same-product-code devices in corpus (test split)", ""]
        pc_count = {}
        corp = corpus[["k", "date", "product_code"]].sort_values("date")
        for q in queries.itertuples():
            pc_count[q.k] = int(((corp.product_code == q.product_code) & (corp.date < q.date)).sum())
        bins = [0, 10, 50, 200, 10 ** 9]; labels = ["<10", "10-49", "50-199", "200+"]
        tab = {}
        for name, d in pq.items():
            d = d[d.split == "test"].copy()
            d["bin"] = pd.cut(d.k.map(pc_count), bins=bins, labels=labels, right=False)
            tab[name] = d.groupby("bin", observed=True).recall10.mean()
        tab = pd.DataFrame(tab)
        qt = queries[queries.split == "test"]
        tab.insert(0, "n", pd.cut(qt.k.map(pc_count), bins=bins, labels=labels, right=False).value_counts())
        lines.append("| same-code candidates | n | " + " | ".join(tab.columns[1:]) + " |")
        lines.append("|---|---|" + "---|" * (len(tab.columns) - 1))
        for p, r in tab.iterrows():
            lines.append(f"| {p} | {int(r.n)} | " + " | ".join(f"{100*r[c]:.0f}" for c in tab.columns[1:]) + " |")
        lines.append("")
    open(os.path.join(RES, "REPORT.md"), "w", encoding="utf-8").write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
