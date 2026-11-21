"""Assemble notebooks/fdasearch_510k_baselines.ipynb: a self-contained Colab that carries the pipeline scripts
(embedded as %%writefile cells so the notebook always mirrors scripts/) and runs the top retrieval methods.
Re-run this after changing any script or after new results land in results/metrics.csv."""
import os, json, textwrap
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = ["download_pdfs.py", "extract_text.py", "build_dataset.py", "fetch_closure.py",
           "run_baselines.py", "summarize_results.py", "export_predx.py"]
OUT = os.path.join(ROOT, "notebooks", "fdasearch_510k_baselines.ipynb")


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": textwrap.dedent(src).strip("\n")}


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": textwrap.dedent(src).strip("\n")}


def results_table():
    """Top methods on the test split from the local full run, for the intro cell."""
    f = os.path.join(ROOT, "results", "metrics.csv")
    if not os.path.exists(f):
        return "(run results/metrics.csv not found when the notebook was generated)"
    m = pd.read_csv(f).drop_duplicates(["method", "split"], keep="last")
    m = m[(m.split == "test") & ~m.method.str.endswith("+pc")].sort_values("recall10", ascending=False)
    show = ["tfidf", "rrf:tfidf|gemini_gemini-embedding-001", "rrf:tfidf|ollama_embeddinggemma", "bm25",
            "gemini:gemini-embedding-001", "ollama:embeddinggemma", "emb:BAAI/bge-m3", "emb:intfloat/e5-base-v2",
            "emb:Qwen/Qwen3-Embedding-4B", "lsa", "pop"]
    nice = {"tfidf": "TF-IDF (1-2 grams, cosine)", "bm25": "BM25", "lsa": "LSA (SVD-300)", "pop": "product-code popularity",
            "ollama:embeddinggemma": "EmbeddingGemma-300M", "gemini:gemini-embedding-001": "Gemini embedding-001",
            "rrf:tfidf|ollama_embeddinggemma": "RRF TF-IDF + EmbeddingGemma", "rrf:tfidf|gemini_gemini-embedding-001": "RRF TF-IDF + Gemini",
            "emb:BAAI/bge-m3": "bge-m3", "emb:intfloat/e5-base-v2": "e5-base-v2", "emb:Qwen/Qwen3-Embedding-4B": "Qwen3-Embedding-4B"}
    rows = ["| method | Recall@10 | Hit@1 | Hit@10 | Hit@100 | MRR |", "|---|---|---|---|---|---|"]
    for meth in show:
        r = m[m.method == meth]
        if len(r):
            r = r.iloc[0]
            rows.append(f"| {nice.get(meth, meth)} | {100*r.recall10:.1f} | {100*r.hit1:.1f} | {100*r.hit10:.1f} | {100*r.hit100:.1f} | {r.mrr:.2f} |")
    n = int(m.n.iloc[0])
    return f"Test split (n={n} subjects cleared 2025-2026), no fine-tuning:\n\n" + "\n".join(rows)


cells = []
cells.append(md(f"""
# Zero-shot predicate retrieval for FDA 510(k) devices

Given a new device's **Indications for Use + Device Description**, rank earlier 510(k) clearances so that the
predicates its summary cites come first. This notebook rebuilds the dataset from public FDA records and runs the
untuned retrieval baselines (lexical, dense embeddings, hybrids). COMS 4705 project: *Retrieving 510(k) Predicate
Devices with Fine-Tuned Text Embeddings* (K. Azami, J. Lamm, S. Chukka).

**Pipeline** (each stage is resumable; re-run a cell after a disconnect and it skips finished work)

1. openFDA 510(k) metadata (176k records) -> summary PDFs from accessdata.fda.gov
2. PyMuPDF text extraction, optional GPU OCR for image-only scans; regex section headers; cited K-numbers with predicate/reference labels
3. optional *closure fetch*: cited predicates older than the corpus window are downloaded too
4. dataset: K-numbers masked, substantial-equivalence sentences stripped; train 2010-2023 / dev 2024 / test 2025+
5. baselines + metrics (Recall@10, Hit@k, MRR with bootstrap CIs); report by review panel and candidate-pool size

**Data source.** `DATA_SOURCE = "drive"` (default) is hermetic with respect to data: it reads the prebuilt v2
bundle (`fdasearch_data_v2` on Google Drive, 670 MB: openFDA export, extracted PDF text incl. OCR and closure docs,
the corpus/query tables and the `predx` export, sha256-verified against `manifest.json`), so nothing is fetched from
FDA and the dataset is byte-identical to the reference run. Models are still downloaded from Hugging Face and
embedded on the Colab GPU (Gemini via the API). `DATA_SOURCE = "fda"` rebuilds the data from openFDA and
accessdata.fda.gov instead.

**Two modes.** `quick` restricts everything to one advisory committee (default Cardiovascular, ~6k documents;
about 10 minutes on a T4 from the bundle, ~15 minutes more when rebuilding). `full` uses all panels (~49k documents;
lexical methods in minutes, each dense model 20-40 minutes on a T4, hours more when rebuilding from FDA).

**Reference results from the full run on an RTX 4090** (same code; numbers in a quick run differ because the
candidate pool is one panel):

{results_table()}

Lexical search wins zero-shot; EmbeddingGemma and Gemini are the best untuned embedders; fusion helps Hit@100.
About 9% of the regex "primary predicate" labels disagree with an LLM read of the summary, so all scores are lower bounds.
"""))

cells.append(md("## 0. Settings"))
cells.append(code("""
DATA_SOURCE = "drive"       # "drive" = prebuilt v2 bundle from Google Drive (hermetic); "fda" = rebuild from FDA downloads
DATA_DIR    = "/content/drive/MyDrive/fdasearch_data_v2"   # bundle location after drive.mount (if it was shared with you: Drive > Organize > Add shortcut to My Drive)
DATA_FOLDER_ID = "1Ac0fRIa2dSDRngPj8Pql8eduyMIQY6bk"        # fallback: fetch the folder by id with gdown (needs "anyone with the link" sharing)

MODE       = "quick"        # "quick" = one advisory committee; "full" = all panels
PANEL      = "CV"           # quick mode only. Codes: OR Orthopedic, RA Radiology, CV Cardiovascular, SU Surgery, DE Dental, HO Hospital, GU Gastro/Urology, NE Neurology, ...
MIN_YEAR   = 2010           # corpus window start (summaries before ~2000 are mostly scans)
RUN_CLOSURE = True          # fda source only: also fetch cited predicates outside the window so they become retrievable
RUN_OCR    = False          # fda source only: OCR image-only PDFs (needs a GPU runtime; ~0.7 s/page on T4, ~7 s/page on CPU)
WORKERS    = 2              # extraction processes (free Colab has 2 vCPUs)

# Dense models, downloaded from Hugging Face and run on the Colab GPU. EmbeddingGemma is gated: accept the license at
# https://huggingface.co/google/embeddinggemma-300m and put a token in Colab secrets as HF_TOKEN.
EMBED_MODELS = ["google/embeddinggemma-300m", "BAAI/bge-m3"]      # [] to skip dense models
RUN_GEMINI   = True         # Gemini embedding-001 via the API; needs GEMINI_API_KEY in Colab secrets (skipped if absent)

import os
def secret(name):
    try:
        from google.colab import userdata
        v = userdata.get(name)
        if v: return v
    except Exception:
        pass
    return os.environ.get(name, "")
HF_TOKEN = secret("HF_TOKEN"); GEMINI_API_KEY = secret("GEMINI_API_KEY")
print("HF token:", "yes" if HF_TOKEN else "no (EmbeddingGemma will be skipped)", "| Gemini key:", "yes" if GEMINI_API_KEY else "no (skipped)")
"""))

cells.append(md("## 1. Environment"))
cells.append(code("""
%pip install -q "pymupdf>=1.24" "pandas>=2.2" pyarrow "bm25s>=0.2" PyStemmer "scikit-learn>=1.4" "sentence-transformers>=5.0" "transformers>=4.56" google-genai requests
if RUN_OCR:
    %pip install -q rapidocr-onnxruntime onnxruntime-gpu
import os, sys, json, glob, subprocess, pathlib
ROOT = pathlib.Path("/content/fdasearch"); (ROOT / "scripts").mkdir(parents=True, exist_ok=True)
for d in ["data/raw", "data/pdfs", "results"]: (ROOT / d).mkdir(parents=True, exist_ok=True)
os.chdir(ROOT)
def sh(cmd):
    # run a shell command and stream its output into the cell (Colab does not reliably show inherited stdio)
    print("$", cmd, flush=True)
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                         env={**os.environ, "PYTHONUNBUFFERED": "1"})
    tail = []
    for line in p.stdout:
        print(line, end="", flush=True); tail.append(line); tail = tail[-40:]
    if p.wait():
        raise RuntimeError(f"command failed ({p.returncode}); last lines:\\n" + "".join(tail[-15:]))
import torch; print("python", sys.version.split()[0], "| GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
"""))

cells.append(md("""
## 2. Pipeline scripts
The cells below write the project's scripts into `/content/fdasearch/scripts/`. They are the exact files used for
the reference results (generated from the repo by `scripts/make_colab.py`).
"""))
for name in SCRIPTS:
    src = open(os.path.join(ROOT, "scripts", name), encoding="utf-8").read()
    cells.append(code(f"%%writefile scripts/{name}\n{src}"))

cells.append(md("""
## 3. Data
**drive**: the bundle is copied from Drive (mounted, or fetched by folder id), every file is checked against the
sha256 in `manifest.json` and the extracted text is unpacked. In `quick` mode the dataset is rebuilt for one panel
from the extracted text (no PDFs needed); in `full` mode the shipped corpus/query tables are used as they are.

**fda**: the downloader sends an identifying research User-Agent (FDA's server rejects anonymous clients) and runs
at ~25 PDFs/s with 6 workers. ~0.6% of summaries have no PDF (404).
"""))
cells.append(code("""
import hashlib, shutil, gzip, zipfile
panel_arg = f"--panel {PANEL}" if MODE == "quick" else ""

def fetch_bundle():
    src = pathlib.Path(DATA_DIR)
    if not src.exists():
        try:
            from google.colab import drive; drive.mount("/content/drive")
        except Exception as e:
            print("Drive mount not available:", e)
    if not src.exists() and DATA_FOLDER_ID:
        sh("pip install -q gdown")
        src = pathlib.Path("/content/fdasearch_data_v2")
        sh(f"gdown --folder https://drive.google.com/drive/folders/{DATA_FOLDER_ID} -O {src}")
    assert (src / "manifest.json").exists(), f"bundle not found at {src}: mount Drive with fdasearch_data_v2 in My Drive, or set DATA_FOLDER_ID"
    man = json.load(open(src / "manifest.json"))
    place = {"device-510k.json.zip": "data/raw/device-510k.json.zip", "extracted.jsonl.gz": "data/extracted.jsonl.gz",
             "extracted_closure.jsonl.gz": "data/extracted_closure.jsonl.gz", "corpus.parquet": "data/corpus_v2.parquet",
             "queries.parquet": "data/queries_v2.parquet", "predx.zip": "data/predx_v2.zip"}
    for f in man["files"]:
        rel = f["path"]
        dst = ROOT / (place.get(rel) or ("results/emb/" + rel[4:] if rel.startswith("emb/") else "results/reference/" + rel.split("/")[-1] if rel.startswith("reference_results/") else rel))
        if dst.exists() and dst.stat().st_size == f["bytes"]:
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        print(f"  {rel} ({f['bytes']/1e6:.0f} MB)", flush=True)
        shutil.copyfile(src / rel, dst)
        h = hashlib.sha256()
        with open(dst, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 24), b""): h.update(chunk)
        assert h.hexdigest() == f["sha256"], f"checksum mismatch for {rel}"
    for gz in ["data/extracted.jsonl.gz", "data/extracted_closure.jsonl.gz"]:
        out = ROOT / gz[:-3]
        if not out.exists():
            with gzip.open(ROOT / gz, "rb") as fi, open(out, "wb") as fo: shutil.copyfileobj(fi, fo, 1 << 24)
    if MODE == "quick":
        sh(f"python scripts/build_dataset.py {panel_arg}")
    else:
        shutil.copyfile(ROOT / "data/corpus_v2.parquet", ROOT / "data/corpus.parquet")
        shutil.copyfile(ROOT / "data/queries_v2.parquet", ROOT / "data/queries.parquet")
    print("bundle ready")

if DATA_SOURCE == "drive":
    fetch_bundle()
else:
    sh(f"python scripts/download_pdfs.py --min-year {MIN_YEAR} --workers 6 {panel_arg}")
    sh(f"python scripts/extract_text.py --workers {WORKERS}")
    if RUN_OCR:
        sh("python scripts/extract_text.py --ocr --workers 1 --redo-low-text")
    sh(f"python scripts/build_dataset.py {panel_arg}")
"""))
cells.append(md("""
### 3b. Closure fetch (fda source only)
The bundle already contains the closure documents. When rebuilding from FDA, predicates cited by the queries but outside the window (mostly 2000s clearances) are downloaded and extracted,
then the dataset is rebuilt so they count as retrievable positives. Older PDFs are often scans: without OCR,
about half of them yield no text and stay unretrievable.
"""))
cells.append(code("""
if RUN_CLOSURE and DATA_SOURCE == "fda":
    sh("python scripts/fetch_closure.py")
    ocr = "--ocr --workers 1" if RUN_OCR else f"--workers {WORKERS}"
    sh(f"python scripts/extract_text.py {ocr} --pdf-dir data/pdfs_closure --out data/extracted_closure.jsonl")
    sh(f"python scripts/build_dataset.py {panel_arg}")
import pandas as pd
q = pd.read_parquet("data/queries.parquet"); c = pd.read_parquet("data/corpus.parquet")
print(f"corpus {len(c)} docs {c.date.min()}..{c.date.max()} | queries", q.split.value_counts().to_dict())
"""))

cells.append(md("""
## 4. Baselines
Candidates for each query are the corpus documents cleared before the subject. `--pc` additionally scores each
method restricted to the subject's product code. Embeddings are cached per K-number in `results/emb/`.
"""))
cells.append(code("""
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN            # read by huggingface_hub in the subprocesses below
    from huggingface_hub import login; login(token=HF_TOKEN)
import transformers; print("transformers", transformers.__version__, "(EmbeddingGemma needs >= 4.56)")
if GEMINI_API_KEY:
    open(".env", "w").write(f"GEMINI_API_KEY={GEMINI_API_KEY}\\n")

sh("python scripts/run_baselines.py --pc --methods random,pop,bm25,tfidf,lsa")
dense_specs = ["emb:" + m for m in EMBED_MODELS if HF_TOKEN or "gemma" not in m]
if RUN_GEMINI and GEMINI_API_KEY:
    dense_specs.append("gemini:gemini-embedding-001")
if dense_specs:
    sh("python scripts/run_baselines.py --pc --methods " + ",".join(dense_specs))
"""))
cells.append(md("""
### 4b. Hybrids
Reciprocal-rank fusion of the saved top-100 lists (`rrf:a|b`; names are the sanitized method names in `results/top/`).
"""))
cells.append(code("""
import re
tags = [re.sub(r"[^A-Za-z0-9_.+-]", "_", m.replace("/", "__")) for m in dense_specs]      # e.g. emb_google__embeddinggemma-300m, gemini_gemini-embedding-001
fusions = [f"rrf:tfidf|{t}" for t in tags] + [f"rrf:bm25|{t}" for t in tags[:1]]
if fusions:
    sh('python scripts/run_baselines.py --methods "' + ",".join(fusions) + '"')
"""))

cells.append(md("## 5. Report"))
cells.append(code("""
sh("python scripts/summarize_results.py > /dev/null")
from IPython.display import Markdown, display
display(Markdown(open("results/REPORT.md").read()))
"""))
cells.append(md("""
### Reproducibility check
With `DATA_SOURCE="drive"` and `MODE="full"` the lexical methods must match the reference run to the fourth decimal
(same texts, same code); dense models re-embedded here should land within a few tenths of a point (GPU/batch
nondeterminism). The reference EmbeddingGemma run went through Ollama (`ollama:embeddinggemma`), so it has no
direct counterpart. In `quick` mode the pool is one panel, so numbers differ by design.
"""))
cells.append(code("""
import pandas as pd
ref = "results/reference/metrics.csv"
if os.path.exists(ref):
    a = pd.read_csv("results/metrics.csv").drop_duplicates(["method", "split"], keep="last").set_index(["method", "split"])
    b = pd.read_csv(ref).set_index(["method", "split"])
    common = a.index.intersection(b.index)
    cmp = pd.DataFrame({"this_run": a.loc[common, "recall10"], "reference": b.loc[common, "recall10"]})
    cmp["diff"] = (cmp.this_run - cmp.reference).round(4)
    print(f"{len(common)} (method, split) pairs in common; max |diff| in Recall@10 = {cmp['diff'].abs().max():.4f}")
    display(cmp.sort_values("reference", ascending=False).head(20))
"""))
cells.append(md("""
### Look at one query
What the model sees (K-numbers masked, substantial-equivalence sentences removed), the cited predicates, and the
TF-IDF top 10.
"""))
cells.append(code("""
import json, numpy as np
q = pd.read_parquet("data/queries.parquet"); c = pd.read_parquet("data/corpus.parquet")
qt = q[q.split == "test"].reset_index(drop=True)
top = np.load("results/top/tfidf.npy")                      # rows follow the dev+test query order used by run_baselines
qd = q[q.split.isin(["dev", "test"])].reset_index(drop=True)
i = int(qd.index[qd.split == "test"][0])                    # first test query; change i to browse
r = qd.iloc[i]
print(r.k, r.date, r.product_code, "|", r.panel); print(r.query[:600], "...\\n")
print("cited predicates:", list(r.predicates_all), "| primary:", r.primary)
for rank, j in enumerate(top[i][:10], 1):
    d = c.iloc[j]; mark = "<-- predicate" if d.k in set(r.positives) else ""
    print(f"{rank:2d}. {d.k} {d.date} {d.product_code} {d.device_name[:60]} {mark}")
"""))

cells.append(md("""
## 6. Keep the results
Zips the metrics, report, per-query ranks and the dataset in the team's `predx` layout (`data/predx/`).
Mount Drive first if you want them to survive the runtime.
"""))
cells.append(code("""
sh("python scripts/export_predx.py")
import shutil, zipfile
with zipfile.ZipFile("/content/fdasearch_results.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for p in ["results/metrics.csv", "results/REPORT.md", "data/dataset_card.json"]:
        if os.path.exists(p): z.write(p)
    for d in ["results/ranks", "data/predx"]:
        for f in glob.glob(d + "/*"): z.write(f)
print("/content/fdasearch_results.zip", f"{os.path.getsize('/content/fdasearch_results.zip')/1e6:.1f} MB")
try:
    from google.colab import files; files.download("/content/fdasearch_results.zip")
except Exception as e:
    print("not in Colab:", e)
# from google.colab import drive; drive.mount('/content/drive'); shutil.copy('/content/fdasearch_results.zip', '/content/drive/MyDrive/')
"""))

nb = {"nbformat": 4, "nbformat_minor": 5, "metadata": {
    "kernelspec": {"name": "python3", "display_name": "Python 3"},
    "language_info": {"name": "python"},
    "colab": {"name": "fdasearch_510k_baselines.ipynb", "provenance": [], "toc_visible": True},
    "accelerator": "GPU"}, "cells": cells}
for i, cl in enumerate(cells):
    cl["id"] = f"cell-{i}"
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(nb, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("wrote", OUT, len(cells), "cells", f"{os.path.getsize(OUT)/1e3:.0f} KB")
