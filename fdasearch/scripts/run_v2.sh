#!/bin/bash
# Dataset v2: OCR'd low-text PDFs + closure-fetched pre-2010 predicates + SE-sentence stripping + train split.
# Archives the v1 data/results, rebuilds, reruns every baseline (embeddings come from the per-K cache; only new docs are embedded).
set -e
cd /e/fdasearch
export PYTHONIOENCODING=utf-8
PY=.venv/Scripts/python
if [ ! -d results_v1 ]; then
  mkdir -p results_v1 data/v1
  cp results/metrics.csv results/REPORT.md results_v1/ && cp -r results/ranks results/top results_v1/
  cp data/corpus.parquet data/queries.parquet data/v1/
  rm -f results/metrics.csv && rm -rf results/ranks results/top
fi
echo "== build =="; $PY -X utf8 scripts/build_dataset.py
echo "== export =="; $PY -X utf8 scripts/export_predx.py
echo "== lexical =="; $PY -I scripts/run_baselines.py --pc --methods random,pop,bm25,tfidf,lsa
echo "== embeddings =="; $PY -I scripts/run_baselines.py --pc --methods ollama:embeddinggemma,emb:BAAI/bge-m3,emb:intfloat/e5-base-v2,emb:BAAI/bge-base-en-v1.5,emb:Qwen/Qwen3-Embedding-0.6B,emb:Alibaba-NLP/gte-modernbert-base,emb:sentence-transformers/all-MiniLM-L6-v2,ollama:nomic-embed-text,emb:Qwen/Qwen3-Embedding-4B
echo "== gemini =="; $PY -I scripts/run_baselines.py --pc --methods gemini:gemini-embedding-001 || echo "gemini failed (quota?) - continuing"
echo "== hybrids =="; $PY -I scripts/run_baselines.py --methods "rrf:tfidf|ollama_embeddinggemma,rrf:bm25|ollama_embeddinggemma,rrf:tfidf|emb_BAAI__bge-m3" 
$PY -I scripts/run_baselines.py --methods "rrf:tfidf|gemini_gemini-embedding-001" || true
echo "== report =="; $PY -I scripts/summarize_results.py > /dev/null
echo "ALL DONE"
