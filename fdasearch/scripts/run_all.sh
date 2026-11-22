#!/bin/bash
# Full zero-shot baseline pipeline. Run from E:\fdasearch.
set -e
cd /e/fdasearch
export PYTHONIOENCODING=utf-8
PY=.venv/Scripts/python
until grep -q "^done" data/pdfs_download_2010.log; do sleep 15; done
echo "== extract =="; $PY -I scripts/extract_text.py --workers 8
echo "== build =="; $PY -I scripts/build_dataset.py
rm -f results/metrics.csv
echo "== lexical =="; $PY -I scripts/run_baselines.py --pc --methods random,pop,bm25,tfidf,lsa
echo "== embeddings =="; $PY -I scripts/run_baselines.py --pc --methods emb:sentence-transformers/all-MiniLM-L6-v2,emb:BAAI/bge-base-en-v1.5,emb:intfloat/e5-base-v2,emb:Alibaba-NLP/gte-modernbert-base,emb:Qwen/Qwen3-Embedding-0.6B,emb:BAAI/bge-m3,ollama:embeddinggemma,ollama:nomic-embed-text,emb:Qwen/Qwen3-Embedding-4B
echo "== hybrids =="; $PY -I scripts/run_baselines.py --methods "rrf:bm25|ollama_embeddinggemma,rrf:bm25|emb_BAAI__bge-base-en-v1.5,rrf:bm25|emb_Qwen__Qwen3-Embedding-4B,rrf:bm25|emb_BAAI__bge-m3,rrf:bm25|ollama_embeddinggemma|emb_Qwen__Qwen3-Embedding-4B"
echo "== report =="; $PY -I scripts/summarize_results.py > /dev/null
echo "ALL DONE"
