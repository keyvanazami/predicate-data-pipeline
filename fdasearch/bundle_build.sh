set -e
cd /e/fdasearch
cp -n data/raw/device-510k.json.zip bundle/ 
cp -n data/corpus.parquet data/queries.parquet bundle/
[ -f bundle/extracted.jsonl.gz ] || gzip -6 -c data/extracted.jsonl > bundle/extracted.jsonl.gz
[ -f bundle/extracted_closure.jsonl.gz ] || gzip -6 -c data/extracted_closure.jsonl > bundle/extracted_closure.jsonl.gz
[ -f bundle/predx.zip ] || (cd data && zip -q -r ../bundle/predx.zip predx)
for t in gemini-embedding-001 embeddinggemma BAAI__bge-m3 intfloat__e5-base-v2 Qwen__Qwen3-Embedding-0.6B nomic-embed-text; do
  cp -n results/emb/$t.d.npz results/emb/$t.q.npz bundle/emb/
done
cp results/metrics.csv results/REPORT.md bundle/reference_results/
cp data/predx/dataset_card.json bundle/reference_results/
(cd bundle && find . -type f ! -name manifest.json -exec sha256sum {} \; > /tmp/sha.txt && ls -l $(cut -c67- /tmp/sha.txt) >/dev/null)
python -X utf8 - <<'PY'
import os, json, hashlib
root="E:/fdasearch/bundle"; m=[]
for d,_,fs in os.walk(root):
    for f in fs:
        if f=="manifest.json": continue
        p=os.path.join(d,f); rel=os.path.relpath(p,root).replace("\\","/")
        h=hashlib.sha256(open(p,"rb").read()).hexdigest()
        m.append(dict(path=rel,bytes=os.path.getsize(p),sha256=h))
json.dump(dict(built="2026-11-21",dataset="fdasearch v2",files=m),open(os.path.join(root,"manifest.json"),"w"),indent=1)
print(sum(x["bytes"] for x in m)/1e9,"GB",len(m),"files")
for x in m: print(f'{x["bytes"]/1e6:8.1f} MB  {x["path"]}  {x["sha256"]}')
PY
