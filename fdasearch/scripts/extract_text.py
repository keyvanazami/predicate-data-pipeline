"""Extract text, sections, and cited K-numbers from downloaded 510(k) PDFs.

Usage: python extract_text.py [--workers 8]
Reads data/pdfs/K*.pdf, writes data/extracted.jsonl (one record per K; incremental: skips Ks already present).
"""
import argparse, glob, json, os, re, sys
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDFS = os.path.join(ROOT, "data", "pdfs")
OUTF = os.path.join(ROOT, "data", "extracted.jsonl")
OCR_MIN_CHARS_PER_PAGE = 100   # below this, the PDF is treated as image-only and OCR'd (if --ocr)
OCR_MAX_PAGES = 12
_OCR = None


def ocr_pages(path, dpi=150, max_pages=OCR_MAX_PAGES):
    """OCR an image-only PDF with RapidOCR (ONNX, CUDA if available). Returns list of page texts.
    Lines are ordered top-to-bottom, left-to-right so section headers stay on their own lines."""
    global _OCR
    if _OCR is None:
        import torch  # noqa: F401  loads the CUDA 12 runtime DLLs that onnxruntime-gpu links against
        from rapidocr_onnxruntime import RapidOCR
        _OCR = RapidOCR(det_use_cuda=True, rec_use_cuda=True, cls_use_cuda=True)
    import pymupdf, numpy as np
    out = []
    for page in pymupdf.open(path)[:max_pages]:
        pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
        res, _ = _OCR(img)
        if not res:
            out.append(""); continue
        res = sorted(res, key=lambda r: (round(r[0][0][1] / 12), r[0][0][0]))
        out.append("\n".join(r[1] for r in res))
    return out

K_RE = re.compile(r"\b[KkPpDd](?:EN|en)?\s?\d{6}\b")
HDR_WORDS = r"(?:510\s*\(?k\)?\s*summary|submitter|applicant|sponsor|device(?:\s+trade)?\s+name|predicate(?:\s+device)?s?|reference\s+device?s?|device\s+description|description\s+of\s+(?:the\s+)?device|indications?\s+for\s+use|intended\s+use|technological\s+characteristics|comparison|substantial\s+equivalence|performance\s+(?:data|testing)|non-?clinical|clinical|conclusion|summary\s+of|regulatory\s+information|classification|common\s+name)"
HDR_RE = re.compile(r"(?im)^[ \t]*(?:[IVX]+[\.\)]?\s*|\d+(?:\.\d+)?[\.\)]?\s*|[A-Z][\.\)]\s*)?(" + HDR_WORDS + r")[^\n]{0,40}$")


def norm_k(s):
    return s.upper().replace(" ", "")


def section_after(text, pat, maxlen=4000):
    """Return body text following the first header matching pat (regex on header word), up to next header."""
    for m in HDR_RE.finditer(text):
        if re.fullmatch(pat, m.group(1).strip(), flags=re.I):
            start = m.end()
            nxt = HDR_RE.search(text, start + 1)
            end = nxt.start() if nxt else len(text)
            body = text[start:end].strip()
            if len(body) > 40:
                return body[:maxlen]
    return ""


def ifu_from_form(text):
    """Indications for Use from FORM FDA 3881 block."""
    m = re.search(r"Indications for Use\s*\(Describe\)\s*(.*?)(?:Type of Use|Prescription Use|PLEASE DO NOT WRITE|CONTINUE ON A SEPARATE PAGE)", text, flags=re.S | re.I)
    if m:
        body = re.sub(r"\s+", " ", m.group(1)).strip()
        if len(body) > 20:
            return body[:4000]
    return ""


def process(path, ocr=False):
    k = os.path.basename(path)[:-4]
    rec = {"k": k}
    try:
        import pymupdf
        d = pymupdf.open(path)
        pages = [p.get_text() for p in d]
        rec["n_pages"] = len(pages)
        text = "\n".join(pages)
        rec["ocr"] = False
        if ocr and len(text) < OCR_MIN_CHARS_PER_PAGE * max(1, len(pages)):
            pages = ocr_pages(path)
            text = "\n".join(pages)
            rec["ocr"] = True
            rec["ocr_pages"] = len(pages)
    except Exception as e:
        rec["error"] = str(e)[:200]
        return rec
    rec["n_chars"] = len(text)
    # locate 510(k) summary portion (after the decision letter + IFU form)
    ms = list(re.finditer(r"510\s*\(?k\)?\s*summary", text, flags=re.I))
    sum_start = 0
    for m in ms:
        # choose the first occurrence that is at line start-ish (header), skip letter boilerplate
        line_start = text.rfind("\n", 0, m.start()) + 1
        if len(text[line_start:m.start()].strip()) <= 12:
            sum_start = line_start
            break
    summary = text[sum_start:]
    rec["summary_start"] = sum_start
    rec["summary_text"] = re.sub(r"[ \t]+", " ", summary)[:12000]
    rec["ifu_form"] = ifu_from_form(text)
    rec["ifu_sec"] = section_after(summary, r"indications?\s+for\s+use|intended\s+use")
    rec["desc_sec"] = section_after(summary, r"device\s+description|description\s+of\s+(?:the\s+)?device")
    # cited K numbers in order, with predicate/reference labeling by nearest preceding cue word
    cited, labels = [], {}
    for m in K_RE.finditer(summary):
        kk = norm_k(m.group(0))
        if kk == k or kk in labels:
            continue
        window = summary[max(0, m.start() - 700):m.start()].lower()
        ip, ir = window.rfind("predicate"), window.rfind("reference device")
        labels[kk] = "reference" if ir > ip else "predicate"
        cited.append(kk)
    rec["cited"] = cited
    rec["cited_labels"] = labels
    # also any K numbers in the whole doc (letter can mention none; IFU form has own K)
    allk = [norm_k(m.group(0)) for m in K_RE.finditer(text)]
    rec["cited_anywhere"] = [x for x in dict.fromkeys(allk) if x != k]
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--pdf-dir", default=PDFS)
    ap.add_argument("--out", default=OUTF)
    ap.add_argument("--ocr", action="store_true", help="OCR image-only PDFs on the GPU (runs sequentially in-process)")
    ap.add_argument("--redo-low-text", action="store_true",
                    help="drop records in --out whose text was below the OCR threshold and extract them again")
    a = ap.parse_args()
    done, low, keep = set(), set(), []
    if os.path.exists(a.out):
        for line in open(a.out, encoding="utf-8"):
            try:
                r = json.loads(line)
            except Exception:
                continue
            done.add(r["k"])
            npages = max(1, r.get("n_pages") or 1)
            if a.redo_low_text and not r.get("ocr") and r.get("n_chars", 0) < OCR_MIN_CHARS_PER_PAGE * npages:
                low.add(r["k"])
            else:
                keep.append(line)
    if low:
        open(a.out, "w", encoding="utf-8").writelines(keep)
        done -= low
        print(f"{len(low)} low-text records removed for re-extraction", flush=True)
    files = [f for f in sorted(glob.glob(os.path.join(a.pdf_dir, "K*.pdf")), reverse=True) if os.path.basename(f)[:-4] not in done]
    if a.limit:
        files = files[: a.limit]
    print(f"{len(files)} new PDFs to extract ({len(done)} done)", flush=True)
    import time
    t0 = time.time()
    with open(a.out, "a", encoding="utf-8") as out:
        if a.ocr or a.workers == 1:
            pool, it = None, (process(f, ocr=a.ocr) for f in files)
        else:
            pool = Pool(a.workers); it = pool.imap_unordered(process, files, chunksize=16)
        for i, rec in enumerate(it, 1):
            out.write(json.dumps(rec, ensure_ascii=False) + "\n"); out.flush()
            if i % (50 if a.ocr else 1000) == 0:
                print(i, f"{(time.time() - t0) / i:.2f}s/doc", flush=True)
        if pool:
            pool.close(); pool.join()
    print("done", flush=True)


if __name__ == "__main__":
    main()
