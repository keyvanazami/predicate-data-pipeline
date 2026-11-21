#!/bin/bash
cd /e/fdasearch
while ! grep -q "^done" data/pdfs_download.log; do sleep 30; done
python -I scripts/download_pdfs.py --min-year 2010 --max-year 2014 --workers 6 > data/pdfs_download_2010.log 2>&1
