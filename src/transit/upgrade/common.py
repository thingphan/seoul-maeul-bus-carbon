from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'data' / 'processed' / 'transit'
RAW = ROOT / 'data' / 'raw' / 'maeul_upgrade'
OUT = ROOT / 'data' / 'processed' / 'transit_v2'
REPORT = ROOT / 'output' / 'maeul_upgrade'
for folder in (RAW, OUT, REPORT):
    folder.mkdir(parents=True, exist_ok=True)

def write_json(path: Path, value):
    def convert(obj):
        if isinstance(obj, np.generic): return obj.item()
        if isinstance(obj, Path): return str(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        raise TypeError(type(obj).__name__)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=convert, allow_nan=False), encoding='utf-8')

def sha256(path: Path):
    digest=hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(4*1024*1024),b''): digest.update(block)
    return digest.hexdigest()

def route_metadata():
    routes=pd.read_csv(BASE/'routes.csv',dtype={'ROUTE_ID':str})
    return routes[routes.kind=='마을버스'].drop_duplicates('ROUTE_ID')

def csv(frame: pd.DataFrame,path: Path):
    path.parent.mkdir(parents=True,exist_ok=True)
    frame.to_csv(path,index=False,encoding='utf-8-sig')
