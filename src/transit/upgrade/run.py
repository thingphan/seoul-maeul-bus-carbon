"""Reproducible v2 pipeline. Public raw data can be downloaded without card IDs."""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
import importlib.metadata
from .common import ROOT,OUT,write_json,sha256

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--download',action='store_true');parser.add_argument('--report',action='store_true');parser.add_argument('--integrated',action='store_true',help='Recompute full route carbon/access study and integrated report after trip pipeline');args=parser.parse_args()
    if args.download:
        subprocess.run([sys.executable,'-X','utf8','-m','src.transit.upgrade.fetch','--start','2026-07-06','--end','2026-08-09','--workers','2','--aggregates'],cwd=ROOT,check=True)
        from .fetch import fetch_aggregate
        fetch_aggregate('OA-21220','2608')
    import pandas as pd
    expected={d.strftime('%Y%m%d') for d in pd.date_range('2026-07-06','2026-08-09')}
    actual={p.stem.removeprefix('cards_') for p in (OUT/'card_partitions').glob('*.parquet')}
    if not expected.issubset(actual):raise RuntimeError(f'Missing raw service-day partitions: {sorted(expected-actual)}. Use --download.')
    for module in ['demand','audit','validate_sources','geometry','energy','model','policy','robustness','figures']:
        subprocess.run([sys.executable,'-X','utf8','-m',f'src.transit.upgrade.{module}'],cwd=ROOT,check=True)
    files=list((ROOT/'src/transit/upgrade').glob('*.py'))+[ROOT/'config/maeul_upgrade/scenario.json']
    write_json(OUT/'execution_manifest.json',{'python':sys.version,'dependencies':{p:importlib.metadata.version(p) for p in ['numpy','pandas','pyarrow','scipy','scikit-learn','xgboost','matplotlib','reportlab']},'source_sha256':{str(p.relative_to(ROOT)):sha256(p) for p in files},'test_period_reserved':'2026-08-03 through 2026-08-09; July27-Aug2 used during model development'})
    if args.report:subprocess.run([sys.executable,'-X','utf8','-m','src.transit.upgrade.report'],cwd=ROOT,check=True)
    if args.integrated:
        for module in ['integrated','integrated_figures','integrated_report']:
            subprocess.run([sys.executable,'-X','utf8','-m',f'src.transit.upgrade.{module}'],cwd=ROOT,check=True)

if __name__=='__main__':main()
