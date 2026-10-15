"""Validate reused route topology with dated stop masters, not distance labels."""
from __future__ import annotations
import json
import pandas as pd
from src.transit.route_data import load_route_stops
from .common import OUT,csv,write_json,route_metadata

def main():
    existing=load_route_stops()
    reference={r:g.sort_values('순번').NODE_ID.astype(str).tolist() for r,g in existing.groupby('ROUTE_ID')}
    records=[]
    # Freeze topology QA from pre-test calibration dates only.
    for path in sorted((OUT/'fetch_manifest').glob('*.json')):
        meta=json.loads(path.read_text(encoding='utf-8'))
        if not '20260720'<=meta['service_day']<='20260726':continue
        dated=pd.read_parquet(OUT/'masters'/f"stop_master_package_{meta['package_day']}.parquet")
        dated['order']=pd.to_numeric(dated.BSST_ORDER,errors='coerce')
        for r,g in dated.groupby('ROUTE_ID'):
            a=reference.get(r,[]);b=g.sort_values('order').BSST_ID.astype(str).tolist()
            records.append({'ROUTE_ID':r,'service_date':meta['service_day'],'exact_order_match':a==b,'stop_set_jaccard':len(set(a)&set(b))/max(1,len(set(a)|set(b))),'reference_stops':len(a),'dated_stops':len(b)})
    daily=pd.DataFrame(records);summary=daily.groupby('ROUTE_ID').agg(exact_order_match=('exact_order_match','all'),matched_days=('service_date','nunique'),min_stop_set_jaccard=('stop_set_jaccard','min')).reset_index()
    summary['topology_ok']=summary.exact_order_match & summary.matched_days.eq(7)
    csv(summary,OUT/'route_topology_validation.csv')
    write_json(OUT/'topology_summary.json',{'routes':len(summary),'exact_same_order_all_7days':int(summary.topology_ok.sum()),'note':'Same ordered stops do not prove unchanged historical street path. Reuse of API path and road profile remains an explicit counterfactual assumption. Energy distance takes the larger of API and dated master distance.'})

if __name__=='__main__':main()
