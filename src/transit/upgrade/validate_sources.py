"""Independent TOPIS section-pass counts: diagnostic, not forced calibration."""
from __future__ import annotations
from zipfile import ZipFile
import pandas as pd
from .common import ROOT,OUT,csv,write_json,route_metadata

def main():
    august=ROOT/'data/raw/maeul_upgrade/aggregates/OA-21220_2608.zip'
    target=OUT/('official_trip_counts_jul_aug.parquet' if august.exists() else 'official_trip_counts.parquet')
    if not target.exists():
        ids=set(route_metadata().ROUTE_ID);parts=[]
        paths=[ROOT/'data/raw/traffic/bus_trips/tpss_sta_route_hturn_202607.zip']
        if august.exists():paths.append(august)
        for path in paths:
            with ZipFile(path) as z:
                for name in z.namelist():
                    if not name.endswith('.csv'):continue
                    for chunk in pd.read_csv(z.open(name),encoding='cp949',dtype={'기준_날짜':str,'노선_ID':str,'정류장_ID':str},chunksize=200000):
                        parts.append(chunk[chunk['노선_ID'].isin(ids)].copy())
        counts=pd.concat(parts,ignore_index=True);counts.to_parquet(target,index=False)
    else:counts=pd.read_parquet(target)
    daily=counts.groupby(['노선_ID','기준_날짜'])['버스운행횟수'].median().rename('official_stop_median_trips').reset_index().rename(columns={'노선_ID':'ROUTE_ID','기준_날짜':'service_date'})
    trips=pd.read_parquet(OUT/'trips.parquet')
    tagged=trips.groupby(['ROUTE_ID','service_date']).size().rename('tagged_trip_keys').reset_index()
    comparison=tagged.merge(daily,on=['ROUTE_ID','service_date'],how='left',validate='one_to_one')
    comparison['tagged_to_official_ratio']=comparison.tagged_trip_keys/comparison.official_stop_median_trips.where(comparison.official_stop_median_trips>0)
    csv(comparison,OUT/'trip_count_validation.csv')
    route=comparison.groupby('ROUTE_ID').agg(tagged_to_official_ratio=('tagged_to_official_ratio','median'),days=('service_date','nunique')).reset_index()
    csv(route,OUT/'route_trip_count_validation.csv')
    write_json(OUT/'source_validation_summary.json',{'route_days':len(comparison),'official_matched_days':int(comparison.official_stop_median_trips.notna().sum()),'ratio_median':float(comparison.tagged_to_official_ratio.median()),'ratio_q10':float(comparison.tagged_to_official_ratio.quantile(.1)),'ratio_q90':float(comparison.tagged_to_official_ratio.quantile(.9)),'routes_outside_20pct':int((~route.tagged_to_official_ratio.between(.8,1.2)).sum()),'note':'Vehicle/departure card keys and median stop-pass counts have different measurement definitions. Ratios outside tolerance are quarantined, not treated as missing empty trips or repaired by multiplying loads.'})

if __name__=='__main__':main()
