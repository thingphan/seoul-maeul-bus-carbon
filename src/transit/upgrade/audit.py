"""Same-day route master distance audit and directly matched passenger-km."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from .common import OUT, csv, write_json, route_metadata

def main():
    daily=[];distance=[]
    for manifest in sorted((OUT/'fetch_manifest').glob('*.json')):
        meta=json.loads(manifest.read_text(encoding='utf-8'));day=meta['service_day'];package=meta['package_day']
        destination=OUT/'audit_partitions'/f'audit_{day}.parquet';destination.parent.mkdir(exist_ok=True)
        if destination.exists(): daily.append(pd.read_parquet(destination));continue
        cards=pd.read_parquet(OUT/'card_partitions'/f'cards_{day}.parquet')
        stops=pd.read_parquet(OUT/'masters'/f'stop_master_package_{package}.parquet')
        stops['metres']=pd.to_numeric(stops.BSST_SUM_DISTANCE,errors='coerce')
        unique=stops.groupby(['ROUTE_ID','BSST_ID']).metres.agg(['nunique','first']).reset_index()
        unique=unique[unique['nunique'].eq(1)]
        pairs=cards.merge(unique[['ROUTE_ID','BSST_ID','first']].rename(columns={'BSST_ID':'RIDE_BSST_ID','first':'board_m'}),on=['ROUTE_ID','RIDE_BSST_ID'],how='left',validate='many_to_one')
        pairs=pairs.merge(unique[['ROUTE_ID','BSST_ID','first']].rename(columns={'BSST_ID':'ALGH_BSST_ID','first':'alight_m'}),on=['ROUTE_ID','ALGH_BSST_ID'],how='left',validate='many_to_one')
        pairs['distance_km']=(pairs.alight_m-pairs.board_m)/1000
        pairs['weight']=pd.to_numeric(pairs.PSNG_NUM,errors='coerce').fillna(0)
        valid=pairs.distance_km.between(0.001,40)
        pairs['pkm']=np.where(valid,pairs.distance_km*pairs.weight,0)
        pairs['matched_weight']=np.where(valid,pairs.weight,0)
        result=pairs.groupby('ROUTE_ID').agg(boardings=('weight','sum'),distance_matched_boardings=('matched_weight','sum'),matched_pkm=('pkm','sum')).reset_index()
        master=pd.read_parquet(OUT/'masters'/f'route_master_package_{package}.parquet')
        master['dated_length_km']=pd.to_numeric(master.ROUTE_DISTANCE,errors='coerce')/1000
        master=master.groupby('ROUTE_ID').dated_length_km.first().reset_index()
        result=result.merge(master,on='ROUTE_ID',how='left');result['service_date']=day
        result.to_parquet(destination,index=False);daily.append(result)
        print('distance audit',day,flush=True)
    f=pd.concat(daily,ignore_index=True).merge(route_metadata()[['ROUTE_ID','노선명','length_km']],on='ROUTE_ID',how='inner')
    trips=pd.read_parquet(OUT/'trips.parquet')
    counts=trips.groupby(['ROUTE_ID','service_date']).size().rename('tagged_trips').reset_index()
    f=f.merge(counts,on=['ROUTE_ID','service_date'],how='left')
    f['distance_match_share']=f.distance_matched_boardings/f.boardings.clip(lower=1)
    f['relative_geometry_difference']=f.length_km/f.dated_length_km-1
    f['matched_average_load']=f.matched_pkm/(f.tagged_trips*f.dated_length_km)
    f.to_parquet(OUT/'daily_distance_audit.parquet',index=False)
    csv(f.groupby(['ROUTE_ID','노선명']).agg(dated_length_km=('dated_length_km','median'),geometry_difference=('relative_geometry_difference','median'),distance_match_share=('distance_match_share','mean'),matched_average_load=('matched_average_load','mean')).reset_index(),OUT/'route_distance_audit.csv')
    write_json(OUT/'distance_summary.json',{'matched_boarding_share':float(f.distance_matched_boardings.sum()/f.boardings.sum()),'routes_geometry_difference_over_10pct':int((f.groupby('ROUTE_ID').relative_geometry_difference.median().abs()>.1).sum()),'note':'Passenger-km is directly matched only for unique stop occurrences and forward cumulative distances. Unmatched/circular/ambiguous pairs are excluded, not silently imputed. Dated distance is not a surveyed GPS distance.'})

if __name__=='__main__': main()
