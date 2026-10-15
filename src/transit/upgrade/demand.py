"""Reconstruct tagged passenger intervals, preserving uncertainty and date provenance.

These are smartcard-derived loads, not automatic passenger counter measurements.
Minute rounding and missing alights produce an interval, not an exact peak.
"""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
from .common import OUT, BASE, csv, write_json, route_metadata

KEYS=['ROUTE_ID','VHCL_ID','TRIP_DEPR_DTM']

def reconstruct(frame: pd.DataFrame):
    f=frame.copy()
    f['weight']=pd.to_numeric(f.PSNG_NUM,errors='coerce').fillna(0).clip(lower=0)
    for column in ['TRIP_DEPR_DTM','RIDE_DTM','ALGH_DTM']:
        f[column+'_ts']=pd.to_datetime(f[column],format='%Y%m%d%H%M%S',errors='coerce')
    valid_key=f.TRIP_DEPR_DTM_ts.notna() & f.VHCL_ID.notna() & f.ROUTE_ID.notna()
    dropped=int((~valid_key).sum());f=f[valid_key].copy()
    f['trip_id']=pd.factorize(pd.MultiIndex.from_frame(f[KEYS]))[0]
    elapsed=(f.RIDE_DTM_ts-f.TRIP_DEPR_DTM_ts).dt.total_seconds()
    board_ok=f.RIDE_DTM_ts.notna() & elapsed.between(0,21600)
    duration=(f.ALGH_DTM_ts-f.RIDE_DTM_ts).dt.total_seconds()
    pair_ok=board_ok & f.ALGH_DTM_ts.notna() & duration.between(0,21600) & ((f.ALGH_DTM_ts-f.TRIP_DEPR_DTM_ts).dt.total_seconds()<=21600)
    f['pair_ok']=pair_ok
    f['uncertain_weight']=f.weight.where(~pair_ok,0)
    f['bad_board_weight']=f.weight.where(~board_ok,0)
    f['elder_weight']=f.weight.where(f.USER_DVS_CD.isin(['06','17']),0)
    f['disability_weight']=f.weight.where(f.USER_DVS_CD.eq('07'),0)
    base=f.groupby('trip_id',sort=False).agg(ROUTE_ID=('ROUTE_ID','first'),VHCL_ID=('VHCL_ID','first'),departure=('TRIP_DEPR_DTM_ts','first'),boardings=('weight','sum'),uncertain_passengers=('uncertain_weight','sum'),bad_board_passengers=('bad_board_weight','sum'),elder_boardings=('elder_weight','sum'),disability_boardings=('disability_weight','sum'),observed_end=('ALGH_DTM_ts','max'))
    good=f[pair_ok]
    boards=good[['trip_id','RIDE_DTM_ts','weight']].rename(columns={'RIDE_DTM_ts':'time'})
    alights=good[['trip_id','ALGH_DTM_ts','weight']].rename(columns={'ALGH_DTM_ts':'time'})
    boards['delta']=boards.weight;alights['delta']=-alights.weight
    # Net event gives a lower peak at minute boundaries. Board-before-alight
    # within a timestamp gives a conservative rounding upper peak.
    events=pd.concat([boards,alights],ignore_index=True)
    if len(events):
        net=events.groupby(['trip_id','time'],sort=True).delta.sum().reset_index()
        net['load']=net.groupby('trip_id').delta.cumsum()
        lower=net.groupby('trip_id').load.max().clip(lower=0)
        events['phase']=np.where(events.delta>0,0,1)
        events=events.sort_values(['trip_id','time','phase'],kind='stable')
        events['load']=events.groupby('trip_id').delta.cumsum()
        rounded=events.groupby('trip_id').load.max().clip(lower=0)
        base['peak_lower']=lower;base['peak_rounded_upper']=rounded
    else:
        base['peak_lower']=0.;base['peak_rounded_upper']=0.
    base[['peak_lower','peak_rounded_upper']]=base[['peak_lower','peak_rounded_upper']].fillna(0)
    base['peak_upper']=base.peak_rounded_upper+base.uncertain_passengers
    base['uncertain_share']=base.uncertain_passengers/base.boardings.clip(lower=1)
    base['duration_observed_min']=(base.observed_end-base.departure).dt.total_seconds()/60
    base['service_date']=base.departure.dt.strftime('%Y%m%d')
    base['hour']=base.departure.dt.hour
    base['weekday']=base.departure.dt.dayofweek
    same_calendar_day=base.observed_end.dt.normalize().eq(base.departure.dt.normalize())
    base['usable']=base.bad_board_passengers.eq(0) & base.uncertain_share.le(.05) & base.duration_observed_min.between(2,180) & same_calendar_day
    return base.reset_index(drop=True),{'rows':len(frame),'dropped_invalid_trip_key':dropped,'complete_pair_rows':int(pair_ok.sum()),'invalid_board_rows':int((~board_ok).sum()),'trip_count':len(base),'usable_trip_count':int(base.usable.sum())}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--refresh',action='store_true');args=parser.parse_args()
    target=OUT/'trip_partitions_v2';target.mkdir(exist_ok=True)
    departure_parts=OUT/'departure_parts';departure_parts.mkdir(exist_ok=True)
    records=[]
    ids=set(route_metadata().ROUTE_ID)
    for path in sorted((OUT/'card_partitions').glob('*.parquet')):
        source_day=path.stem.removeprefix('cards_')
        marker=departure_parts/f'indexed_{source_day}.json'
        if marker.exists() and not args.refresh:continue
        f=pd.read_parquet(path);f=f[f.ROUTE_ID.isin(ids)]
        departure_day=f.TRIP_DEPR_DTM.str.slice(0,8)
        valid_day=departure_day.str.fullmatch(r'20\d{6}',na=False)
        for day,part in f[valid_day].groupby(departure_day[valid_day]):
            part.to_parquet(departure_parts/f'{day}__{source_day}.parquet',index=False,compression='zstd')
        write_json(marker,{'source_day':source_day,'rows':len(f),'invalid_departure_date_rows':int((~valid_day).sum())})
    # The same vehicle/departure can have tags in neighbouring service-day files.
    # Assemble by actual departure day before grouping to avoid duplicate trips.
    days=sorted({p.stem.split('__')[0] for p in departure_parts.glob('*.parquet')})
    for day in days:
        inputs=sorted(departure_parts.glob(f'{day}__*.parquet'))
        dest=target/f'trips_{day}.parquet'
        meta=dest.with_suffix('.json')
        if dest.exists() and not args.refresh and all(p.stat().st_mtime<=dest.stat().st_mtime for p in inputs):continue
        f=pd.concat([pd.read_parquet(p) for p in inputs],ignore_index=True)
        trips,qa=reconstruct(f)
        trips.to_parquet(dest,index=False,compression='zstd');write_json(meta,qa)
        print('departure day',day,qa,flush=True)
    paths=sorted(target.glob('*.parquet'))
    if not paths: raise RuntimeError('No card partitions; run fetch first')
    combined=pd.concat([pd.read_parquet(p) for p in paths],ignore_index=True)
    source_days=sorted(p.stem.removeprefix('cards_') for p in (OUT/'card_partitions').glob('*.parquet'))
    in_window=combined.service_date.between(source_days[0],source_days[-1])
    outside_window=int((~in_window).sum())
    combined=combined[in_window].copy()
    # Enforce the availability boundary on cached older partitions as well.
    # A trip ending after midnight is not a completed previous-day outcome.
    combined['usable'] &= combined.observed_end.dt.normalize().eq(combined.departure.dt.normalize())
    routes=route_metadata()[['ROUTE_ID','노선명','length_km','n_stops']]
    combined=combined.merge(routes,on='ROUTE_ID',how='left',validate='many_to_one')
    combined['split']=np.select([combined.service_date.le('20260719'),combined.service_date.le('20260726'),combined.service_date.le('20260802')],['train','calibration','development'],default='test')
    combined.to_parquet(OUT/'trips.parquet',index=False,compression='zstd')
    summary=combined.groupby(['ROUTE_ID','노선명']).agg(trips=('peak_upper','size'),mean_peak_lower=('peak_lower','mean'),p95_peak_upper=('peak_upper',lambda x:x.quantile(.95)),p99_peak_upper=('peak_upper',lambda x:x.quantile(.99)),max_peak_upper=('peak_upper','max'),elder_boardings=('elder_boardings','sum'),boardings=('boardings','sum'),usable_share=('usable','mean')).reset_index()
    csv(summary,OUT/'route_peak_summary.csv')
    write_json(OUT/'demand_summary.json',{'dates':sorted(combined.service_date.unique().tolist()),'routes':int(combined.ROUTE_ID.nunique()),'trips':len(combined),'departure_keys_outside_downloaded_window':outside_window,'boardings':float(combined.boardings.sum()),'usable_trips':int(combined.usable.sum()),'split_counts':combined.groupby('split').size().to_dict(),'interval_note':'Tagged intervals only; missing pairs and same-minute event order bounded; untagged passengers not observed.'})

if __name__=='__main__': main()
