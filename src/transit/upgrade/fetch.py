"""Download official public daily card ZIPs, retaining only village-bus fields.

File/package day is NOT service day. Every partition uses STDR_DE in the data.
No card identifier is loaded. Vehicle IDs remain in git-ignored local partitions.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import shutil
import time
import urllib.parse
import urllib.request
from zipfile import ZipFile
import pandas as pd
from .common import RAW,OUT,ROOT,sha256,write_json,route_metadata

PAGE='https://t-data.seoul.go.kr/dataprovide/trafficdataviewfile.do?data_id=20006'
FIELDS=['ROUTE_ID','VHCL_ID','TRIP_DEPR_DTM','RIDE_DTM','ALGH_DTM','RIDE_BSST_ID','ALGH_BSST_ID','PSNG_NUM','USER_DVS_CD','TRNS_MNS_CD','STDR_DE']

def request(url,data=None):
    return urllib.request.Request(url,data=data,headers={'User-Agent':'Mozilla/5.0','Referer':PAGE})

def download(url,target:Path,max_bytes=600_000_000,data=None):
    if target.exists(): return
    partial=target.with_suffix(target.suffix+'.part')
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request(url,data),timeout=60) as response, partial.open('wb') as handle:
                total=0
                first=response.read(1024*1024)
                if not first.startswith(b'PK'): raise ValueError('Official response was not ZIP')
                for block in [first]: handle.write(block);total+=len(block)
                for block in iter(lambda:response.read(1024*1024),b''):
                    total+=len(block)
                    if total>max_bytes: raise ValueError('ZIP exceeds configured size cap')
                    handle.write(block)
            with ZipFile(partial) as z:
                if not z.namelist(): raise ValueError('Empty ZIP')
            partial.replace(target)
            return
        except Exception:
            if attempt==2: raise
            time.sleep(2+attempt)

def process_package(file_id,package_day,ids):
    metadata_path=OUT/'fetch_manifest'/f'{package_day}.json'
    if metadata_path.exists():
        import json
        return json.loads(metadata_path.read_text(encoding='utf-8'))
    name=f'TAIMS_{package_day}.zip'
    target=RAW/'cards'/name
    target.parent.mkdir(parents=True,exist_ok=True)
    earlier=ROOT/'output'/'project_upgrade_20261014'/'data_samples'/name
    if earlier.exists() and not target.exists(): shutil.copy2(earlier,target)
    url=f'https://t-data.seoul.go.kr/dataprovide/download.do?id={file_id}'
    download(url,target)
    parts=[];total=0;dates=set()
    with ZipFile(target) as z:
        with z.open('VW_KSCC_DX_CARD.csv') as handle:
            for frame in pd.read_csv(handle,dtype=str,usecols=FIELDS,chunksize=250000,encoding='utf-8-sig'):
                total+=len(frame)
                dates.update(frame.STDR_DE.dropna().unique())
                # Local route membership is supplemented by the official Seoul village type.
                keep=frame.ROUTE_ID.isin(ids) | frame.TRNS_MNS_CD.isin(['105','151'])
                parts.append(frame.loc[keep].copy())
        for member,stem in [('VW_KSCC_DO_CARD.csv','route_master'),('VW_KSCC_DH_CARD.csv','stop_master')]:
            frame=pd.read_csv(z.open(member),dtype=str,encoding='utf-8-sig')
            frame=frame[frame.ROUTE_ID.isin(ids)].copy()
            path=OUT/'masters'/f'{stem}_package_{package_day}.parquet'
            path.parent.mkdir(parents=True,exist_ok=True)
            frame.to_parquet(path,index=False)
    if len(dates)!=1: raise ValueError(f'Package {package_day}: multiple service dates {sorted(dates)}')
    service_day=next(iter(dates))
    village=pd.concat(parts,ignore_index=True)
    for col in ['PSNG_NUM']:
        village[col]=pd.to_numeric(village[col],errors='coerce').astype('float32')
    partition=OUT/'card_partitions'/f'cards_{service_day}.parquet'
    partition.parent.mkdir(parents=True,exist_ok=True)
    if partition.exists(): raise ValueError(f'Duplicate actual service day {service_day}, check packages')
    village.to_parquet(partition,index=False,compression='zstd')
    meta={'source':url,'file_name':name,'package_day':package_day,'service_day':service_day,'zip_bytes':target.stat().st_size,'sha256':sha256(target),'all_rows':total,'village_rows':len(village),'local_matched_routes':int(village.loc[village.ROUTE_ID.isin(ids),'ROUTE_ID'].nunique()),'passenger_boardings':float(village.PSNG_NUM.sum()),'columns':FIELDS}
    write_json(metadata_path,meta)
    print(f'{package_day} -> service {service_day}: {len(village):,} village tags',flush=True)
    return meta

def fetch_aggregate(inf_id,seq='2607'):
    target=RAW/'aggregates'/f'{inf_id}_{seq}.zip'
    target.parent.mkdir(parents=True,exist_ok=True)
    if inf_id=='OA-21220' and seq=='2607':
        old=ROOT/'data'/'raw'/'traffic'/'bus_trips'/'tpss_sta_route_hturn_202607.zip'
        if old.exists() and not target.exists(): shutil.copy2(old,target)
    url='https://datafile.seoul.go.kr/bigfile/iot/inf/nio_download.do?&useCache=false'
    data=urllib.parse.urlencode({'infId':inf_id,'seqNo':'','seq':seq,'infSeq':'3'}).encode()
    download(url,target,max_bytes=250_000_000,data=data)
    write_json(target.with_suffix('.json'),{'dataset':inf_id,'sequence':seq,'url':url,'bytes':target.stat().st_size,'sha256':sha256(target)})
    print(f'aggregate {inf_id}: {target.stat().st_size/1e6:.1f}MB',flush=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start',default='2026-07-06')
    parser.add_argument('--end',default='2026-08-02')
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--aggregates',action='store_true')
    args=parser.parse_args()
    with urllib.request.urlopen(request(PAGE),timeout=30) as response:
        html=response.read().decode('utf-8')
    page_path=RAW/'card_download_page.html';page_path.write_text(html,encoding='utf-8')
    options={day:file_id for file_id,day in re.findall(r'<option\s+value="(\d+)">TAIMS_(\d{8})\.zip</option>',html)}
    # Select package candidates using the observed 4-day publication offset;
    # inspect STDR_DE to validate it, never label observations by this assumption.
    days=pd.date_range(args.start,args.end)
    candidates=[(options[(d+pd.Timedelta(days=4)).strftime('%Y%m%d')],(d+pd.Timedelta(days=4)).strftime('%Y%m%d')) for d in days]
    ids=set(route_metadata().ROUTE_ID)
    failures=[];records=[]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures={executor.submit(process_package,*item,ids):item for item in candidates}
        for future in as_completed(futures):
            try: records.append(future.result())
            except Exception as exc:
                failures.append({'candidate':futures[future],'error':str(exc)})
                print('FETCH FAILURE',futures[future],str(exc),flush=True)
    write_json(OUT/'fetch_summary.json',{'requested_start':args.start,'requested_end':args.end,'packages':sorted(records,key=lambda r:r['service_day']),'failures':failures})
    actual={r['service_day'] for r in records}
    expected={d.strftime('%Y%m%d') for d in days}
    if actual!=expected: raise RuntimeError(f'Service-day coverage mismatch: missing={sorted(expected-actual)}, extra={sorted(actual-expected)}')
    if args.aggregates:
        for inf_id in ['OA-21218','OA-21217','OA-21220']: fetch_aggregate(inf_id)
    if failures: raise RuntimeError('Some public files could not be downloaded')

if __name__=='__main__': main()
