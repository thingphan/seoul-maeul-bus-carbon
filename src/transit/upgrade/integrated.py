"""Recompute the full carbon/accessibility study without overwriting v1 inputs.

The v2 trip-level validation remains a separate cohort (35 days / 247 routes).
This route-level cohort uses July aggregated passenger-km and excludes its QA
routes. Do not combine the two cohorts' emissions as a single realised effect.
"""
from __future__ import annotations
import importlib.util
import json
import shutil
import sys
import numpy as np
import pandas as pd
from .common import ROOT,BASE,OUT,csv,write_json,sha256

FULL=ROOT/'data/processed/transit_full'

def legacy(name):
    path=ROOT/'src/transit'/name
    spec=importlib.util.spec_from_file_location('integrated_'+name[:2],path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.TRANSIT=FULL
    return module

def main():
    FULL.mkdir(parents=True,exist_ok=True)
    names=['routes.csv','route_links.csv','route_segments.csv','bus_stops.csv','route_demand.csv','route_stops.parquet','grid_population_daytime.csv','grid_walk_access.csv','link_speed_weekday_profile.parquet','route_feasibility.csv']
    input_hash={}
    for name in names:
        source=BASE/name
        shutil.copy2(source,FULL/name);input_hash[name]=sha256(source)
    sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'src/transit'))
    import route_data
    route_data.TRANSIT=FULL;route_data.ROUTE_STOP_CACHE=FULL/'route_stops.parquet';route_data.TRIP_CACHE=FULL/'bus_trip_counts.parquet'
    official=pd.read_parquet(OUT/'official_trip_counts_jul_aug.parquet')
    official.to_parquet(route_data.TRIP_CACHE,index=False)
    energy=pd.read_csv(OUT/'route_hour_energy.csv',dtype={'ROUTE_ID':str})
    mean=energy[energy.hour.between(7,21)].groupby(['ROUTE_ID','vehicle']).kwh_per_km.mean().unstack()
    routes=pd.read_csv(FULL/'routes.csv',dtype={'ROUTE_ID':str})
    reference=routes.merge(mean.reference.rename('kwh_per_km'),on='ROUTE_ID',how='inner')
    reference['is_base']=True
    csv(reference,FULL/'route_energy.csv')
    carbon_module=legacy('05_build_route_carbon_breakeven.py');carbon_module.main()
    carbon=pd.read_csv(FULL/'route_carbon_breakeven.csv',dtype={'ROUTE_ID':str})
    cfg=json.loads((ROOT/'config/maeul_upgrade/scenario.json').read_text(encoding='utf-8'))
    # Use the same grid-side charging boundary for the electric passenger car.
    carbon['ev_car_kwh_per_km']/=cfg['charger_efficiency_assumed']
    carbon['car_전기_g_per_km']=carbon.ev_car_kwh_per_km*cfg['grid_kg_co2_per_kwh']*1000
    shares=carbon_module.car_fleet_shares()
    carbon['car_g_per_km']=sum(share*carbon[f'car_{fuel}_g_per_km'] for fuel,share in shares.items())
    carbon['car_g_per_pkm']=carbon.car_g_per_km/carbon_module.CAR_OCCUPANCY
    carbon['breakeven_load']=carbon.bus_g_per_km/carbon.car_g_per_km*carbon_module.CAR_OCCUPANCY
    carbon['breakeven_trips']=carbon.pkm_per_day/(carbon.breakeven_load*carbon.length_km)
    carbon['bus_vs_car']=carbon.bus_g_per_pkm/carbon.car_g_per_pkm
    carbon['bus_better']=carbon.bus_vs_car.lt(1)
    carbon['co2_saved_kg_per_day']=carbon.pkm_per_day*(carbon.car_g_per_pkm-carbon.bus_g_per_pkm)/1000
    carbon['car_rider_kg']=carbon.avg_trip_km*carbon.car_g_per_pkm/1000
    carbon['added_trip_breakeven_riders']=carbon.added_trip_kg/carbon.car_rider_kg
    carbon['bus_vs_ev_car']=(carbon.bus_kwh_per_km/carbon.avg_load)/(carbon.ev_car_kwh_per_km/carbon_module.CAR_OCCUPANCY)
    csv(carbon,FULL/'route_carbon_breakeven.csv')
    from route_classes import assign_classes,DEPENDENCE
    demand=pd.read_csv(FULL/'route_demand.csv',dtype={'ROUTE_ID':str})
    p=assign_classes(carbon[~carbon.load_qa_flag].merge(demand[['ROUTE_ID',*DEPENDENCE]],on='ROUTE_ID',validate='one_to_one'))
    p['avg_peak_load_per_trip']=p.peak_onboard/p.trips_per_day
    p['van_kwh_per_km']=p.ROUTE_ID.map(mean.small)
    p['van_g_per_km']=p.van_kwh_per_km*cfg['grid_kg_co2_per_kwh']*1000
    p['van_feasible']=p.avg_peak_load_per_trip.le(8) # screening ONLY, verified by trip model later
    convert=p.class_id.eq(2)&p.van_feasible
    vkm=p.trips_per_day*p.length_km
    p['van_co2_cut_kg_per_day']=np.where(convert,(p.bus_ev_g_per_km-p.van_g_per_km)*vkm/1000,0)
    p['van_co2_cut_vs_current_kg_per_day']=np.where(convert,(p.bus_g_per_km-p.van_g_per_km)*vkm/1000,0)
    csv(p,FULL/'route_priority.csv')
    ops=legacy('07_build_route_operations.py');ops.main()
    access=legacy('10_route_removal_access.py');access.main()
    removal=pd.read_csv(FULL/'route_removal_access.csv',dtype={'ROUTE_ID':str})
    hourly=pd.read_csv(FULL/'route_hourly_load.csv',dtype={'ROUTE_ID':str})
    operations=pd.read_csv(FULL/'route_operations.csv',dtype={'ROUTE_ID':str})
    high=removal[removal.class_id.isin([1,2])];low=removal[removal.class_id.isin([3,4])]
    lowhours=hourly[hourly.low_load]
    summary={'routes_total':int(routes.kind.eq('마을버스').sum()),'routes_analyzed':len(p),'excluded_routes':int(carbon.load_qa_flag.sum()),'bus_better':int(p.bus_better.sum()),'bus_better_share':float(p.bus_better.mean()),'bus_g_per_pkm_median':float(p.bus_g_per_pkm.median()),'car_g_per_pkm_median':float(p.car_g_per_pkm.median()),'avg_load_median':float(p.avg_load.median()),'breakeven_load_median':float(p.breakeven_load.median()),'current_t_per_day':float(p.bus_co2_kg_per_day.sum()/1000),'electric_t_per_day':float(p.bus_co2_kg_per_day_all_electric.sum()/1000),'electrification_cut_t_per_day':float((p.bus_co2_kg_per_day-p.bus_co2_kg_per_day_all_electric).sum()/1000),'low_hour_trip_share':float(lowhours.trips.sum()/hourly.trips.sum()),'low_hour_vkm_share':float(lowhours.vkm.sum()/hourly.vkm.sum()),'hour_small_current_cut_t_per_day':float(hourly.van_cut_vs_current_kg.sum()/1000),'hour_small_size_cut_t_per_day':float(hourly.van_cut_kg.sum()/1000),'high_dependence_affected_net_median':float(high.net_elderly_affected.median()),'high_dependence_affected_simple_median':float(high.simple_elderly_affected.median()),'low_dependence_affected_net_median':float(low.net_elderly_affected.median()),'low_dependence_affected_simple_median':float(low.simple_elderly_affected.median()),'new_disconnected_elderly_route_sum_not_population_total':float(removal.net_elderly_disconnected_new.sum()),'class_counts':p['class'].value_counts().sort_index().to_dict(),'screening_high_dep_small_candidates':int((p.class_id.eq(2)&p.van_feasible).sum()),'added_trip_breakeven_median':float(p.added_trip_breakeven_riders.median()),'expansion_candidates':int(operations.carbon_service_candidate.sum()),'input_sha256':input_hash,'notes':['July aggregate cohort versus 35-day trip cohort are different samples.','Bus diesel/CNG are transferred empirical ratios, not new fuel-meter observations.','Mean 07–21 weekday speeds used for route energy; actual fleet vehicle size not surveyed.','Class 2 means relatively low carbon efficiency, not necessarily worse than cars.','Potential scenarios overlap and must not be added.','Car charging loss 0.9 applied consistently with electric bus grid-side consumption.','New disconnected cells counted as impacted; finite additional-time mean excludes them.']}
    links=pd.read_csv(FULL/'route_links.csv',dtype={'LINK_ID':str});links=links[links.kind.eq('마을버스')]
    tiers=pd.read_parquet(FULL/'link_speed_weekday_profile.parquet',columns=['LINK_ID','tier']).drop_duplicates('LINK_ID')
    links=links.merge(tiers,on='LINK_ID',validate='many_to_one')
    summary['speed_tier_matched_length_share']=(links.groupby('tier').LENGTH.sum()/links.LENGTH.sum()).to_dict()
    summary['boardings_per_day_all_routes']=float(demand[demand.kind.eq('마을버스')].boardings_per_day.sum())
    summary['boardings_per_day_analyzed']=float(p.boardings_per_day.sum())
    write_json(FULL/'integrated_summary.json',summary)
    # Passenger occupancy, vehicle size and fuel-factor sensitivity.
    records=[]
    for occupancy in [1.2,1.35,1.82]:
        for bus_factor in [1.,1.28,1.5]:
            for car_factor in [1.,.8]:
                ratio=(p.bus_g_per_pkm*bus_factor)/(p.car_g_per_km*car_factor/occupancy)
                records.append({'car_occupancy':occupancy,'bus_emission_factor':bus_factor,'car_emission_factor':car_factor,'bus_better':int(ratio.lt(1).sum()),'routes':len(p)})
    csv(pd.DataFrame(records),FULL/'carbon_sensitivity.csv')
    for name in names:
        if sha256(BASE/name)!=input_hash[name]:raise AssertionError('Preserved original input modified')
    print(json.dumps({k:v for k,v in summary.items() if k!='input_sha256'},ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
