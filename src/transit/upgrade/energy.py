"""Auditable same-electricity-reference size effect with explicit stop cycles."""
from __future__ import annotations
import json
from dataclasses import replace
import numpy as np
import pandas as pd
from src.energy.ev_calibration import calibrate_county_electric
from src.energy.link_energy import VTCPEMParameters,calculate_link_energy
from src.transit.route_data import load_route_link_hours
from .common import ROOT,OUT,csv,write_json,route_metadata

def stop_energy_kwh(mass_kg,speed_kmh,stops,efficiency,regen=.5,aux_kw=3,dwell_seconds=15):
    kinetic=.5*mass_kg*(speed_kmh/3.6)**2/3_600_000
    return stops*(kinetic*(1/efficiency-regen)+aux_kw*dwell_seconds/3600)

def build_table(cfg,speed_delta=0):
    frame=load_route_link_hours('마을버스')
    frame['speed_kmh']=(frame.speed_kmh+speed_delta).clip(lower=3)
    county,_=calibrate_county_electric()
    routes=route_metadata()[['ROUTE_ID','노선명','length_km','n_stops']]
    rows=[]
    for kind,mass,aux,area,cd in [('reference',cfg['reference_operating_mass_kg'],cfg['reference_aux_kw'],county.frontal_area_m2,county.drag_coefficient),('small',cfg['small_operating_mass_kg'],cfg['small_aux_kw'],cfg['small_frontal_area_m2'],cfg['small_drag_coefficient_assumed'])]:
        params=replace(VTCPEMParameters(),mass_kg=mass,auxiliary_power_w=aux*1000,frontal_area_m2=area,drag_coefficient=cd)
        out=calculate_link_energy(frame[['LINK_ID','LENGTH','speed_kmh','slope_pct']],params)
        temp=frame[['ROUTE_ID','시간코드','LENGTH']].copy();temp['kwh']=out.energy_ev_kwh.to_numpy()
        grouped=temp.groupby(['ROUTE_ID','시간코드']).agg(kwh=('kwh','sum'),metres=('LENGTH','sum')).reset_index().merge(routes,on='ROUTE_ID',how='left')
        grouped['cruise_kwh_per_km']=grouped.kwh/(grouped.metres/1000)
        grouped['stop_kwh_per_km']=stop_energy_kwh(mass,cfg['stop_target_speed_kmh_assumed'],grouped.n_stops*cfg['stop_fraction_assumed'],params.total_efficiency,cfg['regenerative_efficiency_assumed'],aux,cfg['dwell_seconds_assumed'])/grouped.length_km
        grouped['kwh_per_km']=(grouped.cruise_kwh_per_km+grouped.stop_kwh_per_km)/cfg['charger_efficiency_assumed']
        grouped['vehicle']=kind;grouped['hour']=grouped['시간코드']-1
        rows.append(grouped[['ROUTE_ID','hour','vehicle','cruise_kwh_per_km','stop_kwh_per_km','kwh_per_km']])
    energy=pd.concat(rows,ignore_index=True)
    # Outside 07–21, use nearest available hour; retain the extrapolation flag.
    all_hours=[]
    for hour in range(24):
        source=min(21,max(7,hour))
        tmp=energy[energy.hour.eq(source)].copy();tmp['hour']=hour;tmp['speed_extrapolated']=hour!=source;all_hours.append(tmp)
    return pd.concat(all_hours,ignore_index=True),county

def main():
    cfg=json.loads((ROOT/'config/maeul_upgrade/scenario.json').read_text(encoding='utf-8'))
    energy,county=build_table(cfg)
    csv(energy,OUT/'route_hour_energy.csv')
    write_json(OUT/'energy_summary.json',{'county_fitted_cd':county.drag_coefficient,'unidentified_extra_power_w':0,'scenario':cfg,'note':'Energy is modelled, not meter-validated. Stop-cycle recovery does not recover downhill link energy. Weekday road speed profile reused on all replay dates; outside source hours nearest-hour extrapolation. Size effect against a common electric reference excludes electrification gains and manufacturing emissions.'})

if __name__=='__main__': main()
