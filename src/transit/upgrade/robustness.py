"""Prespecified physical/operational stress scenarios, not confidence intervals."""
from __future__ import annotations
import json
import pandas as pd
from .common import ROOT,OUT,csv,write_json
from .energy import build_table
from .policy import build_duties,solve_portfolio,replay

def main():
    base=json.loads((ROOT/'config/maeul_upgrade/scenario.json').read_text(encoding='utf-8'))
    predictions=pd.read_parquet(OUT/'predictions.parquet')
    scenarios=[('base',{},0),('capacity_buffer_13',{'small_passenger_capacity':13},0),('M3_wheelchair_19',{'small_passenger_capacity':19,'small_operating_mass_kg':5200},0),('strong_cooling',{'small_aux_kw':5,'reference_aux_kw':8},0),('no_stop_regeneration',{'regenerative_efficiency_assumed':0},0),('road_speed_minus5',{},-5),('road_speed_plus5',{},5),('usable_battery_70pct',{'usable_battery_fraction_assumed':.7},0),('depot_8km_each_way',{'depot_distance_km_each_way_assumed':8},0)]
    records=[]
    for name,changes,delta in scenarios:
        cfg=base|changes;energy,_=build_table(cfg,delta)
        trips,duties=build_duties(predictions,cfg,energy)
        cal=duties[duties.split.eq('calibration')&duties.eligible]
        allocation,_,solver=solve_portfolio(cal,20)
        test=duties[duties.split.eq('test')];chosen=test[replay(test,allocation)]
        days=test.service_date.nunique()
        records.append({'scenario':name,'capacity':cfg['small_passenger_capacity'],'installed':int(allocation.sum()),'routes':len(allocation),'saved_kg_per_day':float(chosen.saved_kg.sum()/days),'electricity_saving_won_per_day':float(chosen.saved_kwh.sum()*cfg['electricity_won_per_kwh_assumed']/days),'selected_trips':int(chosen.trips.sum()),'solver_gap':solver['gap']})
        print('stress',records[-1],flush=True)
    csv(pd.DataFrame(records),OUT/'robustness_metrics.csv')
    write_json(OUT/'robustness_summary.json',{'scenarios':records,'note':'Deterministic parameter scenarios, not sampling confidence intervals or observed winter performance. M3 has different capacity and 5.2t operating-mass assumption; domestic approval still unverified.'})

if __name__=='__main__':main()
