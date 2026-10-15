"""Fleet-constrained whole-duty replacement, planned before final test dates.

One binary per route/vehicle/day duty and integer installed vehicles per route.
No on-road vehicle swaps, reduced departures, assumed extra passenger capacity,
or pooled vehicles silently moving between depots are permitted.
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from scipy.optimize import milp,Bounds,LinearConstraint
from scipy.sparse import coo_matrix
from .common import ROOT,OUT,csv,write_json

def solve_portfolio(duties,cap):
    d=duties.reset_index(drop=True);routes=sorted(d.ROUTE_ID.unique());rindex={r:i for i,r in enumerate(routes)}
    n=len(d);m=len(routes)
    if n==0: return pd.Series(dtype=int),np.zeros(0,dtype=bool),{'success':True,'objective':0.,'gap':0.}
    c=np.r_[-d.saved_kg.to_numpy(),np.full(m,1e-7)]
    row=[];col=[];value=[];upper=[]
    for (r,day),indices in d.groupby(['ROUTE_ID','service_date']).groups.items():
        k=len(upper)
        for i in indices:row.append(k);col.append(i);value.append(1)
        row.append(k);col.append(n+rindex[r]);value.append(-1);upper.append(0.)
    k=len(upper)
    for i in range(m):row.append(k);col.append(n+i);value.append(1)
    upper.append(float(cap))
    matrix=coo_matrix((value,(row,col)),shape=(len(upper),n+m)).tocsr()
    result=milp(c,integrality=np.ones(n+m),bounds=Bounds(np.zeros(n+m),np.r_[np.ones(n),np.full(m,cap)]),constraints=LinearConstraint(matrix,-np.inf,np.array(upper)),options={'time_limit':60,'mip_rel_gap':.001})
    if result.x is None:raise RuntimeError(f'MILP has no feasible solution: {result.message}')
    allocation=pd.Series(np.rint(result.x[n:]).astype(int),index=routes)
    allocation=allocation[allocation>0]
    chosen=result.x[:n]>.5
    if allocation.sum()>cap:raise AssertionError('Fleet cap violated')
    for (r,day),indices in d.groupby(['ROUTE_ID','service_date']).groups.items():
        if chosen[list(indices)].sum()>allocation.get(r,0):raise AssertionError('Daily route stock violated')
    return allocation,chosen,{'success':bool(result.success),'message':result.message,'objective':float(-result.fun),'gap':float(getattr(result,'mip_gap',0))}

def build_duties(predictions,cfg,energy_override=None):
    energy=pd.read_csv(OUT/'route_hour_energy.csv',dtype={'ROUTE_ID':str}) if energy_override is None else energy_override
    energy=energy.pivot(index=['ROUTE_ID','hour'],columns='vehicle',values='kwh_per_km').reset_index().rename(columns={'reference':'reference_kwh_km','small':'small_kwh_km'})
    distance=pd.read_parquet(OUT/'daily_distance_audit.parquet')[['ROUTE_ID','service_date','dated_length_km','relative_geometry_difference']]
    f=predictions.merge(energy,on=['ROUTE_ID','hour'],how='left',validate='many_to_one').merge(distance,on=['ROUTE_ID','service_date'],how='left',validate='many_to_one')
    f['energy_distance_km']=f[['dated_length_km','length_km']].max(axis=1)
    f['small_kwh']=f.small_kwh_km*f.energy_distance_km
    f['reference_kwh']=f.reference_kwh_km*f.energy_distance_km
    # Same date vehicle operation on multiple routes cannot be assigned twice.
    multi=f.groupby(['service_date','VHCL_ID']).ROUTE_ID.nunique().rename('vehicle_routes').reset_index()
    f=f.merge(multi,on=['service_date','VHCL_ID'],how='left')
    keys=['ROUTE_ID','노선명','VHCL_ID','service_date','split']
    d=f.groupby(keys,observed=True).agg(trips=('departure','size'),pred_upper=('pred_calibrated','max'),pred_unadjusted=('pred_q99','max'),empirical_upper=('empirical_q99','max'),peak_lower=('peak_lower','max'),peak_upper=('peak_upper','max'),usable_share=('usable','mean'),small_kwh=('small_kwh','sum'),reference_kwh=('reference_kwh','sum'),small_kwh_km=('small_kwh_km','mean'),reference_kwh_km=('reference_kwh_km','mean'),geometry_error=('relative_geometry_difference',lambda s:s.abs().max()),vehicle_routes=('vehicle_routes','max'),elder_boardings=('elder_boardings','sum'),boardings=('boardings','sum'),vehicle_km=('energy_distance_km','sum')).reset_index()
    deadhead=2*cfg['depot_distance_km_each_way_assumed']
    d.small_kwh+=deadhead*d.small_kwh_km;d.reference_kwh+=deadhead*d.reference_kwh_km
    # kWh figures are grid-side. Battery capacity check uses battery-side energy.
    d['battery_kwh']=d.small_kwh*cfg['charger_efficiency_assumed']
    d['saved_kwh']=(d.reference_kwh-d.small_kwh).clip(lower=0)
    d['saved_kg']=d.saved_kwh*cfg['grid_kg_co2_per_kwh']
    d['capacity_ok']=d.pred_upper.le(cfg['small_passenger_capacity'])
    d['battery_ok']=d.battery_kwh.le(cfg['small_battery_kwh']*cfg['usable_battery_fraction_assumed']*(1-cfg['battery_daily_reserve_fraction_assumed']))
    topology=pd.read_csv(OUT/'route_topology_validation.csv',dtype={'ROUTE_ID':str})
    d['geometry_ok']=d.ROUTE_ID.map(topology.set_index('ROUTE_ID').topology_ok).fillna(False)
    d['single_route']=d.vehicle_routes.eq(1)
    source=pd.read_csv(OUT/'trip_count_validation.csv',dtype={'ROUTE_ID':str,'service_date':str})
    prior=source[source.service_date.le('20260726')].groupby('ROUTE_ID').tagged_to_official_ratio.median()
    d['source_ratio_prior']=d.ROUTE_ID.map(prior)
    d['source_ok']=d.source_ratio_prior.between(.8,1.2)
    d['eligible']=d.capacity_ok & d.battery_ok & d.geometry_ok & d.single_route & d.source_ok & d.trips.ge(3)
    return f,d

def replay(duties,allocation):
    chosen=pd.Series(False,index=duties.index)
    for (r,day),group in duties[duties.eligible].groupby(['ROUTE_ID','service_date']):
        n=int(allocation.get(r,0))
        chosen.loc[group.nlargest(n,'saved_kg').index]=True
    return chosen

def marginal_greedy(duties,cap):
    """Strong comparator for this separable common-unit fleet-cap special case.

    Diminishing daily top-k gains make marginal allocation optimal in this
    specific model. MILP does not itself create a methodological advantage.
    """
    gains={}
    for route,group in duties.groupby('ROUTE_ID'):
        per_day=[g.saved_kg.sort_values(ascending=False).to_numpy() for _,g in group.groupby('service_date')]
        gains[route]=[sum(v[k] if len(v)>k else 0 for v in per_day) for k in range(max(map(len,per_day)))]
    allocation=pd.Series(0,index=list(gains),dtype=int)
    for _ in range(cap):
        options={r:g[allocation[r]] for r,g in gains.items() if allocation[r]<len(g)}
        if not options:break
        route=max(options,key=options.get)
        if options[route]<=0:break
        allocation[route]+=1
    return allocation[allocation>0]

def main():
    cfg=json.loads((ROOT/'config/maeul_upgrade/scenario.json').read_text(encoding='utf-8'))
    predictions=pd.read_parquet(OUT/'predictions.parquet')
    trips,duties=build_duties(predictions,cfg)
    duties.to_parquet(OUT/'duties.parquet',index=False)
    cal=duties[duties.split.eq('calibration')];test=duties[duties.split.eq('test')]
    eligible=cal[cal.eligible]
    records=[];allocations=[];chosen_frames=[]
    test_days=test.service_date.nunique()
    for cap in cfg['fleet_caps_assumed']:
        allocation,chosen,solver=solve_portfolio(eligible,cap)
        selected=replay(test,allocation);s=test[selected]
        assessment=trips[trips.split.eq('test')].merge(s[['ROUTE_ID','VHCL_ID','service_date']].assign(selected=True),on=['ROUTE_ID','VHCL_ID','service_date'],how='left')
        assessed=assessment[assessment.selected.eq(True)&assessment.usable]
        unsafe=int(assessed.peak_upper.gt(cfg['small_passenger_capacity']).sum())
        verified_count=len(assessed)
        record={'fleet_cap':cap,'installed_vehicles':int(allocation.sum()),'routes':len(allocation),'selected_duty_days':len(s),'selected_trips':int(s.trips.sum()),'selected_vkm_per_day':float(s.vehicle_km.sum()/test_days),'saved_kg_per_day':float(s.saved_kg.sum()/test_days),'electricity_saving_won_per_day':float(s.saved_kwh.sum()*cfg['electricity_won_per_kwh_assumed']/test_days),'assessable_selected_trips':verified_count,'unsafe_assessable_trips':unsafe,'unsafe_rate':unsafe/verified_count if verified_count else None,'unassessable_selected_share':1-verified_count/max(1,float(s.trips.sum())),'elder_boardings_per_day':float(s.elder_boardings.sum()/test_days),'solver':solver}
        record['annualized_electricity_only_break_even_premium_per_vehicle_won']=record['electricity_saving_won_per_day']*365/max(1,allocation.sum())
        records.append(record)
        allocations.append(allocation.rename('vehicles').rename_axis('ROUTE_ID').reset_index().assign(fleet_cap=cap))
        chosen_frames.append(s.assign(fleet_cap=cap))
        print('policy',record,flush=True)
    csv(pd.DataFrame([{k:v for k,v in r.items() if k!='solver'} for r in records]),OUT/'policy_metrics.csv')
    csv(pd.concat(allocations,ignore_index=True),OUT/'fleet_allocation.csv')
    selected=pd.concat(chosen_frames,ignore_index=True)
    selected.to_parquet(OUT/'selected_duties.parquet',index=False)
    # Same candidate set, same stock cap: a simple route-level greedy comparator.
    cap=20
    route_gain=eligible.groupby('ROUTE_ID').saved_kg.sum()
    route_stock=eligible.groupby(['ROUTE_ID','service_date']).size().groupby('ROUTE_ID').max()
    ranking=(route_gain/route_stock).sort_values(ascending=False)
    greedy=pd.Series(dtype=int);remaining=cap
    for r in ranking.index:
        n=int(route_stock[r])
        if n<=remaining:greedy.loc[r]=n;remaining-=n
    greedy_selected=test[replay(test,greedy)]
    marginal=marginal_greedy(eligible,cap)
    marginal_selected=test[replay(test,marginal)]
    _,mip_chosen,_=solve_portfolio(eligible,cap)
    marginal_cal=eligible[replay(eligible,marginal)]
    if not np.isclose(eligible.loc[mip_chosen,'saved_kg'].sum(),marginal_cal.saved_kg.sum(),rtol=.002,atol=.01):raise AssertionError('MILP and exact special-case marginal baseline disagree')
    attrition=[]
    current=pd.Series(True,index=test.index)
    for label,mask in [('Tagged duty-days',pd.Series(True,index=test.index)),('Predicted upper <=16',test.capacity_ok),('Overnight battery + reserve',test.battery_ok),('Dated stop sequence matches',test.geometry_ok),('Source count consistency',test.source_ok),('One route and >=3 tagged trips',test.single_route&test.trips.ge(3))]:
        current &= mask
        attrition.append({'stage':label,'duty_days':int(current.sum()),'trips':int(test.loc[current,'trips'].sum()),'potential_saved_kg_per_day':float(test.loc[current,'saved_kg'].sum()/test_days)})
    csv(pd.DataFrame(attrition),OUT/'constraint_attrition.csv')
    write_json(OUT/'policy_summary.json',{'capacity':cfg['small_passenger_capacity'],'planning_dates':sorted(cal.service_date.unique().tolist()),'test_dates':sorted(test.service_date.unique().tolist()),'metrics':records,'greedy20_saved_kg_per_day':float(greedy_selected.saved_kg.sum()/test_days),'greedy20_installed':int(greedy.sum()),'marginal_greedy20_saved_kg_per_day':float(marginal_selected.saved_kg.sum()/test_days),'marginal_greedy_matches_calibration_objective':True,'attrition':attrition,'limitations':['Tag-derived duty-days omit untagged/empty departures. Fleet requirement is a lower bound until independent operation coverage is verified.','No domestic approval, purchase quote, actual depot or charging infrastructure verified.','Whole-duty substitution preserves recorded departure pattern but road clearance and wheelchair boarding must be field-verified.','Daily energy is modelled and seasonally unvalidated; fixed electric reference is not surveyed current fleet.','Electricity-only break-even premium is not total ownership cost or fiscal carbon-abatement cost.','Daily policy replay uses prior-date updated features. Fleet allocation uses calibration dates only.','With a common-unit stock cap and separable daily gains, marginal greedy is optimal too. No claim of MILP superiority over that strong comparator.']})

if __name__=='__main__': main()
