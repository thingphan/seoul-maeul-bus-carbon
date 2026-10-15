"""Chronological quantile prediction and upper-tail policy validation."""
from __future__ import annotations
import numpy as np
import pandas as pd
from xgboost import XGBRegressor,DMatrix
from .common import OUT, BASE, csv, write_json

def calibration_offset(residuals,alpha=.01):
    """Finite-sample upper residual order statistic, without IID time guarantees."""
    values=np.sort(np.asarray(residuals,dtype=float))
    if not len(values): raise ValueError('Empty calibration set')
    rank=min(len(values),int(np.ceil((len(values)+1)*(1-alpha))))
    return max(0.,float(values[rank-1]))

def features(trips,priority=None):
    f=trips.copy().sort_values(['service_date','ROUTE_ID','hour','departure'])
    # Earlier dates only: each day's lagged values are fixed before that day.
    daily=f[f.usable].groupby(['ROUTE_ID','hour','service_date']).peak_upper.agg(mean='mean',q95=lambda x:x.quantile(.95),count='size').reset_index()
    all_days=pd.DataFrame({'service_date':sorted(f.service_date.unique())})
    grid=f[['ROUTE_ID','hour']].drop_duplicates().merge(all_days,how='cross').merge(daily,on=['ROUTE_ID','hour','service_date'],how='left')
    grid=grid.sort_values(['ROUTE_ID','hour','service_date'])
    for col in ['mean','q95','count']:
        grid['lag7_'+col]=grid.groupby(['ROUTE_ID','hour'])[col].transform(lambda s:s.shift(1).rolling(7,min_periods=1).mean())
    f=f.merge(grid[['ROUTE_ID','hour','service_date','lag7_mean','lag7_q95','lag7_count']],on=['ROUTE_ID','hour','service_date'],how='left',validate='many_to_one')
    if priority is None: priority=pd.read_csv(BASE/'route_priority.csv',dtype={'ROUTE_ID':str})
    columns=['catchment_elderly_share','no_alt_boarding_share','dependence_score']
    f=f.merge(priority[['ROUTE_ID']+columns],on='ROUTE_ID',how='left',validate='many_to_one')
    f['route_category']=f.ROUTE_ID.astype('category')
    f['hour_sin']=np.sin(2*np.pi*f.hour/24);f['hour_cos']=np.cos(2*np.pi*f.hour/24)
    f['weekend']=(f.weekday>=5).astype(int)
    # Official 2026 calendar: Constitution Day was restored as a public holiday.
    f['public_holiday']=f.service_date.eq('20260717').astype(int)
    cols=['route_category','hour','weekday','weekend','public_holiday','hour_sin','hour_cos','length_km','n_stops','lag7_mean','lag7_q95','lag7_count']+columns
    return f,cols

def metrics(frame,name,pred,capacity=16):
    y=frame.peak_upper.to_numpy();point=np.asarray(pred)
    eligible=point<=capacity
    return {'model':name,'n_test':len(y),'upper_coverage':float(np.mean(y<=point)),'mean_upper':float(np.mean(point)),'pinball_99':float(np.mean(np.maximum(.99*(y-point),-.01*(y-point)))),'convertible_trip_share':float(np.mean(eligible)),'unsafe_selected_rate':float(np.mean(y[eligible]>capacity)) if eligible.any() else None,'unsafe_selected_trips':int(np.sum(eligible&(y>capacity))),'selected_trips':int(eligible.sum())}

def main():
    trips=pd.read_parquet(OUT/'trips.parquet')
    if not {'train','calibration','test'}.issubset(set(trips.split)): raise RuntimeError('Require complete chronological training/calibration/test periods')
    f,columns=features(trips)
    train=f[(f.split=='train')&f.usable];cal=f[(f.split=='calibration')&f.usable];test=f[(f.split=='test')&f.usable]
    if train.service_date.max()>=cal.service_date.min() or cal.service_date.max()>=test.service_date.min(): raise AssertionError('Date leakage')
    model_dir=OUT/'models';model_dir.mkdir(exist_ok=True)
    # Heteroskedastic target scaling is based on strictly earlier demand history.
    # The fallback is a single constant fitted on training labels, never future dates.
    f['demand_scale']=f.lag7_mean.fillna(train.peak_upper.mean()).clip(lower=1)+2
    train=f[(f.split=='train')&f.usable]
    naive=XGBRegressor(objective='reg:quantileerror',quantile_alpha=.99,n_estimators=320,max_depth=6,learning_rate=.06,subsample=.85,colsample_bytree=.9,min_child_weight=20,tree_method='hist',enable_categorical=True,n_jobs=4,random_state=20261014)
    naive.fit(train[columns],train.peak_upper)
    f['pred_naive']=naive.predict(f[columns]).clip(min=0)
    naive.save_model(model_dir/'xgb_unscaled_q99.json')
    # Hyperparameters fixed before final test; no tuning against calibration/test.
    for q in [.5,.95,.99]:
        model=XGBRegressor(objective='reg:quantileerror',quantile_alpha=q,n_estimators=320,max_depth=6,learning_rate=.06,subsample=.85,colsample_bytree=.9,min_child_weight=20,tree_method='hist',enable_categorical=True,n_jobs=4,random_state=20261014)
        model.fit(train[columns],train.peak_upper/train.demand_scale)
        f[f'pred_q{int(q*100)}']=model.predict(f[columns]).clip(min=0)*f.demand_scale
        model.save_model(model_dir/f'xgb_q{int(q*100)}.json')
        print('fitted quantile',q,flush=True)
        if q==.99:
            csv(pd.DataFrame({'feature':columns,'gain_importance':model.feature_importances_}).sort_values('gain_importance',ascending=False),OUT/'feature_importance.csv')
    # Quantile crossing is removed with a cumulative maximum, before calibration.
    pred_cols=['pred_q50','pred_q95','pred_q99']
    f[pred_cols]=np.maximum.accumulate(f[pred_cols].to_numpy(),axis=1)
    cal=f[(f.split=='calibration')&f.usable]
    offset=calibration_offset((cal.peak_upper-cal.pred_q99)/cal.demand_scale)
    f['pred_calibrated']=f.pred_q99+offset*f.demand_scale
    # Independent empirical baseline fitted ONLY on training dates.
    baseline=train.groupby(['ROUTE_ID','weekday','hour'],observed=True).peak_upper.quantile(.99).rename('empirical_q99').reset_index()
    fallback=train.groupby(['ROUTE_ID','hour'],observed=True).peak_upper.quantile(.99).rename('fallback').reset_index()
    f=f.merge(baseline,on=['ROUTE_ID','weekday','hour'],how='left').merge(fallback,on=['ROUTE_ID','hour'],how='left')
    f['empirical_q99']=f.empirical_q99.fillna(f.fallback).fillna(train.peak_upper.quantile(.99))
    test=f[(f.split=='test')&f.usable]
    records=[metrics(test,'Historical Q99',test.empirical_q99),metrics(test,'XGBoost Q99',test.pred_q99),metrics(test,'Calibrated XGBoost Q99',test.pred_calibrated)]
    # Ablate lagged demand, retaining the same training and test split.
    short=[c for c in columns if not c.startswith('lag7_')]
    ablation=XGBRegressor(objective='reg:quantileerror',quantile_alpha=.99,n_estimators=320,max_depth=6,learning_rate=.06,subsample=.85,colsample_bytree=.9,min_child_weight=20,tree_method='hist',enable_categorical=True,n_jobs=4,random_state=20261014)
    ablation.fit(train[short],train.peak_upper/train.demand_scale)
    records.append(metrics(test,'XGBoost without lagged tree features',ablation.predict(test[short]).clip(min=0)*test.demand_scale))
    records.append(metrics(test,'Unscaled XGBoost Q99',test.pred_naive))
    q99=XGBRegressor();q99.load_model(model_dir/'xgb_q99.json')
    explained=test.sample(n=min(1500,len(test)),random_state=20261014)
    contributions=q99.get_booster().predict(DMatrix(explained[columns],enable_categorical=True),pred_contribs=True)
    csv(pd.DataFrame({'feature':columns,'mean_abs_tree_shap':np.abs(contributions[:,:-1]).mean(axis=0)}).sort_values('mean_abs_tree_shap',ascending=False),OUT/'shap_summary.csv')
    csv(pd.DataFrame(records),OUT/'model_metrics.csv')
    # Store no vehicle ID in report-facing predictions. Internal IDs stay ignored.
    f.to_parquet(OUT/'predictions.parquet',index=False,compression='zstd')
    write_json(OUT/'model_summary.json',{'features':columns,'target_scaling':'peak_upper / (strictly prior 7-day mean peak + 2); missing history uses training-set global mean','train_dates':sorted(train.service_date.unique().tolist()),'calibration_dates':sorted(cal.service_date.unique().tolist()),'development_dates':sorted(f.loc[f.split.eq('development'),'service_date'].unique().tolist()),'test_dates':sorted(test.service_date.unique().tolist()),'train_n':len(train),'calibration_n':len(cal),'test_n':len(test),'residual_offset_normalized':offset,'capacity':16,'metrics':records,'temporal_caveat':'Calibration is an empirical upper-tail adjustment; serial dependence precludes claiming distribution-free conditional coverage. Lagged labels use strictly earlier dates; test-day outcomes update subsequent days, never same-day forecasts. July27-Aug2 was inspected during development; Aug3-Aug9 reserved as final confirmation.'})

if __name__=='__main__': main()
