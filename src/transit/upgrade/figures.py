"""Publication figures from saved test results; no generative chart artwork."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from .common import ROOT,BASE,OUT,REPORT,write_json,csv

NAVY='#234E70';TEAL='#15847D';ORANGE='#C56B35';GREY='#83909C'

def setup():
    font=ROOT/'assets/fonts/NotoSansKR-Regular.ttf'
    if not font.exists():font=__import__('pathlib').Path('C:/Windows/Fonts/malgun.ttf')
    if font.exists():font_manager.fontManager.addfont(str(font));plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({'font.size':10,'axes.titlesize':11,'axes.labelsize':10,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#B3BCC4','axes.unicode_minus':False,'figure.dpi':130,'savefig.dpi':250,'svg.fonttype':'none'})

def save(fig,name):
    folder=REPORT/'figures';folder.mkdir(exist_ok=True)
    fig.savefig(folder/f'{name}.png',bbox_inches='tight',facecolor='white')
    fig.savefig(folder/f'{name}.svg',bbox_inches='tight',facecolor='white')
    plt.close(fig)

def main():
    setup()
    f=pd.read_parquet(OUT/'predictions.parquet');test=f[(f.split=='test')&f.usable].copy()
    old=pd.read_csv(BASE/'route_hourly_load.csv',dtype={'ROUTE_ID':str})[['ROUTE_ID','hour','avg_load']]
    test=test.merge(old.rename(columns={'avg_load':'legacy_avg_load'}),on=['ROUTE_ID','hour'],how='left',validate='many_to_one')
    low=test[test.legacy_avg_load.lt(3)]
    old_peak=pd.read_csv(BASE/'route_priority.csv',dtype={'ROUTE_ID':str})[['ROUTE_ID','avg_peak_load_per_trip','class_id']]
    route_new=test.groupby('ROUTE_ID').peak_upper.quantile(.99).rename('test_peak_q99').reset_index().merge(old_peak,on='ROUTE_ID')
    old_candidates=route_new[route_new.avg_peak_load_per_trip.le(8)]
    findings={'test_usable_trips':len(test),'legacy_low_average_selected_trips':len(low),'legacy_low_average_upper_exceed16':int(low.peak_upper.gt(16).sum()),'legacy_low_average_lower_exceed16':int(low.peak_lower.gt(16).sum()),'legacy_low_average_upper_unsafe_rate':float(low.peak_upper.gt(16).mean()),'legacy_low_average_lower_unsafe_rate':float(low.peak_lower.gt(16).mean()),'legacy_route_candidate_count':len(old_candidates),'legacy_route_candidates_q99_over16':int(old_candidates.test_peak_q99.gt(16).sum())}
    priority_candidates=old_candidates[old_candidates.class_id.eq(2)]
    findings['legacy_high_dependence_candidate_count']=len(priority_candidates)
    findings['legacy_high_dependence_candidates_q99_over16']=int(priority_candidates.test_peak_q99.gt(16).sum())
    fig,axes=plt.subplots(1,2,figsize=(10.1,3.35),gridspec_kw={'width_ratios':[1.15,1]})
    bins=np.arange(0,52,2)
    axes[0].hist(low.peak_upper.clip(upper=50),bins=bins,color=NAVY,edgecolor='white',linewidth=.5)
    axes[0].axvline(16,color=ORANGE,linestyle='--',linewidth=1.5)
    axes[0].set(title='A  기존 평균 3명 미만 시간대의 운행별 피크',xlabel='태그 재구성 피크 상한 (명)',ylabel='검증 가능한 운행 수')
    axes[0].text(.97,.94,f"정원 16명 초과 상한: {findings['legacy_low_average_upper_unsafe_rate']:.1%}\n하한으로도 초과: {findings['legacy_low_average_lower_unsafe_rate']:.1%}",transform=axes[0].transAxes,ha='right',va='top',fontsize=9)
    names=['서대문01','관악04']
    data=[test[test['노선명'].eq(n)].peak_upper.to_numpy() for n in names]
    boxes=axes[1].boxplot(data,positions=[1,2],widths=.5,patch_artist=True,showfliers=False,whis=(5,95))
    for box,col in zip(boxes['boxes'],[TEAL,NAVY]):box.set_facecolor(col);box.set_alpha(.75)
    axes[1].axhline(16,color=ORANGE,linestyle='--');axes[1].set_xticks([1,2],names)
    axes[1].set(title='B  과거 사례 노선의 실제 운행 변동',ylabel='태그 재구성 피크 상한 (명)')
    for i,(name,values) in enumerate(zip(names,data)):
        if len(values):
            axes[1].text(i+1,float(np.quantile(values,.95))+2,f'n={len(values):,}',ha='center',fontsize=9)
            findings[name]={'n':len(values),'median_upper':float(np.median(values)),'q95_upper':float(np.quantile(values,.95)),'q99_upper':float(np.quantile(values,.99)),'lower_exceed16_share':float(test[test['노선명'].eq(name)].peak_lower.gt(16).mean())}
    fig.tight_layout();save(fig,'01_average_vs_trip_peak')
    csv(route_new,OUT/'legacy_candidate_reassessment.csv')

    metrics=pd.read_csv(OUT/'model_metrics.csv')
    labels=['경험\nQ99','규모보정\nXGB','오차보정\nXGB','이력피처\n제외','원척도\nXGB']
    x=np.arange(len(metrics));fig,axes=plt.subplots(1,2,figsize=(10.1,3.25))
    axes[0].bar(x,metrics.upper_coverage*100,color=[GREY,NAVY,TEAL,ORANGE,GREY],width=.6)
    axes[0].axhline(99,color=ORANGE,linestyle='--',linewidth=1)
    axes[0].set_ylim(max(80,float(metrics.upper_coverage.min()*100)-2),100)
    axes[0].set_xticks(x,labels);axes[0].set(title='A  미래 운행에서 피크 상한을 덮는 비율',ylabel='피크 상한 포함률 (%)')
    for i,row in metrics.iterrows():axes[0].text(i,row.upper_coverage*100+.15,f'{row.upper_coverage:.2%}',ha='center',fontsize=8)
    axes[1].bar(x-.16,metrics.convertible_trip_share*100,width=.3,color=NAVY,label='16명 이하로 예측한 운행 비중')
    axes[1].bar(x+.16,metrics.unsafe_selected_rate*100,width=.3,color=ORANGE,label='선택 운행 중 상한 초과 비율')
    axes[1].set_xticks(x,labels);axes[1].set(title='B  전환 여지와 정원 초과 위험의 교환',ylabel='비율 (%)');axes[1].legend(frameon=False,fontsize=8)
    fig.tight_layout();save(fig,'02_ai_validation')

    attr=pd.read_csv(OUT/'constraint_attrition.csv');policy=pd.read_csv(OUT/'policy_metrics.csv')
    stage_labels=['전체 태그 운행 묶음','AI 용량 기준 통과','배터리·예비량 통과','과거 정류장 순서 일치','운행횟수 대조 통과','하루 고정·단일노선 통과']
    fig,axes=plt.subplots(1,2,figsize=(10.1,3.45),gridspec_kw={'width_ratios':[1.3,1]})
    y=np.arange(len(attr));axes[0].barh(y,attr.potential_saved_kg_per_day/1000,color=[GREY,NAVY,NAVY,NAVY,NAVY,TEAL]);axes[0].invert_yaxis();axes[0].set_yticks(y,stage_labels,fontsize=9)
    axes[0].set(title='A  운영 조건에 따른 전환 범위 축소',xlabel='공통 전기차 기준 잠재 감축 (톤/일)')
    for i,row in attr.iterrows():
        amount=f'{row.potential_saved_kg_per_day/1000:.1f}톤/일' if row.potential_saved_kg_per_day>=1000 else f'{row.potential_saved_kg_per_day:.1f}kg/일'
        axes[0].text(row.potential_saved_kg_per_day/1000+.05,i,f'{row.duty_days:,} 묶음 · {amount}',va='center',fontsize=7.8)
    axes[0].set_xlim(0,max(.1,attr.potential_saved_kg_per_day.max()/1000)*1.35)
    axes[1].plot(policy.fleet_cap,policy.saved_kg_per_day,'o-',color=TEAL,linewidth=2,label='MILP 사전 배치 + 검증주 재생')
    summary=json.loads((OUT/'policy_summary.json').read_text(encoding='utf-8'))
    axes[1].scatter([20],[summary['greedy20_saved_kg_per_day']],marker='x',s=70,color=ORANGE,label='노선 단위 탐욕 비교')
    axes[1].set(title='B  차량 확보 규모별 검증주 감축',xlabel='확보 가능한 소형차 상한 (대, 가정)',ylabel='모형 감축 (kgCO₂eq/일)');axes[1].legend(frameon=False,fontsize=8)
    axes[1].set_ylim(bottom=0);fig.tight_layout();save(fig,'03_constraints_and_fleet')

    shap=pd.read_csv(OUT/'shap_summary.csv').head(7)
    translations={'route_category':'노선','hour':'운행 시간','weekday':'요일','weekend':'주말','public_holiday':'공휴일','hour_sin':'시간 주기(sin)','hour_cos':'시간 주기(cos)','length_km':'경로 길이','n_stops':'정류장 수','lag7_mean':'최근 7일 평균 피크','lag7_q95':'최근 7일 상위 피크','lag7_count':'최근 7일 표본 수','catchment_elderly_share':'주변 고령 비율','no_alt_boarding_share':'대체수단 부족','dependence_score':'교통약자 의존도'}
    fig,ax=plt.subplots(figsize=(5.6,3.2));ax.barh(np.arange(len(shap)),shap.mean_abs_tree_shap,color=NAVY);ax.set_yticks(np.arange(len(shap)),[translations.get(v,v) for v in shap.feature]);ax.invert_yaxis();ax.set(xlabel='규모로 나눈 피크 예측의 평균 절대 TreeSHAP 기여',title='Q99 예측을 설명하는 변수');fig.tight_layout();save(fig,'04_model_explanation')
    write_json(OUT/'report_findings.json',findings)

if __name__=='__main__':main()
