"""Figures for the full study: carbon, accessibility and operation validation."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .figures import setup,NAVY,TEAL,ORANGE,GREY
from .integrated import FULL
from .common import OUT,ROOT,write_json,csv

FIG=ROOT/'output/maeul_integrated/figures'

def save(fig,name):
    FIG.mkdir(parents=True,exist_ok=True)
    for ext in ['png','svg']:fig.savefig(FIG/f'{name}.{ext}',bbox_inches='tight',facecolor='white')
    plt.close(fig)

def main():
    setup();summary=json.loads((FULL/'integrated_summary.json').read_text(encoding='utf-8'))
    p=pd.read_csv(FULL/'route_priority.csv',dtype={'ROUTE_ID':str})
    removal=pd.read_csv(FULL/'route_removal_access.csv',dtype={'ROUTE_ID':str})
    f=pd.read_parquet(OUT/'predictions.parquet');test=f[f.split.eq('test')&f.usable].copy()
    hourly=pd.read_csv(FULL/'route_hourly_load.csv',dtype={'ROUTE_ID':str})
    test=test.merge(hourly[['ROUTE_ID','hour','avg_load']],on=['ROUTE_ID','hour'],how='left',validate='many_to_one')
    low=test[test.avg_load.lt(3)]
    q=test.groupby('ROUTE_ID').peak_upper.quantile(.99)
    screening=p[p.class_id.eq(2)&p.van_feasible]
    matched=screening[screening.ROUTE_ID.isin(q.index)]
    findings={'low_mean_test_trips':len(low),'low_mean_lower_over16':int(low.peak_lower.gt(16).sum()),'low_mean_upper_over16':int(low.peak_upper.gt(16).sum()),'low_mean_lower_over16_share':float(low.peak_lower.gt(16).mean()),'low_mean_upper_over16_share':float(low.peak_upper.gt(16).mean()),'screening_high_dep_candidates':len(screening),'screening_candidates_test_matched':len(matched),'screening_candidates_q99_over16':int(matched.ROUTE_ID.map(q).gt(16).sum())}
    case=[]
    for name in ['서대문01','관악04']:
        a=removal[removal['노선명'].eq(name)].iloc[0];t=test[test['노선명'].eq(name)]
        case.append({'route':name,'elderly_new_over10_net':float(a.net_elderly_over_10min_new),'elderly_new_over10_simple':float(a.simple_elderly_over_10min_new),'peak_q95_upper':float(t.peak_upper.quantile(.95)),'class_id':int(a.class_id)})
    findings['case_routes']=case;write_json(FULL/'integrated_findings.json',findings)

    fig,ax=plt.subplots(figsize=(10.4,1.25));ax.set_xlim(0,10);ax.set_ylim(0,1);ax.axis('off')
    labels=['경로·승하차·운행\n생활인구·대체수단','도로 속도 AI\n+ 차량 물리 모형','탄소 효율 비교\n+ 고령자 접근성','전기화·소형화\n정책 후보 도출','운행별 피크 AI\n+ 차량 배치 검증']
    for i,label in enumerate(labels):
        x=i*2+.1
        ax.add_patch(plt.Rectangle((x,.25),1.7,.55,fill=False,edgecolor=NAVY,linewidth=1))
        ax.text(x+.85,.525,label,ha='center',va='center',fontsize=10)
        if i<4:ax.annotate('',xy=(x+2,.525),xytext=(x+1.7,.525),arrowprops={'arrowstyle':'->','color':GREY})
    ax.text(5,.02,'서비스 유지와 탄소 감축을 함께 판단하는 노선·차량 정책',ha='center',fontsize=10,color=NAVY)
    fig.tight_layout();save(fig,'00_full_flow')

    fig,axes=plt.subplots(1,2,figsize=(10.3,3.2),gridspec_kw={'width_ratios':[1.25,1]})
    colors=[TEAL,ORANGE,NAVY,GREY]
    for k,color in enumerate(colors,1):
        subset=p[p.class_id.eq(k)];axes[0].scatter(subset.avg_load,subset.bus_g_per_pkm,s=20,alpha=.75,color=color)
    axes[0].axhline(summary['car_g_per_pkm_median'],color=ORANGE,linestyle='--',linewidth=1)
    axes[0].set_yscale('log');axes[0].set(xlabel='노선 평균 재차인원 (명)',ylabel='버스 승객·km당 모형 배출 (gCO₂eq)',title='A  재차인원과 탄소 효율')
    axes[0].text(.97,.95,f"승용차보다 낮음 {summary['bus_better']}/{summary['routes_analyzed']}개\n점선: 승용차 노선 중앙값",transform=axes[0].transAxes,ha='right',va='top',fontsize=9)
    vals=[summary['electrification_cut_t_per_day'],summary['hour_small_current_cut_t_per_day'],summary['hour_small_size_cut_t_per_day']]
    labs=['전면 전기화','평균 저재차\n시간대 소형화','소형화 중\n크기 효과']
    axes[1].bar(np.arange(3),vals,color=[NAVY,TEAL,ORANGE],width=.55)
    axes[1].set_xticks(np.arange(3),labs,fontsize=9);axes[1].set(ylabel='모형 감축 잠재량 (tCO₂eq/일)',title='B  정책별 감축 잠재량')
    axes[1].set_ylim(0,max(vals)*1.2)
    for i,value in enumerate(vals):axes[1].text(i,value+.2,f'{value:.2f}',ha='center',fontsize=9)
    fig.tight_layout();save(fig,'01_carbon_and_potential')

    fig,axes=plt.subplots(1,2,figsize=(10.3,3.2),gridspec_kw={'width_ratios':[1.2,1]})
    median_dep=p.dependence_score.median();median_carbon=p.bus_g_per_pkm.median()
    for k,color in enumerate(colors,1):
        subset=p[p.class_id.eq(k)];axes[0].scatter(subset.dependence_score,subset.bus_g_per_pkm,s=22,alpha=.75,color=color)
    axes[0].axvline(median_dep,color=GREY,linestyle='--');axes[0].axhline(median_carbon,color=GREY,linestyle='--')
    axes[0].set_yscale('log');axes[0].set(xlabel='교통약자 의존도 점수',ylabel='버스 승객·km당 모형 배출 (gCO₂eq)',title='A  탄소 효율과 서비스 필요의 결합')
    axes[0].text(.97,.96,'② 규모 조정 검토',transform=axes[0].transAxes,ha='right',fontsize=9,color=ORANGE)
    axes[0].text(.97,.05,'① 지원·조건부 증편',transform=axes[0].transAxes,ha='right',fontsize=9,color=TEAL)
    axes[0].text(.03,.96,'④ 노선 조정 검토',transform=axes[0].transAxes,fontsize=9,color=GREY)
    axes[0].text(.03,.05,'③ 현행 유지',transform=axes[0].transAxes,fontsize=9,color=NAVY)
    high=removal[removal.class_id.isin([1,2])];lowdep=removal[removal.class_id.isin([3,4])]
    x=np.arange(2);net=[lowdep.net_elderly_affected.median(),high.net_elderly_affected.median()];simple=[lowdep.simple_elderly_affected.median(),high.simple_elderly_affected.median()]
    axes[1].bar(x-.16,simple,width=.3,color=GREY,label='직선×1.3');axes[1].bar(x+.16,net,width=.3,color=NAVY,label='도로망')
    axes[1].set_xticks(x,['의존도 낮음','의존도 높음']);axes[1].set(title='B  개별 노선 제거 시 접근성 손실',ylabel='접근 1분 이상 증가 고령 생활인구\n노선 중앙값 (명)');axes[1].legend(frameon=False,fontsize=8)
    for i,(a,b) in enumerate(zip(simple,net)):
        axes[1].text(i-.16,a+30,f'{a:,.0f}',ha='center',fontsize=8);axes[1].text(i+.16,b+30,f'{b:,.0f}',ha='center',fontsize=8)
    axes[1].set_ylim(0,max(net+simple)*1.2);fig.tight_layout();save(fig,'02_equity_and_access')

    metric=pd.read_csv(OUT/'model_metrics.csv').iloc[:3]
    fig,axes=plt.subplots(1,2,figsize=(10.3,2.7))
    axes[0].hist(low.peak_upper.clip(upper=50),bins=np.arange(0,52,2),color=NAVY,edgecolor='white')
    axes[0].axvline(16,color=ORANGE,linestyle='--');axes[0].set(title='A  평균 3명 미만 시간대도 피크가 발생',xlabel='운행별 태그 피크 상한 (명)',ylabel='검증 가능 운행 수')
    axes[0].text(.97,.93,f"16명 초과: 하한 {findings['low_mean_lower_over16_share']:.1%}\n상한 {findings['low_mean_upper_over16_share']:.1%}",transform=axes[0].transAxes,ha='right',va='top',fontsize=9)
    x=np.arange(3);axes[1].bar(x-.16,metric.convertible_trip_share*100,width=.3,color=NAVY,label='16명 이하로 예측한 운행 비중');axes[1].bar(x+.16,metric.unsafe_selected_rate*100,width=.3,color=ORANGE,label='선택 운행 중 태그 상한 초과')
    axes[1].set_xticks(x,['과거 경험 Q99','규모보정 XGB','오차보정 XGB'],fontsize=9);axes[1].set(title='B  전환 여지와 정원 초과 위험',ylabel='비율 (%)');axes[1].legend(frameon=False,fontsize=8)
    fig.tight_layout();save(fig,'03_peak_and_ai')

if __name__=='__main__':main()
