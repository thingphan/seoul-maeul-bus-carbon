"""Generate a five-page Korean research report and editable, auditable text."""
from __future__ import annotations
import html
import json
import re
import shutil
from pathlib import Path
import pandas as pd
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,PageBreak
from .common import ROOT,OUT,REPORT,write_json,sha256

REPO='https://github.com/ejjunm/seoul-maeul-bus-carbon'
REFERENCES=[
 ('서울시 (2025.12.21). 마을버스 서비스 개선 합의.', 'https://www.seoul.go.kr/news/news_report.do?nttNo=448993'),
 ('김성아 외 (2023). 서울시 고령자의 대중교통 이용에 영향을 미치는 도시특성 분석. 국토지리학회지 57(4), 377-390.', 'https://doi.org/10.22905/kaopqj.2023.57.4.2'),
 ('김명환·이재현 (2026). 교통약자 수요 기반 저상버스 배치 전략을 통한 이용 형평성 개선효과 분석. 대한지리학회지 61(3), 351-370.', 'https://doi.org/10.22776/kgs.2026.61.3.351'),
 ('Romano et al. (2019). Conformalized Quantile Regression.', 'https://arxiv.org/abs/1905.03222'),
 ('Nakao et al. (2026). Joint optimization of charging infrastructure and fleet mix for CO₂-constrained feeder services. Applied Energy 405, 127216.', 'https://doi.org/10.1016/j.apenergy.2025.127216'),
 ('현대자동차. 카운티 일렉트릭 제조사 제원·정속 주행거리.', 'https://www.hyundai.com/kr/ko/c/products/bus/county-electric'),
 ('Karsan. e-JEST 제원·휠체어 배치별 정원.', 'https://www.karsan.com/en/electric-bus/e-jest/e-jest-specs'),
 ('기후에너지환경부 (2025.12.18). 2023년 전력배출계수 0.4173tCO₂eq/MWh 공표.', 'https://niwdc.mcee.go.kr/home/web/board/read.do?boardId=1829260&boardMasterId=939&menuId=10598'),
 ('XGBoost. Quantile regression 공식 예제.', 'https://xgboost.readthedocs.io/en/stable/python/examples/quantile_regression.html'),
 ('SciPy. milp 공식 문서.', 'https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.milp.html'),
 ('인사혁신처 (2026.4.29). 제헌절 공휴일 재지정.', 'https://www.mpm.go.kr/mpm/comm/newsPress/newsPressRelease/?boardId=bbs_0000000000000029&cntId=4250&mode=view'),
]

def main():
    def load(name):return json.loads((OUT/name).read_text(encoding='utf-8'))
    demand=load('demand_summary.json');model=load('model_summary.json');policy=load('policy_summary.json');finding=load('report_findings.json');validation=load('source_validation_summary.json');distance=load('distance_summary.json')
    metrics=pd.read_csv(OUT/'model_metrics.csv');pmetrics=pd.read_csv(OUT/'policy_metrics.csv')
    chosen=pmetrics[pmetrics.fleet_cap.eq(20)].iloc[0]
    calibrated=metrics[metrics.model.eq('Calibrated XGBoost Q99')].iloc[0]
    empirical=metrics[metrics.model.eq('Historical Q99')].iloc[0]
    allocations=pd.read_csv(OUT/'fleet_allocation.csv',dtype={'ROUTE_ID':str})
    trips=pd.read_parquet(OUT/'trips.parquet');test=trips[trips.split.eq('test')]
    route_names=trips[['ROUTE_ID','노선명']].drop_duplicates('ROUTE_ID').set_index('ROUTE_ID')['노선명']
    candidate=allocations[allocations.fleet_cap.eq(20)].copy();candidate['노선명']=candidate.ROUTE_ID.map(route_names)
    rows=[]
    for name,reason in [('서대문01','평균 기반 소형화 후보를 재검토'),('관악04','피크 수요에 맞는 수송 용량 유지')]:
        item=finding.get(name,{})
        rows.append([name,f"{item.get('q95_upper',0):.1f}명",reason])
    for row in candidate.nlargest(3,'vehicles').itertuples(index=False):rows.append([getattr(row,'노선명'),f'{row.vehicles}대 배분','하루 고정 소형차 시범 조건 확인'])
    regular='C:/Windows/Fonts/malgun.ttf';bold='C:/Windows/Fonts/malgunbd.ttf'
    pdfmetrics.registerFont(TTFont('KR',regular));pdfmetrics.registerFont(TTFont('KR-Bold',bold))
    pdfmetrics.registerFontFamily('KR',normal='KR',bold='KR-Bold',italic='KR',boldItalic='KR-Bold')
    body=ParagraphStyle('body',fontName='KR',fontSize=9.7,leading=14.5,spaceAfter=6,wordWrap='CJK',textColor=colors.HexColor('#202830'))
    head=ParagraphStyle('head',parent=body,fontName='KR-Bold',fontSize=12.3,leading=18,spaceBefore=7,spaceAfter=8)
    sub=ParagraphStyle('sub',parent=body,fontName='KR-Bold',fontSize=10.3,leading=15,spaceBefore=6,spaceAfter=5)
    small=ParagraphStyle('small',parent=body,fontSize=8.0,leading=11.4,spaceAfter=5)
    caption=ParagraphStyle('caption',parent=small,fontSize=8.3,leading=11.6,spaceAfter=8,textColor=colors.HexColor('#465360'))
    cell=ParagraphStyle('cell',parent=small,fontSize=8.0,leading=11.2,spaceAfter=0)
    width=A4[0]-36*mm;story=[];markdown=[]
    def paragraph(text,style=body):story.append(Paragraph(text,style));markdown.append(text.replace('<b>','**').replace('</b>','**').replace('<br/>','\n'))
    def heading(text):paragraph(text,head);markdown[-1]='## '+markdown[-1]
    def subtitle(text):paragraph(text,sub);markdown[-1]='### '+markdown[-1]
    def table(values,widths):
        wrapped=[[Paragraph(str(value),cell) for value in row] for row in values]
        t=Table(wrapped,colWidths=widths,repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#EDF1F4')),('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor('#687988')),('LINEBELOW',(0,1),(-1,-1),.3,colors.HexColor('#CCD3D9')),('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
        story.append(t);story.append(Spacer(1,7))
        markdown.append('\n'.join(['| '+' | '.join(map(str,row))+' |' for row in values[:1]]+['| '+' | '.join(['---']*len(values[0]))+' |']+['| '+' | '.join(map(str,row))+' |' for row in values[1:]]))
    def figure(name,text,image_width=width):
        path=REPORT/'figures'/f'{name}.png'
        public=ROOT/'docs/report/images/upgrade';public.mkdir(parents=True,exist_ok=True)
        for suffix in ['.png','.svg']:shutil.copy2(path.with_suffix(suffix),public/(name+suffix))
        from PIL import Image as PILImage
        with PILImage.open(path) as pic:w,h=pic.size
        story.append(Image(str(path),width=image_width,height=image_width*h/w,hAlign='CENTER'))
        story.append(Spacer(1,4));story.append(Paragraph(text,caption))
        markdown.append(f'![{text}](images/upgrade/{name}.png)\n\n{text}')
    def newpage():story.append(PageBreak());markdown.append('\n<!-- page break -->\n')
    def link(url,label):return f'<link href="{html.escape(url,quote=True)}" color="#234E70">{label}</link>'

    # 1: problem, policy context, objectives, provenance.
    paragraph('서울 마을버스의 서비스 유지형 탈탄소화',ParagraphStyle('title',parent=head,fontSize=19,leading=27,spaceAfter=5))
    paragraph('AI 피크 수요 예측과 제약 기반 차량 배치',ParagraphStyle('title2',parent=head,fontSize=14,leading=21,spaceAfter=9))
    paragraph('전예준·김명관 | 덜컹덜컹달려간다마을버스야 | 분석·문헌 확인: 2026.10.14',small)
    paragraph(f"<b>평균 수요에 따른 소형화 후보를 운행별 수요와 운영 제약으로 검증하였다.</b> 기존 평균 3명 미만 시간대의 검증 가능 운행 중 {finding['legacy_low_average_lower_unsafe_rate']:.1%}는 태그 재구성 하한만으로도 16명을 초과했다. AI 상한 예측과 차량 배치를 연결해, 서비스를 유지하면서 전환할 조건과 제외할 이유를 제시한다.")
    heading('1. 주제 및 2. 분석 배경·목적')
    paragraph('마을버스의 짧은 이동은 언덕·골목의 주거지를 지하철과 생활시설에 연결한다. 이용량과 운송수입이 적더라도, 보행이 어렵거나 대체수단이 부족한 시민에게는 전체 이동을 성립시키는 구간일 수 있다. 따라서 탄소 감축은 이러한 연결을 유지하는 운행·차량 조건과 함께 평가해야 한다.')
    paragraph('서울시는 2025년 12월 마을버스 서비스 개선 합의에서 2026년 재정지원 412억→500억원, 전체 운행 약 5% 확대를 발표했다[1]. 이는 서비스 확충 계획이며 확인된 실현 성과가 아니다. 필요한 운행을 유지·확대할 때 어느 크기의 친환경 차량을 배치해야 하는지에 대한 판단이 필요하다. 500억원을 본 분석의 차량 구매 예산으로 사용하거나 증편률을 7월 기록에 다시 더하지 않았다.')
    paragraph('고령자의 교통카드 이용과 도시특성 연구[2], 교통약자 수요 기반 저상버스 배치 연구[3]는 이용과 접근 조건의 중요성을 보여준다. 본 연구는 이 문제를 <b>평균 기반 후보의 피크 위험 검증→AI 상한 예측→차량·충전 제약에 따른 선별</b>로 연결한다. 참여 목적은 더 큰 감축치를 제시하는 데 그치지 않고, 지원·차량 구매·시범운행의 판단 근거를 구체화하는 것이다.')
    heading('3. 활용 데이터')
    data=[['활용 데이터·선정 이유','제공기관 / 플랫폼','출처 URL'],
     ['교통카드 이용·노선·정류장: 운행별 피크, 경로코드 승차','서울시 / T-DATA',link('https://t-data.seoul.go.kr/dataprovide/trafficdataviewfile.do?data_id=20006','t-data.seoul.go.kr / 20006')],
     ['정류장별 운행횟수: 카드 운행 키의 독립 대조','서울시·TOPIS / 열린데이터광장',link('https://data.seoul.go.kr/dataList/OA-21220/S/1/datasetView.do','data.seoul.go.kr / OA-21220')],
     ['도로망·5분 속도: 주행 저항과 운행 환경','국토교통부 / ITS',link('https://www.its.go.kr/nodelink/nodelinkStatus?service=inquiryNodelink','its.go.kr / nodelink')+'; '+link('https://www.its.go.kr/opendata/','its.go.kr / opendata')],
     ['노선 경로·정류장: 도로 매칭·노선 거리','서울시 / 버스 API·열린데이터광장',link('http://ws.bus.go.kr/api/rest/busRouteInfo/getRoutePath','ws.bus.go.kr / getRoutePath')+'; '+link('https://data.seoul.go.kr/dataList/OA-1095/L/1/datasetView.do','OA-1095')],
     ['시간대 승하차: 기존 평균 기준의 후보 재평가','서울시 / 열린데이터광장',link('https://data.seoul.go.kr/dataList/OA-12913/F/1/datasetView.do','data.seoul.go.kr / OA-12913')],
     ['DEM·생활인구: 경사 및 노선 주변 인구 속성','국토지리정보원·서울시 / 국토정보·열린데이터',link('https://map.ngii.go.kr/','map.ngii.go.kr')+'; '+link('https://data.seoul.go.kr/dataVisual/seoul/seoulLivingPopulation.do','서울 생활인구')],
     ['전기차 제원·배출계수: 정원·배터리·탄소 환산','현대·Karsan·기후부 / 공식 홈페이지','제조사 제원 [6,7], 전력계수 [8]']]
    table(data,[width*.40,width*.27,width*.33])
    paragraph(f"교통카드: 7/6~8/9, 35일·247개 노선·{demand['trips']:,}개 운행 키. 기존 도로 속도·인구는 7월, API 경로·정류장 일부는 9월 자료이다. 시점·거리 차이를 진단해 정책 후보를 제한했다. 속도 AI의 추가 입력 상세 출처는 코드 문서에 제공한다.",small)

    # 2: reproducible methods and AI, concise without hiding definitions.
    newpage();heading('4. 분석 과정 및 방법')
    subtitle('4.1 운행별 수요 재구성과 자료 대조')
    paragraph('차량·노선·출발시각을 운행 키로 묶고, 인접 자료일의 동일 키를 합쳤다. 승차·하차 사건을 시각순으로 누적해 운행별 최대 재차를 재구성했다. 같은 분에 발생한 승하차는 순변화 기반 하한과 승차 우선 상한을 계산하고, 하차 미완결 승객을 상한에 더했다. 이는 태그 기반 구간이며 미태그 승객까지 포함한 실측 정원은 아니다.')
    paragraph(f"출발 이전 승차, 미완결 5% 초과, 비정상 관측시간·자정 이후 미종료 운행을 모델 검증에서 제외했다. 전체 키 중 {demand['usable_trips']/demand['trips']:.1%}를 검증 가능 표본으로 사용했다. 이전 날짜의 결과는 다음 날짜 시작 전에 확보된 값만 쓴다. TOPIS 정류장 통과횟수 대비 카드 운행 키의 중앙 비율은 {validation['ratio_median']:.3f}였다. 재방문·순환 구간의 거리 불확실성은 별도 관리했다.")
    subtitle('4.2 차종·에너지·정수계획')
    paragraph('소형차는 e-JEST M2 휠체어 배치의 운전자 제외 16명·105kWh를 참조한다[7]. 국내 인증과 노선 적합성은 미확인이다. 공통 7t 전기차와 5t 소형차의 동일 노선 주행 저항·냉방·정지출발·체류·충전손실을 계산해 차량 크기 효과만 비교했다. 카운티의 303/338km를 서로 다른 시험주기로 오해한 보정을 수정하고, 정속 자료로 Cd만 식별했다[6].')
    paragraph('MILP는 노선별 소형차 수(정수)와 차량·날짜 운행 묶음의 전환(이진)을 선택해 모형 탄소 감축을 최대화한다. 같은 날 한 차량의 전체 운행을 교체하며 기록된 출발 패턴과 경로를 유지한다. 예측 상한≤16명, 배터리·예비량, 차량 수, 과거 운행횟수 조건을 적용한다. 보정 7일의 정류장 순서가 일치할 때 경로를 재사용하고 에너지 거리는 두 출처 중 큰 값을 쓴다. 보정 주간으로 차량을 사전 배분한 뒤 검증 주간에 순차 재생하였다.')
    heading('5. AI 기술 활용')
    paragraph('기존 XGBoost 도로 속도 산출물을 에너지 입력으로 재사용하고, 새로 XGBoost 분위수 회귀 Q50·Q95·Q99를 구현했다[9]. 노선·시간·요일·공휴일·경로 특성과 이전 7일 수요를 입력한다. 노선별 규모 차이는 과거 피크 평균+2명으로 목표를 정규화해 반영하고, 보정 주간의 정규화 잔차 상위 순서통계량으로 Q99를 보완했다[4]. TreeSHAP은 예측의 변수 기여를 설명하며 인과효과로 해석하지 않는다.')
    table([['학습','오차 보정·차량 배분','개발 비교','독립 최종 검증'],['7/6~7/19','7/20~7/26','7/27~8/2','8/3~8/9']],[width/4]*4)
    paragraph('원척도 모델을 개선하는 과정에서 확인한 개발 주간은 독립 검증에서 제외했다. 같은 날·미래의 수요는 피처에 넣지 않는다. 최종 검증은 앞선 날짜의 결과만 다음 날짜에 업데이트하는 방식이며, 시간 의존성 때문에 보정 상한의 보편적 99% 보장은 주장하지 않는다.',small)
    figure('04_model_explanation','그림 1. 독립 검증 표본의 Q99 TreeSHAP. 정규화 목표의 예측 기여이며 최종 인원은 과거 수요 규모를 곱해 얻는다.',image_width=width*.64)
    paragraph('<b>생성형 AI(Codex)</b>는 수집·모델·제약 코드 작성과 단위·누수·해석 검토에 사용했다. 검토 기준을 정리한 재현 프롬프트: ① “이전 날짜만으로 수요 피처를 만들고, 개발·보정·최종 검증의 역할을 분리하라.” ② “차량 정원에서 운전자를 제외하고, 태그 누락·배터리·자료 차이를 반영하되 실측 배출이나 실제 구매비로 단정하지 마라.”',small)

    # 3: empirical discovery and model comparison.
    newpage();heading('6. 분석 내용 및 결과')
    subtitle('6.1 평균 기준의 소형화 후보는 피크 검증이 필요하다')
    paragraph(f"기존 평균 3명 미만 시간대에 해당하는 최종 검증 가능 운행은 {finding['legacy_low_average_selected_trips']:,}개였다. 이 중 16명 초과 비중은 재구성 하한 {finding['legacy_low_average_lower_unsafe_rate']:.1%}, 상한 {finding['legacy_low_average_upper_unsafe_rate']:.1%}였다. 이는 태그·시각 처리에 따른 범위이며 통계적 신뢰구간이 아니다.")
    figure('01_average_vs_trip_peak','그림 2. 기존 평균 기준으로 선별한 시간대에도 피크 수요가 존재한다. 우측 상자는 중앙값·사분위, 수염은 5~95% 범위이며 일평균 구간 최대와 다른 지표이다.')
    paragraph(f"기존 고의존 소형화 후보 {finding['legacy_high_dependence_candidate_count']}개 중 {finding['legacy_high_dependence_candidates_q99_over16']}개는 검증주 피크 상한 Q99가 16명을 넘었다. 전체 평균 최대≤8명 집합도 {finding['legacy_route_candidate_count']}개 중 {finding['legacy_route_candidates_q99_over16']}개가 초과했다. 서대문01·관악04의 Q95는 {finding.get('서대문01',{}).get('q95_upper',0):.1f}명·{finding.get('관악04',{}).get('q95_upper',0):.1f}명으로, 기존 후보를 곧바로 하루 전체 소형차로 바꿀 근거는 부족하다.")
    subtitle('6.2 AI의 가치는 예측 정확도와 전환 위험을 함께 줄이는 데 있다')
    table([['방법','상한 포함률','16명 이하 예측 비중','선택 운행의 상한 초과율']]+[[{'Historical Q99':'과거 경험 Q99','XGBoost Q99':'규모보정 XGB Q99','Calibrated XGBoost Q99':'오차보정 XGB Q99','XGBoost without lagged tree features':'트리 이력피처 제외','Unscaled XGBoost Q99':'원척도 XGB Q99'}.get(r.model,r.model),f'{r.upper_coverage:.2%}',f'{r.convertible_trip_share:.2%}',f'{r.unsafe_selected_rate:.2%}' if pd.notna(r.unsafe_selected_rate) else '선택 없음'] for r in metrics.itertuples(index=False)],[width*.34,width*.18,width*.23,width*.25])
    figure('02_ai_validation',f"그림 3. 최종 검증 {len(test[test.usable]):,}개 운행에서의 방법 비교. ‘상한 초과’는 실제 잠재 피크의 정확한 초과율이 아니라 검증 가능한 태그 상한 기준이다.")
    paragraph(f"보정 모델의 포함률은 {calibrated.upper_coverage:.2%}, 선택 운행 초과율은 {calibrated.unsafe_selected_rate:.2%}였다. 경험 Q99의 {empirical.upper_coverage:.2%}·{empirical.unsafe_selected_rate:.2%}와 비교해 안전 여지와 전환 범위의 교환을 확인했다. 원척도 모델처럼 아무 운행도 선택하지 않는 높은 포함률은 정책 성능이 충분하다는 뜻이 아니다.",small)

    # 4: constrained policy outcomes and concrete cases.
    newpage();subtitle('6.3 수요·충전·자료 조건을 적용하면 전환 범위가 달라진다')
    paragraph('그림 4는 모든 차량을 작은 차로 바꾸는 잠재치에서 용량·배터리·거리·자료 대조 조건을 순서대로 적용한 결과이다. 단계의 감축량은 공통 전기차 기준 크기 효과이며 기존 현재 차량 구성 대비 7.53톤/일과 분모·차종·모형이 다르다. 조건별 감소를 실패로 감추지 않고 필요한 추가 조사와 도입 제약으로 해석한다.')
    figure('03_constraints_and_fleet','그림 4. A: 운행 묶음별 제약 선별. B: 사전 배분 차량의 검증주 재생. 배출은 운행 전력사용의 CO₂eq이며 차량 제조는 제외한다. 차량 수는 실제 예산·보유대수가 아닌 가정이다.')
    table([['확보 상한','배분 차량 / 노선','검증주 모형 감축','전력비 절감(가정)']]+[[f'{r.fleet_cap}대',f'{r.installed_vehicles}대 / {r.routes}개',f'{r.saved_kg_per_day:.1f}kg/일',f'{r.electricity_saving_won_per_day:,.0f}원/일'] for r in pmetrics[pmetrics.fleet_cap.le(5)].itertuples(index=False)],[width*.17,width*.25,width*.27,width*.31])
    paragraph(f"20대 상한에서도 {int(chosen.installed_vehicles)}대·{int(chosen.routes)}개 노선만 배분되어 {chosen.saved_kg_per_day:.1f}kg/일의 크기 효과를 얻었다. 검증주 {int(chosen.selected_trips):,}개 기록 운행을 전환한 시나리오다. 이 조건에서는 추가 구매보다 용량·운영 적합성이 더 큰 제약이었다. 동일 특수 구조의 한계편익 탐욕법은 보정 기간 최적값과 일치해, MILP만의 우월성을 주장하지 않는다.")
    table([['사례 노선','검증 지표','판단의 변화']]+rows,[width*.18,width*.26,width*.56])
    paragraph(f"20대 조건의 전환 운행 중 {chosen.unassessable_selected_share:.1%}는 태그 품질 기준상 정원 검증이 불가능했다. 검증 가능 {int(chosen.assessable_selected_trips):,}개 중 상한 초과는 {int(chosen.unsafe_assessable_trips)}개였다. ‘상한 예측을 적용했다’는 이유만으로 승차 거부가 없다고 단정할 수 없으며, 실제 피크·휠체어 이용과 공차 운행을 확인해야 한다.")
    subtitle('6.4 비용은 확인된 범위와 가격 미확인 영역을 구분한다')
    paragraph(f"전력비 200원/kWh를 가정한 절감은 하루 {chosen.electricity_saving_won_per_day:,.0f}원이다. 365일 단순 환산 시 차량당 연 {chosen.annualized_electricity_only_break_even_premium_per_vehicle_won:,.0f}원의 순추가비용까지 상쇄한다. 구매·보조금·인건비·정비비를 포함한 TCO는 아니므로, 실제 견적으로 추가 구매와 교체 시점 전환을 비교해야 한다.")
    stress=pd.read_csv(OUT/'robustness_metrics.csv')
    same=stress[stress.capacity.eq(16)]
    paragraph(f"같은 16명 정원에서 속도·냉방·회생·배터리·차고지 가정을 바꾼 감축은 {same.saved_kg_per_day.min():.1f}~{same.saved_kg_per_day.max():.1f}kg/일이었다. 이는 조건별 민감도 범위이며 신뢰구간이 아니다. 시간 블록 교대나 낮 충전은 실제 차고지·근무표 확보 후 별도 검증해야 한다.",small)

    # 5: actionable policy, limitations, references.
    newpage();heading('7. 정책 제안 및 기대효과')
    paragraph('서울시·자치구는 운행 정상화와 친환경 차량 지원을 연계하되, 평균 재차만으로 소형차 구매를 결정하지 않아야 한다. 운수사는 AI를 수송 용량의 대체 근거가 아니라 전환 위험을 사전에 식별하는 도구로 사용할 수 있다. 노선·정류장·기록된 출발 패턴을 유지한 조건에서 다음과 같이 적용한다.')
    table([['정책 단계','적용 대상·실행 내용','확인할 성과'],['후보 재검증','평균 기준 후보를 운행별 피크·태그 품질·차량 정원으로 다시 선별','정원 초과·누락 운행·피크 대체차 필요'],['교체 시점의 저상 소형차 시범','배터리·자료 조건을 통과한 후보의 국내 인증·도로 폭·회차·충전·휠체어 탑승 확인','실차 kWh/km·전력비·대기·승차 거부·접근성'],['필요 노선의 용량·지원 유지','서대문01·관악04 등 피크가 큰 사례는 수송 용량을 유지하고 적합한 전기화 검토','운행 이행률·교통약자 탑승·서비스 연속성'],['구매·지원 의사결정','실제 견적·차량 수명·정비·보조금을 받아 총비용과 감축량 재산정','총비용·감축비용·증분 예산·고령자 이용 변화']],[width*.20,width*.50,width*.30])
    paragraph('기대효과는 감축량 자체와 함께 오판을 줄이는 데 있다. 피크 수요 때문에 유지할 용량, 자료 부족으로 보류할 후보, 충전·경제성 조건을 확인할 후보를 구분하면 재정지원과 차량 구매의 근거가 구체화된다. 경로·배차를 유지하는 설계는 주변 고령 생활인구의 접근 기회를 보호하는 방향이지만, 실제 고령 이용자의 편익 증가를 측정한 결과는 아니다.')
    subtitle('한계와 후속 검증')
    paragraph('분석은 여름 5주에 한정되며 검증 가능 태그 운행의 선택 편향이 있다. 경로코드는 모든 고령 이용자를 포괄하지 않고 장애코드는 휠체어 여부와 다르다. 카드 키와 TOPIS 통과횟수의 정의 차이, 미태그·무승객 운행, 7월/9월 공간자료 차이가 남고 평일 속도를 주말·야간에 근사 적용했다. 실제 차종·차고지·충전기·근무표·실차 전력 확보 전에는 배치를 실행 확정안이나 감축 실적으로 사용할 수 없다. 노선 폐지 접근성·목적지 편익은 새 검증에서 재산정하지 않았다.')
    heading('8. 분석 도구 및 참고문헌')
    paragraph('Python 3.13, pandas·NumPy·PyArrow, XGBoost 3.4, SciPy MILP(HiGHS), Matplotlib, ReportLab. 그림 1~4는 실제 산출물로 생성했다(필수 분석결과 이미지 포함). 공개 자료의 SHA-256·내부 날짜·코드·설정 해시, 테스트와 재현 절차를 함께 보존하였다.',small)
    for i,(title,url) in enumerate(REFERENCES,1):paragraph(f'[{i}] {title} '+link(url,'원문'),small)
    paragraph('분석 코드 및 상세 데이터 출처: '+link(REPO,REPO),small)
    paragraph('신규 구현·입력 계약·추가 진단: docs/MAEUL_UPGRADE_IMPLEMENTATION.md 및 docs/report/maeul_bus_upgrade_validation.md. 제출 전 저장소에 신규 모듈과 재현 입력 안내를 반영해야 한다.',small)

    output=ROOT/'output/pdf';output.mkdir(parents=True,exist_ok=True)
    target=output/'서울_마을버스_AI_차량정책_분석보고서_5p.pdf'
    def footer(canvas,doc):
        canvas.saveState();canvas.setFont('KR',8);canvas.setFillColor(colors.HexColor('#64727D'))
        canvas.drawString(18*mm,12*mm,'서울 마을버스 · AI 수요 및 차량 정책 분석')
        canvas.drawRightString(A4[0]-18*mm,12*mm,str(doc.page));canvas.restoreState()
    doc=SimpleDocTemplate(str(target),pagesize=A4,rightMargin=18*mm,leftMargin=18*mm,topMargin=15*mm,bottomMargin=19*mm,title='서울 마을버스의 서비스 유지형 탈탄소화',author='전예준·김명관')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    mdtext='# '+'\n\n'.join(markdown)
    mdtext=re.sub(r'<link href="([^"]*)"[^>]*>(.*?)</link>',lambda m:f'[{html.unescape(m.group(2))}]({html.unescape(m.group(1))})',mdtext)
    md=ROOT/'docs/report/maeul_bus_upgrade_report.md';md.write_text(mdtext,encoding='utf-8')
    detail=ROOT/'docs/report/maeul_bus_upgrade_validation.md'
    def mdtable(frame):
        return '\n'.join(['| '+' | '.join(frame.columns)+' |','| '+' | '.join(['---']*len(frame.columns))+' |']+['| '+' | '.join(str(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None)])
    stress=pd.read_csv(OUT/'robustness_metrics.csv')
    full='\n\n'.join(['# 업그레이드 분석 상세 검증', '## 데이터 품질\n```json\n'+json.dumps({'demand':demand,'source_validation':validation,'distance':distance,'topology':load('topology_summary.json'),'candidate_reassessment':finding},ensure_ascii=False,indent=2)+'\n```', '## AI 검증\n'+mdtable(metrics), '## 차량 배치 검증\n'+mdtable(pmetrics), '## 물리·운영 조건 민감도\n'+mdtable(stress), '## 구현·가정\n'+(ROOT/'docs/MAEUL_UPGRADE_IMPLEMENTATION.md').read_text(encoding='utf-8'), '## 원자료와 기존 속도 산출물의 상세 출처\n'+(ROOT/'docs/report/maeul_bus_report.md').read_text(encoding='utf-8').split('## 3. 활용 데이터')[1].split('## 4.')[0], '## 연구·정책 출처\n'+'\n'.join(f'{i}. [{t}]({u})' for i,(t,u) in enumerate(REFERENCES,1))])
    detail.write_text(full,encoding='utf-8')
    snapshots=ROOT/'docs/report/metrics/upgrade';snapshots.mkdir(parents=True,exist_ok=True)
    for name in ['model_metrics.csv','policy_metrics.csv','constraint_attrition.csv','robustness_metrics.csv','fleet_allocation.csv','shap_summary.csv']:
        shutil.copy2(OUT/name,snapshots/name)
    provenance=[]
    for path in sorted((OUT/'fetch_manifest').glob('*.json')):
        item=json.loads(path.read_text(encoding='utf-8'))
        provenance.append({key:item[key] for key in ['source','file_name','package_day','service_day','sha256','zip_bytes','village_rows']})
    write_json(snapshots/'public_source_manifest.json',provenance)
    import pymupdf
    pdf=pymupdf.open(target);render=ROOT/'tmp/pdfs/maeul_upgrade';render.mkdir(parents=True,exist_ok=True)
    for i,page in enumerate(pdf):page.get_pixmap(matrix=pymupdf.Matrix(1.3,1.3)).save(render/f'page_{i+1}.png')
    write_json(REPORT/'report_validation.json',{'pages':len(pdf),'target_pages':5,'pdf':str(target),'pdf_sha256':sha256(target),'report_source_sha256':sha256(Path(__file__)),'input_result_sha256':{name:sha256(OUT/name) for name in ['model_metrics.csv','policy_metrics.csv','report_findings.json','robustness_metrics.csv']},'page_text_chars':[len(p.get_text()) for p in pdf],'replacement_glyphs':sum(p.get_text().count('\ufffd') for p in pdf)})
    print('report pages',len(pdf),'PDF',target,flush=True)
    if len(pdf)!=5:raise RuntimeError(f'Layout requires revision: {len(pdf)} pages, expected 5')

if __name__=='__main__':main()
