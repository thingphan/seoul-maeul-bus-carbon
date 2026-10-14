"""보고서(docs/report/maeul_bus_report.md)의 핵심 수치를 분석 산출물에서 다시 계산해 출력한다(읽기만 함).

보고서 순서대로 출력한다. 감축량은 모두 '현재 차량 구성 대비' 하루 값이며, 연 값은 하루 × 365이다.
AI 속도 시험·가정 민감도·군집 점검·B/C 등급별 성능은 같은 폴더의 다른 스크립트가 계산한다.

입력: data/processed/transit/의 01~10 산출물, data/processed/seoul_link_dimension.csv
사용: python src/transit/checks/report_numbers.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
TRANSIT = PROJECT_ROOT / "data" / "processed" / "transit"
DIMENSION_CSV = PROJECT_ROOT / "data" / "processed" / "seoul_link_dimension.csv"
CASES = ("서대문01", "관악04")


def read(name: str) -> pd.DataFrame:
    return pd.read_csv(TRANSIT / name, dtype={"ROUTE_ID": str, "LINK_ID": str, "NODE_ID": str})


def section(title: str) -> None:
    print(f"\n== {title}")


def main() -> None:
    carbon_all = read("route_carbon_breakeven.csv")
    c = carbon_all[~carbon_all["load_qa_flag"]]
    p = read("route_priority.csv")
    o = read("route_operations.csv")
    h = read("route_hourly_load.csv")
    rm = read("route_removal_access.csv")
    feas = read("route_feasibility.csv")
    demand = read("route_demand.csv")
    stops = read("bus_stops.csv")

    section("배경")
    print(f"마을버스 노선 {len(carbon_all)}개, 분석 {len(c)}개(제외 {int(carbon_all['load_qa_flag'].sum())}개), "
          f"하루 승차 {carbon_all['boardings_per_day'].sum():,.0f}건, 평균 승차거리 중앙 {c['avg_trip_km'].median():.2f}km")
    village = stops["served_kinds"].str.contains("마을버스")
    no_alt = (stops["dist_to_station_m"] > 500) & ~stops["trunk_stop_within_200m"].astype(bool)
    print(f"대체수단 없는 마을버스 정류장(지하철 500m 밖·일반버스 200m 안 없음) {no_alt[village].mean():.1%}")
    print("운행비 미충당 노선(승차당 실수입 600·900·1,200원):",
          {fare: int(o[f"deficit_at_{fare}"].sum()) for fare in (600, 900, 1200)})
    print(f"차량·km당 CO2 중앙: 마을버스 {c['bus_g_per_km'].median():.0f}g, 승용차 {c['car_g_per_km'].median():.0f}g")

    section("AI 속도 보완(마을버스 노선)")
    links = read("route_links.csv")
    links = links[links["kind"] == "마을버스"]
    tier = pd.read_parquet(TRANSIT / "link_speed_weekday_profile.parquet", columns=["LINK_ID", "tier"]).drop_duplicates("LINK_ID")
    lanes = pd.read_csv(DIMENSION_CSV, usecols=["LINK_ID", "LANES"], dtype={"LINK_ID": str})
    m = links.merge(tier, on="LINK_ID").merge(lanes, on="LINK_ID", how="left")
    share = m.groupby("tier")["LENGTH"].sum() / m["LENGTH"].sum()
    print("노선 길이 기준 등급 비중:", share.round(3).to_dict())
    m["one_lane"] = m["LANES"] <= 1
    m["ai"] = m["tier"] != "A_lookup"
    one_lane = (m["LENGTH"] * m["one_lane"]).groupby(m["ai"]).sum() / m.groupby("ai")["LENGTH"].sum()
    print(f"1차로 비중: AI 추정 구간 {one_lane[True]:.0%}, 관측 구간 {one_lane[False]:.0%}")
    ai_by_class = p[["ROUTE_ID", "class_id"]].merge(feas, on="ROUTE_ID").groupby("class_id")["unobserved_share"].median()
    print("유형별 AI 추정 구간 비중(노선 중앙값):", ai_by_class.round(3).to_dict())

    section("결과 6.1 조건부 친환경")
    print(f"버스 우위 노선 {int(c['bus_better'].sum())}/{len(c)} ({c['bus_better'].mean():.1%})")
    print(f"승객·km당 CO2 중앙: 버스 {c['bus_g_per_pkm'].median():.1f}g, 승용차 {c['car_g_per_pkm'].median():.1f}g")
    print(f"손익분기 재차인원 중앙 {c['breakeven_load'].median():.2f}명, 실제 재차인원 중앙 {c['avg_load'].median():.2f}명")
    worse = c[~c["bus_better"]]
    print(f"승용차보다 배출이 많은 노선 {len(worse)}개, 그중 재차인원 2명 미만 {int((worse['avg_load'] < 2).sum())}개")
    g = h.groupby("hour")[["pkm", "vkm"]].sum()
    print("시간대 재차인원(8시·18시·20~22시):", round(g.loc[8, "pkm"] / g.loc[8, "vkm"], 2), round(g.loc[18, "pkm"] / g.loc[18, "vkm"], 2),
          round(g.loc[20:22, "pkm"].sum() / g.loc[20:22, "vkm"].sum(), 2))
    evening = h[h["hour"].between(20, 22)]
    print(f"20~22시 재차인원 3명 미만 운행 비중 {evening.loc[evening['low_load'], 'trips'].sum() / evening['trips'].sum():.1%}")
    low = h[h["low_load"]].merge(c[["ROUTE_ID", "boardings_per_day"]], on="ROUTE_ID")
    low["boardings_per_trip"] = low["board_share"] * low["boardings_per_day"] / low["trips"]
    low = low.sort_values("boardings_per_trip")
    weighted_median = low["boardings_per_trip"][(low["trips"].cumsum() / low["trips"].sum()) >= 0.5].iloc[0]
    print(f"재차인원 3명 미만 운행 1회의 누적 승차 중앙(운행 가중) {weighted_median:.1f}명")
    added = c["added_trip_breakeven_riders"]
    print(f"증편 손익분기 전환인원 중앙 {added.median():.1f}명(사분위 {added.quantile(.25):.1f}~{added.quantile(.75):.1f})")

    section("결과 6.2 감축 시나리오(하루 톤, 연 = × 365)")
    now = c["bus_co2_kg_per_day"].sum() / 1000
    electric = now - c["bus_co2_kg_per_day_all_electric"].sum() / 1000
    offpeak = o["van_offpeak_cut_vs_current_kg_per_day"].sum() / 1000
    low_share = h.loc[h["low_load"], "trips"].sum() / h["trips"].sum()
    two = p[p["class_id"] == 2]
    print(f"현재 배출 {now:.2f}, 전면 전기화 감축 {electric:.2f}({electric / now:.1%}, 연 {electric * 365:,.0f})")
    print(f"km당 CO2 중앙: 전기 마을버스 {c['bus_ev_g_per_km'].median():.0f}g, 소형 전기차 {p['van_g_per_km'].median():.0f}g")
    print(f"3명 미만 시간대 소형화: 운행 {low_share:.1%}, 감축 {offpeak:.2f}(연 {offpeak * 365:,.0f}), 전기화 대비 {offpeak / electric:.1%}, "
          f"차량 크기 효과 {o['van_offpeak_cut_kg_per_day'].sum() / 1000:.2f}")
    print(f"소형 차량 전환 유형 {len(two)}개 중 전환 가능 {int(two['van_feasible'].sum())}개, 감축 "
          f"{p['van_co2_cut_vs_current_kg_per_day'].sum() / 1000:.2f}(연 {p['van_co2_cut_vs_current_kg_per_day'].sum() / 1000 * 365:,.0f})")

    section("결과 6.3 네 유형")
    ops = o[["ROUTE_ID", "deficit_at_900", "route_cars_per_1000"]]
    table = (p.merge(ops, on="ROUTE_ID").merge(rm[["ROUTE_ID", "simple_elderly_affected", "net_elderly_affected"]], on="ROUTE_ID")
             .merge(feas[["ROUTE_ID", "slope_mean", "one_lane_share"]], on="ROUTE_ID"))
    summary = table.groupby("class").agg(
        노선=("ROUTE_ID", "size"), 재차인원=("avg_load", "median"), g_pkm=("bus_g_per_pkm", "median"),
        고령비율=("catchment_elderly_share", "median"), 미충당900=("deficit_at_900", "mean"),
        승용차1000명당=("route_cars_per_1000", "median"), 경사=("slope_mean", "median"), 일차로=("one_lane_share", "median"),
        영향고령_직선=("simple_elderly_affected", "median"), 영향고령_도로망=("net_elderly_affected", "median"))
    print(summary.round(3).to_string())
    high = rm["class_id"].isin([1, 2])
    print(f"영향 고령 인구 중앙: 의존 높은 유형(①②) {rm.loc[high, 'simple_elderly_affected'].median():,.0f}~"
          f"{rm.loc[high, 'net_elderly_affected'].median():,.0f}명, 낮은 유형(③④) {rm.loc[~high, 'simple_elderly_affected'].median():,.0f}~"
          f"{rm.loc[~high, 'net_elderly_affected'].median():,.0f}명")
    rho = p.merge(demand[["ROUTE_ID", "route_cars_per_1000"]], on="ROUTE_ID")[["catchment_elderly_share", "route_cars_per_1000"]]
    print(f"고령 비율과 1,000명당 승용차 순위상관 {rho.corr('spearman').iloc[0, 1]:.2f}")
    print(f"증편 검토 후보 {int(o['carbon_service_candidate'].sum())}개")

    section("노선 사례")
    for name in CASES:
        row = p[p["노선명"] == name].iloc[0]
        r = rm.set_index("ROUTE_ID").loc[row["ROUTE_ID"]]
        f = feas.set_index("ROUTE_ID").loc[row["ROUTE_ID"]]
        print(f"{name}: 경사 {f['slope_mean']:.1f}%, {row['length_km']:.1f}km, 하루 {row['trips_per_day']:.0f}회·{row['boardings_per_day']:,.0f}명, "
              f"{row['bus_g_per_pkm']:.0f}g/승객km, 구간 최대 재차 {row['avg_peak_load_per_trip']:.1f}명, "
              f"소형 전환 감축 {row['van_co2_cut_vs_current_kg_per_day']:.0f}kg/일 | 제거 시 영향 고령 {r['net_elderly_affected']:,.0f}명 "
              f"(평균 +{r['net_added_min_mean']:.1f}분), 새로 10분 초과 {r['simple_elderly_over_10min_new']:,.0f}~{r['net_elderly_over_10min_new']:,.0f}명")


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
