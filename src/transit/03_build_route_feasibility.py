"""노선별 자율주행 운행 적합도를 기존 자율주행 시범노선의 조건과 비교해 평가한다.

입력
  data/processed/transit/{routes,route_links}.csv           (01 산출물)
  data/processed/all_link_speed_prediction_*.csv            (models/04: A 조회표 + B/C 예측 전 링크 속도)
  data/processed/seoul_link_dimension.csv

처리
  1. 평일 시간대별 링크 속도 프로필: 예측 파일 중 평일만 골라 링크×시간 평균을 낸다.
     자율주행 노선이 다니는 골목은 상당수가 미관측(B/C)이므로 예측 속도가 필수다.
  2. 노선 지표(모두 '클수록 자율주행에 어려운 조건'):
       slope_p90          경사(절대값, ±12% 클립) 90분위 [%]
       slope_mean         길이 가중 평균 경사 [%]
       one_lane_share     1차로 링크 길이 비중
       fast_mixed_share   평일 낮(10~16시) 일반차량 속도 40km/h 초과 구간 비중
       congestion_ratio   평일 07~21시 중 가장 느린 시간 / 가장 빠른 시간 노선 통행시간
       bump_per_km        과속방지턱 인접 링크 수 / km
       school_share       학교·보호구역 200m 이내 링크 길이 비중
     참고 지표: unobserved_share(B/C 예측 구간 비중), 평일 낮 평균 통행속도.
  3. 기준 노선: 낮 시간 생활권형 자율주행 노선(동작A01·서대문A01·동대문A01·상암A21·
     청계A01·청와대A01). 새벽·심야 노선은 간선도로 야간 운행이라 조건이 달라 제외한다.
     지표별 '검증된 범위' = 기준 노선 중 가장 어려운 값. 마을버스 노선이 이 값을 넘으면
     그 지표는 아직 실제 운행으로 검증되지 않은 조건으로 표시한다.

출력 (data/processed/transit/)
  link_speed_weekday_profile.parquet  링크×시간코드 평일 평균 속도(+tier)
  route_feasibility.csv               노선 1행: 지표, 기준 초과 지표 목록, 검증 범위 내 지표 수
  feasibility_envelope.csv            지표별 기준 노선 값과 검증 범위

사용: python src/transit/03_build_route_feasibility.py
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = PROJECT_ROOT / "data" / "processed"
TRANSIT = PROCESSED / "transit"
PROFILE_PATH = TRANSIT / "link_speed_weekday_profile.parquet"

BENCHMARK_ROUTES = ["동작A01", "서대문A01", "동대문A01", "상암A21", "청계A01", "청와대A01"]
INDICATORS = ["slope_p90", "slope_mean", "one_lane_share", "fast_mixed_share", "congestion_ratio", "bump_per_km", "school_share"]
INDICATOR_LABELS = {
    "slope_p90": "급경사", "slope_mean": "평균경사", "one_lane_share": "1차로 비중",
    "fast_mixed_share": "고속 혼재 교통", "congestion_ratio": "시간대 혼잡 변동",
    "bump_per_km": "과속방지턱 밀도", "school_share": "학교 주변 비중",
}
DAYTIME = range(11, 18)        # 시간코드 11~17 = 10~16시
OPERATING = range(8, 23)       # 시간코드 8~22 = 07~21시
FAST_MIXED_KMH = 40.0
SLOPE_CLIP = 12.0


def build_weekday_profile(path: Path = PROFILE_PATH) -> pd.DataFrame:
    if path.exists():
        return pd.read_parquet(path)
    pattern = re.compile(r"all_link_speed_prediction_(\d{12})\.csv$")
    files = [f for f in sorted(PROCESSED.glob("all_link_speed_prediction_*.csv"))
             if (m := pattern.search(f.name)) and pd.Timestamp(m.group(1)[:8]).dayofweek < 5]
    if not files:
        raise SystemExit("평일 예측 속도 파일이 없습니다. 먼저 src/models/04_batch_predict_all_link_speed.py를 실행하세요.")
    # 파일 하나가 한 시각(전 링크)이므로 시간코드별로 링크 속도 합과 파일 수를 누적한다.
    sums: dict[int, pd.Series] = {}
    counts: dict[int, pd.Series] = {}
    tiers: pd.Series | None = None
    for f in files:
        frame = pd.read_csv(f, usecols=["LINK_ID", "시간코드", "speed_kmh", "tier"], dtype={"LINK_ID": str})
        if tiers is None:
            tiers = frame.drop_duplicates("LINK_ID").set_index("LINK_ID")["tier"]
        for code, part in frame.groupby("시간코드"):
            speed = part.set_index("LINK_ID")["speed_kmh"]
            ones = pd.Series(1.0, index=speed.index)
            code = int(code)
            sums[code] = speed if code not in sums else sums[code].add(speed, fill_value=0)
            counts[code] = ones if code not in counts else counts[code].add(ones, fill_value=0)
    profile = (pd.concat({code: sums[code] / counts[code] for code in sorted(sums)}, names=["시간코드", "LINK_ID"])
               .rename("speed_kmh").reset_index()[["LINK_ID", "시간코드", "speed_kmh"]])
    profile["tier"] = profile["LINK_ID"].map(tiers)
    profile.to_parquet(path, index=False)
    print(f"  평일 예측 파일 {len(files)}개로 프로필 생성")
    return profile


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values)
    cum = np.cumsum(weights[order])
    return float(values[order][np.searchsorted(cum, q * cum[-1])])


def route_indicators(links: pd.DataFrame, speed_wide: pd.DataFrame) -> dict[str, float]:
    w = links["LENGTH"].to_numpy()
    km = w.sum() / 1000.0
    slope = links["slope_signed"].abs().clip(upper=SLOPE_CLIP).to_numpy()
    speeds = speed_wide.reindex(links["LINK_ID"].to_numpy())
    day_speed = speeds[[h for h in DAYTIME if h in speeds.columns]].mean(axis=1).to_numpy()
    op_cols = [h for h in OPERATING if h in speeds.columns]
    hourly_time = (w[:, None] / (speeds[op_cols].to_numpy() / 3.6)).sum(axis=0)
    return {
        "length_km": km,
        "slope_p90": weighted_quantile(slope, w, 0.9),
        "slope_mean": float(np.average(slope, weights=w)),
        "one_lane_share": float(np.average(links["LANES"].to_numpy() <= 1, weights=w)),
        "fast_mixed_share": float(np.average(np.nan_to_num(day_speed) > FAST_MIXED_KMH, weights=w)),
        "congestion_ratio": float(hourly_time.max() / hourly_time.min()),
        "bump_per_km": float((links["speed_bump_count"].to_numpy() > 0).sum() / km),
        "school_share": float(np.average(links["school_zone_count"].to_numpy() > 0, weights=w)),
        "unobserved_share": float(np.average(links["tier"].to_numpy() != "A_lookup", weights=w)),
        "day_speed_kmh": float(km / (w / (day_speed / 3.6)).sum() * 3600),
    }


def main() -> None:
    print("[1/4] 평일 시간대별 전 링크 속도 프로필...")
    profile = build_weekday_profile()
    speed_wide = profile.pivot(index="LINK_ID", columns="시간코드", values="speed_kmh")
    tiers = profile.drop_duplicates("LINK_ID").set_index("LINK_ID")["tier"]

    print("[2/4] 노선 링크에 도로 속성 결합...")
    routes = pd.read_csv(TRANSIT / "routes.csv", dtype={"ROUTE_ID": str})
    route_links = pd.read_csv(TRANSIT / "route_links.csv", dtype={"ROUTE_ID": str, "LINK_ID": str})
    dimension = pd.read_csv(
        PROCESSED / "seoul_link_dimension.csv",
        usecols=["LINK_ID", "LANES", "slope_pct", "speed_bump_count", "school_zone_count"], dtype={"LINK_ID": str},
    )
    route_links = route_links.merge(dimension, on="LINK_ID", how="left")
    # 역방향으로 통과한 링크는 오르막·내리막이 뒤집힌다.
    route_links["slope_signed"] = np.where(route_links["reversed"], -1, 1) * route_links["slope_pct"].fillna(0)
    route_links["tier"] = route_links["LINK_ID"].map(tiers)

    print("[3/4] 노선 지표 계산...")
    rows = []
    for (route_id, name, kind), links in route_links.groupby(["ROUTE_ID", "노선명", "kind"], sort=False):
        rows.append({"ROUTE_ID": route_id, "노선명": name, "kind": kind, **route_indicators(links, speed_wide)})
    feas = pd.DataFrame(rows).merge(routes[["ROUTE_ID", "qa_flag"]], on="ROUTE_ID", how="left")

    bench = feas[feas["노선명"].isin(BENCHMARK_ROUTES)]
    missing = sorted(set(BENCHMARK_ROUTES) - set(bench["노선명"]))
    if missing:
        raise SystemExit(f"기준 노선이 재구성 결과에 없습니다: {missing}")
    envelope = bench[INDICATORS].max()
    env_table = bench.set_index("노선명")[INDICATORS].T
    env_table["검증범위(최대)"] = envelope
    env_table.insert(0, "지표", [INDICATOR_LABELS[i] for i in env_table.index])
    env_table.to_csv(TRANSIT / "feasibility_envelope.csv", encoding="utf-8-sig")

    exceed = feas[INDICATORS].gt(envelope)
    feas["n_within_envelope"] = (~exceed).sum(axis=1)
    feas["exceeded"] = exceed.apply(lambda r: "|".join(INDICATOR_LABELS[i] for i in INDICATORS if r[i]), axis=1)
    feas.to_csv(TRANSIT / "route_feasibility.csv", index=False, encoding="utf-8-sig")

    print("[4/4] 요약")
    print(env_table.round(3).to_string())
    village = feas[feas["kind"] == "마을버스"]
    print(f"\n  마을버스 {len(village)}개: 모든 지표가 검증 범위 내 {(village['n_within_envelope'] == len(INDICATORS)).sum()}개")
    print("  지표별 검증 범위 초과 노선 수:", {INDICATOR_LABELS[i]: int(exceed.loc[village.index, i].sum()) for i in INDICATORS})
    print("  검증 범위 내 지표 수 분포:", village["n_within_envelope"].value_counts().sort_index().to_dict())
    print(f"  저장: {TRANSIT}")


if __name__ == "__main__":
    main()
