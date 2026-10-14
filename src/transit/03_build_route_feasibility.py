"""노선별 평일 시간대 속도 프로필과 도로 조건 지표를 만든다.

입력
  data/processed/transit/{routes,route_links}.csv           (01 산출물)
  data/processed/all_link_speed_prediction_*.csv            (models/04: A 조회표 + B/C 예측 전 링크 속도)
  data/processed/seoul_link_dimension.csv

처리
  1. 평일 시간대별 링크 속도 프로필: 예측 파일 중 평일만 골라 링크×시간 평균을 낸다(04~07의 속도 입력).
     마을버스가 다니는 골목은 상당수가 미관측(B/C)이므로 AI 예측 속도가 필수다.
  2. 노선 지표(링크 길이 가중):
       slope_mean         진행 방향을 반영한 절대 경사 평균 [%](±12% 제한)
       one_lane_share     1차로 링크 길이 비중
       unobserved_share   AI 추정(B/C) 구간 비중
       day_speed_kmh      평일 낮(10~16시) 평균 통행속도

출력 (data/processed/transit/)
  link_speed_weekday_profile.parquet  링크×시간코드 평일 평균 속도(+tier)
  route_feasibility.csv               노선 1행: 길이와 도로 조건 지표

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

DAYTIME = range(11, 18)        # 시간코드 11~17 = 10~16시
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


def route_indicators(links: pd.DataFrame, speed_wide: pd.DataFrame) -> dict[str, float]:
    w = links["LENGTH"].to_numpy()
    km = w.sum() / 1000.0
    slope = links["slope_signed"].abs().clip(upper=SLOPE_CLIP).to_numpy()
    speeds = speed_wide.reindex(links["LINK_ID"].to_numpy())
    day_speed = speeds[[h for h in DAYTIME if h in speeds.columns]].mean(axis=1).to_numpy()
    return {
        "length_km": km,
        "slope_mean": float(np.average(slope, weights=w)),
        "one_lane_share": float(np.average(links["LANES"].to_numpy() <= 1, weights=w)),
        "unobserved_share": float(np.average(links["tier"].to_numpy() != "A_lookup", weights=w)),
        "day_speed_kmh": float(km / (w / (day_speed / 3.6)).sum() * 3600),
    }


def main() -> None:
    print("[1/3] 평일 시간대별 전 링크 속도 프로필...")
    profile = build_weekday_profile()
    speed_wide = profile.pivot(index="LINK_ID", columns="시간코드", values="speed_kmh")
    tiers = profile.drop_duplicates("LINK_ID").set_index("LINK_ID")["tier"]

    print("[2/3] 노선 링크에 도로 속성 결합...")
    routes = pd.read_csv(TRANSIT / "routes.csv", dtype={"ROUTE_ID": str})
    route_links = pd.read_csv(TRANSIT / "route_links.csv", dtype={"ROUTE_ID": str, "LINK_ID": str})
    dimension = pd.read_csv(PROCESSED / "seoul_link_dimension.csv", usecols=["LINK_ID", "LANES", "slope_pct"],
                            dtype={"LINK_ID": str})
    route_links = route_links.merge(dimension, on="LINK_ID", how="left")
    # 역방향으로 통과한 링크는 오르막·내리막이 뒤집힌다.
    route_links["slope_signed"] = np.where(route_links["reversed"], -1, 1) * route_links["slope_pct"].fillna(0)
    route_links["tier"] = route_links["LINK_ID"].map(tiers)

    print("[3/3] 노선 지표 계산...")
    rows = []
    for (route_id, name, kind), links in route_links.groupby(["ROUTE_ID", "노선명", "kind"], sort=False):
        rows.append({"ROUTE_ID": route_id, "노선명": name, "kind": kind, **route_indicators(links, speed_wide)})
    feas = pd.DataFrame(rows).merge(routes[["ROUTE_ID", "qa_flag"]], on="ROUTE_ID", how="left")
    feas.to_csv(TRANSIT / "route_feasibility.csv", index=False, encoding="utf-8-sig")

    village = feas[feas["kind"] == "마을버스"]
    print(f"  마을버스 {len(village)}개 중앙값: 경사 {village['slope_mean'].median():.2f}%, "
          f"1차로 {village['one_lane_share'].median():.0%}, AI 추정 구간 {village['unobserved_share'].median():.0%}, "
          f"낮 속도 {village['day_speed_kmh'].median():.1f}km/h")
    print(f"  저장: {TRANSIT}")


if __name__ == "__main__":
    main()
