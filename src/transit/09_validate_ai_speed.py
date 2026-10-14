"""AI(B/C 모델)로 채운 속도가 실제 마을버스 운행 속도와 비슷한 수준·시간대 흐름인지 확인한다.

입력
  data/raw/traffic/통행속도 노선 통계.csv  (TOPIS 버스 노선유형별 시간대별 공간속도, 2026.10.6~8 화~목)
  data/processed/transit/link_speed_weekday_profile.parquet(03), route_links.csv(01), route_feasibility.csv(03)
  운행횟수(route_data.py)

처리
  1. 실측: 노선유형 2(마을버스), 시간대 107~122(=07~22시). 3일을 합쳐 여행거리 누계 ÷ 여행시간 누계로
     시간대 속도를 낸다(공간속도와 같은 정의). 값 앞에 붙은 BOM 문자는 지운다.
  2. 모델: 마을버스 노선 링크의 같은 시각 평일 속도(시간코드 = 시 + 1)를 노선 하루 운행횟수 × 링크 길이로
     가중한 조화평균(= 총 거리 ÷ 총 시간). 실측도 버스가 지나간 구간마다 쌓이므로 운행 많은 길에 가중된다.
  3. 비교: 시간대별 차이, 평균 차이(편향), 평균 절대 차이, 시간대 흐름 상관.
     실측은 10월·버스, 모델은 7월·일반 차량 속도라 수준이 완전히 같을 필요는 없고, 비슷한 범위와 같은
     출퇴근 흐름을 보이는지를 본다.
  4. 마을버스 노선에서 AI 추정(미관측) 구간이 차지하는 비중(03의 unobserved_share)을 함께 정리한다.

출력
  data/processed/transit/ai_speed_validation.csv  시간대별 실측·모델 속도

사용: python src/transit/09_validate_ai_speed.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from route_data import route_trips_per_day

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRANSIT = PROJECT_ROOT / "data" / "processed" / "transit"
ROUTE_SPEED_CSV = PROJECT_ROOT / "data" / "raw" / "traffic" / "통행속도 노선 통계.csv"
VILLAGE_TYPE = "2"
HOURS = range(7, 23)
AI_SHARE_FLAG = 0.3


def observed_speed() -> pd.Series:
    raw = pd.read_csv(ROUTE_SPEED_CSV, encoding="utf-8-sig", dtype=str)
    raw.columns = [c.replace("﻿", "").strip() for c in raw.columns]
    raw = raw.apply(lambda s: s.str.replace("﻿", "", regex=False).str.strip())
    raw = raw[raw["노선유형"] == VILLAGE_TYPE].copy()
    for col in ("여행거리누계", "여행시간누계"):
        raw[col] = pd.to_numeric(raw[col])
    raw["hour"] = pd.to_numeric(raw["시간대"]) - 100
    sums = raw[raw["hour"].isin(HOURS)].groupby("hour")[["여행거리누계", "여행시간누계"]].sum()
    return (sums["여행거리누계"] / sums["여행시간누계"] * 3.6).rename("observed_kmh")


def model_speed() -> pd.Series:
    links = pd.read_csv(TRANSIT / "route_links.csv", dtype={"ROUTE_ID": str, "LINK_ID": str})
    links = links[links["kind"] == "마을버스"]
    trips = route_trips_per_day(set(links["ROUTE_ID"]))
    links = links.join(trips, on="ROUTE_ID")
    if links["trips_per_day"].isna().any():
        print(f"  운행횟수가 없는 노선 {links.loc[links['trips_per_day'].isna(), 'ROUTE_ID'].nunique()}개는 제외")
        links = links.dropna(subset=["trips_per_day"])
    codes = [h + 1 for h in HOURS]
    profile = pd.read_parquet(TRANSIT / "link_speed_weekday_profile.parquet", columns=["LINK_ID", "시간코드", "speed_kmh"])
    frame = links.merge(profile[profile["시간코드"].isin(codes)], on="LINK_ID")
    frame["w"] = frame["LENGTH"] * frame["trips_per_day"]
    frame["t"] = frame["w"] / frame["speed_kmh"]
    sums = frame.groupby("시간코드")[["w", "t"]].sum()
    speed = (sums["w"] / sums["t"]).rename("model_kmh")
    speed.index = speed.index - 1
    return speed.rename_axis("hour")


def main() -> None:
    print("[1/2] 실측(TOPIS 마을버스)과 모델 속도...")
    table = pd.concat([observed_speed(), model_speed()], axis=1, join="inner")
    table["diff_kmh"] = table["model_kmh"] - table["observed_kmh"]
    table.to_csv(TRANSIT / "ai_speed_validation.csv", encoding="utf-8-sig")
    print(table.round(1).T.to_string())
    print(f"  범위: 실측 {table['observed_kmh'].min():.1f}~{table['observed_kmh'].max():.1f}km/h, "
          f"모델 {table['model_kmh'].min():.1f}~{table['model_kmh'].max():.1f}km/h")
    print(f"  평균 차이(모델−실측) {table['diff_kmh'].mean():+.1f}km/h, 평균 절대 차이 {table['diff_kmh'].abs().mean():.1f}km/h, "
          f"시간대 흐름 상관 {np.corrcoef(table['observed_kmh'], table['model_kmh'])[0, 1]:.2f}")

    print("[2/2] 마을버스 노선의 AI 추정 구간 비중...")
    feas = pd.read_csv(TRANSIT / "route_feasibility.csv", dtype={"ROUTE_ID": str})
    village = feas[feas["kind"] == "마을버스"]
    share = village["unobserved_share"]
    print(f"  노선 {len(village)}개: 평균 {share.mean():.0%}, 중앙 {share.median():.0%}, 최대 {share.max():.0%}, "
          f"{AI_SHARE_FLAG:.0%} 이상 노선 {int((share >= AI_SHARE_FLAG).sum())}개")


if __name__ == "__main__":
    main()
