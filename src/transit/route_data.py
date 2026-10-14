"""여러 단계가 함께 쓰는 입력 준비 함수.

- load_route_stops: 노선별 정류소 엑셀(OA-1095, 읽는 데 수 초)을 parquet로 캐시한다.
- load_trip_counts: TOPIS 노선·정류장별 운행횟수(OA-21220, 원자료 약 320MB)에서 마을버스·자율주행
  노선 행만 골라 캐시한다. 일 합계가 빈 행(시간대 칸도 모두 빈 행 = 그날 운행 기록 없음)은 빼고,
  남은 행의 시간대 빈칸은 그 시간 운행 0회로 채운다(시간대 합 = 일 합계).
- route_trips_per_day: 노선별 하루 운행횟수(05 탄소 비교, 07 시간대별 탑승이 같은 정의를 쓴다).
- load_stop_ridership: 승하차(OA-12913, 7월 월합계)를 노선번호×정류장별 하루 평균(시간대별)으로.
- load_route_link_hours: 노선 링크 × 평일 운행 시간대 속도·경사 표(04·05·06 공통 입력).
- load_dong_names: 행정동코드(8자리) → '구 동' 이름.

캐시는 원자료 파일보다 오래됐으면 자동으로 다시 만든다(원자료가 없으면 있는 캐시를 그대로 쓴다).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_TRAFFIC = PROJECT_ROOT / "data" / "raw" / "traffic"
TRANSIT = PROJECT_ROOT / "data" / "processed" / "transit"
ROUTE_STOP_XLSX = RAW_TRAFFIC / "서울시버스노선별정류소정보(20260902).xlsx"
RIDERSHIP_CSV = RAW_TRAFFIC / "2026년_버스노선별_정류장별_시간대별_승하차_인원_정보(07월).csv"
TRIP_DIR = RAW_TRAFFIC / "bus_trips"
REGISTRATION_CSV = RAW_TRAFFIC / "서울시 자치구 읍면동별 연료별 자동차 등록현황(행정동)(26년7월).csv"
ROUTE_STOP_CACHE = TRANSIT / "route_stops.parquet"
TRIP_CACHE = TRANSIT / "bus_trip_counts.parquet"

DAYS_IN_MONTH = 31                      # 승하차 7월 월합계 → 하루 평균
TRIP_WINDOW = ("2026-07-06", "2026-07-31")  # 운행횟수 파일이 덮는 기간(승하차와 같은 7월)
OPERATING_HOURS = range(8, 23)          # 속도 프로필 시간코드 8~22 = 07~21시
HOUR_COLS = [f"버스운행횟수_{h:02d}시" for h in range(24)]


def is_stale(cache: Path, sources: list[Path]) -> bool:
    """캐시가 없거나 원자료 중 하나라도 캐시보다 새로우면 True. 원자료가 없으면 캐시를 믿는다."""
    if not cache.exists():
        return True
    return any(p.exists() and p.stat().st_mtime > cache.stat().st_mtime for p in sources)


def load_route_stops() -> pd.DataFrame:
    """노선별 정류소 표(ROUTE_ID, 노선명, 순번, NODE_ID, ARS_ID, 정류소명, X좌표, Y좌표)."""
    if is_stale(ROUTE_STOP_CACHE, [ROUTE_STOP_XLSX]):
        stops = pd.read_excel(ROUTE_STOP_XLSX, dtype={"ROUTE_ID": str, "노선명": str, "NODE_ID": str, "ARS_ID": str})
        ROUTE_STOP_CACHE.parent.mkdir(parents=True, exist_ok=True)
        stops.to_parquet(ROUTE_STOP_CACHE, index=False)
        return stops
    return pd.read_parquet(ROUTE_STOP_CACHE)


def load_trip_counts() -> pd.DataFrame:
    """마을버스·자율주행 노선의 날짜×노선×정류장 운행횟수(일 합계와 시간대별)."""
    sources = sorted(TRIP_DIR.glob("tpss_sta_route_hturn_*.csv"))
    if not is_stale(TRIP_CACHE, sources):
        return pd.read_parquet(TRIP_CACHE)
    if not sources:
        raise SystemExit(f"운행횟수 원자료가 없습니다: {TRIP_DIR}")
    from route_kinds import classify_route  # route_kinds가 이 모듈을 쓰므로 순환 import를 피한다.

    stops = load_route_stops()
    target_ids = set(stops.loc[classify_route(stops["노선명"]) != "일반", "ROUTE_ID"])
    parts = []
    for path in sources:
        frame = pd.read_csv(path, encoding="cp949", dtype={"기준_날짜": str, "노선_ID": str, "정류장_ID": str},
                            usecols=["기준_날짜", "노선_ID", "정류장_ID", "버스운행횟수", *HOUR_COLS])
        parts.append(frame[frame["노선_ID"].isin(target_ids)])
    trips = pd.concat(parts, ignore_index=True)
    trips["버스운행횟수"] = pd.to_numeric(trips["버스운행횟수"], errors="coerce")
    trips = trips.dropna(subset=["버스운행횟수"]).reset_index(drop=True)
    trips[HOUR_COLS] = trips[HOUR_COLS].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    TRIP_CACHE.parent.mkdir(parents=True, exist_ok=True)
    trips.to_parquet(TRIP_CACHE, index=False)
    print(f"  운행횟수 캐시 생성: {len(trips):,}행 → {TRIP_CACHE}")
    return trips


def route_trips_per_day(route_ids: set[str], hourly: bool = False) -> pd.Series | pd.DataFrame:
    """노선별 하루 운행횟수: 날짜마다 정류장별 운행횟수의 중앙값을 취해 TRIP_WINDOW 평균.

    정류장 중앙값을 쓰는 이유: 기점·종점 부근 정류장은 회차·출입고 운행이 섞여 값이 튄다.
    hourly=True면 시간대(0~23시)별로 같은 계산을 한 표를 돌려준다.
    """
    trips = load_trip_counts()
    trips = trips[trips["노선_ID"].isin(route_ids)]
    trips = trips[pd.to_datetime(trips["기준_날짜"], format="%Y%m%d").between(*TRIP_WINDOW)]
    cols = HOUR_COLS if hourly else ["버스운행횟수"]
    per_day = trips.groupby(["노선_ID", "기준_날짜"])[cols].median().groupby("노선_ID").mean()
    if hourly:
        per_day.columns = range(24)
        return per_day
    return per_day["버스운행횟수"].rename("trips_per_day")


def load_stop_ridership() -> pd.DataFrame:
    """노선번호(ride_key)×정류장(NODE_ID)별 하루 평균 승차·하차. 열: board_0..23, alight_0..23."""
    ride = pd.read_csv(RIDERSHIP_CSV, encoding="cp949", low_memory=False, dtype={"노선번호": str, "표준버스정류장ID": str})
    renamed = {}
    for col in ride.columns:
        for kind, label in (("board", "승차총승객수"), ("alight", "하차총승객수")):
            if col.endswith(label):
                renamed[col] = f"{kind}_{int(col.removesuffix(label).removesuffix('시'))}"
    if sorted(renamed.values()) != sorted(f"{k}_{h}" for k in ("board", "alight") for h in range(24)):
        raise SystemExit("승하차 파일의 시간대 열(00~23시 승차·하차)이 예상과 다릅니다.")
    ride = ride.rename(columns=renamed).rename(columns={"노선번호": "ride_key", "표준버스정류장ID": "NODE_ID"})
    ride["ride_key"] = ride["ride_key"].str.strip()
    value_cols = [f"{k}_{h}" for k in ("board", "alight") for h in range(24)]
    out = ride.groupby(["ride_key", "NODE_ID"], as_index=False)[value_cols].sum()
    out[value_cols] = out[value_cols] / DAYS_IN_MONTH
    return out


def load_dong_names() -> pd.Series:
    """행정동코드(8자리) → '구 동' 이름(자동차 등록현황 파일의 행정동 표기)."""
    reg = pd.read_csv(REGISTRATION_CSV, encoding="cp949", dtype={"동_코드": str}, usecols=["동_코드", "읍면동(행정동)"])
    reg = reg[reg["읍면동(행정동)"].str.strip() != "기타"]
    return reg.assign(code=reg["동_코드"].str[:8]).drop_duplicates("code").set_index("code")["읍면동(행정동)"].str.strip()


def load_route_link_hours(kind: str | None = None) -> pd.DataFrame:
    """노선 링크 × 평일 운행 시간대 표: LENGTH, 경사(역방향 통과는 부호 반전), 시간대 속도."""
    links = pd.read_csv(TRANSIT / "route_links.csv", dtype={"ROUTE_ID": str, "LINK_ID": str})
    if kind is not None:
        links = links[links["kind"] == kind]
    dimension = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "seoul_link_dimension.csv",
                            usecols=["LINK_ID", "slope_pct"], dtype={"LINK_ID": str})
    links = links.merge(dimension, on="LINK_ID", how="left")
    links["slope_pct"] = np.where(links["reversed"], -1, 1) * links["slope_pct"].fillna(0)
    profile = pd.read_parquet(TRANSIT / "link_speed_weekday_profile.parquet", columns=["LINK_ID", "시간코드", "speed_kmh"])
    profile = profile[profile["시간코드"].isin(OPERATING_HOURS)]
    covered = profile.groupby("LINK_ID")["speed_kmh"].count().reindex(links["LINK_ID"].unique(), fill_value=0)
    gaps = int((covered < len(OPERATING_HOURS)).sum())
    if gaps:
        raise SystemExit(f"속도 프로필에 운행 시간대 속도가 빠진 노선 링크가 {gaps}개 있습니다. 03을 다시 실행하세요.")
    frame = links.merge(profile, on="LINK_ID", how="inner")
    frame["speed_kmh"] = frame["speed_kmh"].clip(lower=3.0)
    return frame
