"""마을버스·자율주행 노선의 수요(필요도) 지표를 만든다.

입력
  data/processed/transit/{bus_stops,routes}.csv           (01 산출물)
  data/raw/traffic/서울시버스노선별정류소정보(20260902).xlsx  (노선-정류소 소속)
  data/raw/traffic/2026년_버스노선별_정류장별_시간대별_승하차_인원_정보(07월).csv
  data/raw/traffic/250_LOCAL_RESD_202607/            (250m 격자 생활인구, 일별)
  data/raw/traffic/250_LOCAL_RESD_ADMDONG_202607/    (행정동 생활인구, 일별·연령별)
  data/raw/traffic/match/match.shp                   (250m 격자 CELL_ID -> polygon)
  data/raw/traffic/서울시 자치구 읍면동별 연료별 자동차 등록현황(행정동)(26년7월).csv
  생활인구 결과(grid_population_daytime.csv, dong_population_night.csv)는 캐시로 쓰고, 원자료가 더
  새롭거나 --rebuild를 주면 다시 만든다. 원자료가 없으면 있는 캐시를 그대로 쓴다.

처리
  1. 승하차: 7월 월합계를 31일로 나눠 노선·정류소별 일평균 승차·하차를 만든다.
     마을버스는 '노선번호'가 정류소 파일의 '노선명'과 같다(예: 강남01).
  2. 고령 생활인구: 격자 연령별 칸은 소수 인원이 '*'로 가려져 있어 그대로 합하면
     과소 추정된다. 대신 격자 총인구에 같은 일자·시간 행정동의 65세 이상 비율을
     곱해 추정한다. 평일(7/7~7/20 중 월~금) 10~16시 평균을 낮 시간 대표값으로 쓴다.
     분기 노선(서대문02대·02소 등)은 승하차가 합쳐져 있어 route_kinds.py 규칙으로 정류장별로 나눈다.
  3. 노선 단위로 정류소 반경 300m 격자(중복 제거)의 낮 시간 인구·고령 인구를 합산하고,
     지하철역 500m 밖이면서 일반버스 정류소 200m 안에도 없는 '대체수단 없는 정류소'
     비율과 그 정류소들의 승차 비중을 계산한다.
  4. 노선 주변 승용차 보유: 행정동별 자가용 승용차 등록대수(2026.7) ÷ 평일 새벽(03~04시) 생활인구
     (주민 수 근사) × 1,000. 정류소마다 가장 가까운 격자의 대표 행정동(낮 인구가 가장 많이 잡힌
     행정동) 값을 붙여 노선 정류소 평균을 낸다.

출력 (data/processed/transit/)
  grid_population_daytime.csv  격자 1행: 중심좌표(EPSG:5179), 낮 인구, 고령 인구, 고령 비율, 대표 행정동
  dong_population_night.csv    행정동 1행: 평일 새벽(03~04시) 평균 생활인구
  stop_ridership.csv           노선×정류소 일평균 승하차
  route_demand.csv             노선 1행 수요 지표

사용: python src/transit/02_build_route_demand.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from route_data import REGISTRATION_CSV, is_stale, load_route_stops, load_stop_ridership
from route_kinds import branch_stop_weights, ridership_key

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_TRAFFIC = PROJECT_ROOT / "data" / "raw" / "traffic"
TRANSIT = PROJECT_ROOT / "data" / "processed" / "transit"
GRID_POP_DIR = RAW_TRAFFIC / "250_LOCAL_RESD_202607"
DONG_POP_DIR = RAW_TRAFFIC / "250_LOCAL_RESD_ADMDONG_202607"
GRID_SHP = RAW_TRAFFIC / "match" / "match.shp"
GRID_CACHE = TRANSIT / "grid_population_daytime.csv"
NIGHT_CACHE = TRANSIT / "dong_population_night.csv"

DAYTIME_HOURS = range(10, 17)
NIGHT_HOURS = (3, 4)
STOP_CATCHMENT_M = 300.0
STATION_WALK_M = 500.0
ELDERLY_COLS = ["남자 65~69세", "남자 70세 이상", "여자 65~69세", "여자 70세 이상"]


def is_weekday(date_text: pd.Series) -> pd.Series:
    return pd.to_datetime(date_text.astype(str), format="%Y%m%d").dt.dayofweek < 5


def build_population() -> tuple[pd.DataFrame, pd.DataFrame]:
    """격자별 평일 낮(10~16시) 평균 생활인구·고령 인구 추정치와 행정동별 평일 새벽 인구."""
    dong_frames, night_frames = [], []
    for path in sorted(DONG_POP_DIR.glob("*.csv")):
        dong = pd.read_csv(path, encoding="cp949", usecols=["일자", "시간", "행정동코드", "생활인구합계", *ELDERLY_COLS])
        weekday = is_weekday(dong["일자"])
        night = dong[weekday & dong["시간"].isin(NIGHT_HOURS)]
        night_frames.append(pd.DataFrame({"행정동코드": night["행정동코드"].astype(str).str.strip(),
                                          "night_pop": night["생활인구합계"]}))
        dong = dong[weekday & dong["시간"].isin(DAYTIME_HOURS)]
        elderly = dong[ELDERLY_COLS].apply(pd.to_numeric, errors="coerce").sum(axis=1)
        dong_frames.append(pd.DataFrame({
            "일자": dong["일자"].astype(str), "시간": dong["시간"].astype(int),
            "행정동코드": dong["행정동코드"].astype(str).str.strip(),
            "elderly_share": elderly / dong["생활인구합계"].where(dong["생활인구합계"] > 0),
        }))
    dong_share = pd.concat(dong_frames, ignore_index=True)

    grid_frames = []
    for path in sorted(GRID_POP_DIR.glob("*.csv")):
        grid = pd.read_csv(path, encoding="cp949", usecols=["일자", "시간", "행정동코드", "250M격자", "생활인구합계"], dtype=str)
        grid = grid[is_weekday(grid["일자"]) & grid["시간"].astype(int).isin(DAYTIME_HOURS)]
        grid_frames.append(pd.DataFrame({
            "일자": grid["일자"], "시간": grid["시간"].astype(int), "행정동코드": grid["행정동코드"].str.strip(),
            "CELL_ID": grid["250M격자"].str.strip(), "pop": pd.to_numeric(grid["생활인구합계"], errors="coerce"),
        }))
    grid = pd.concat(grid_frames, ignore_index=True).merge(dong_share, on=["일자", "시간", "행정동코드"], how="left")
    grid["elderly"] = grid["pop"] * grid["elderly_share"]
    # 격자가 행정동 경계에 걸치면 행이 나뉘므로 (격자, 일자, 시간)으로 먼저 합친 뒤 시간 평균을 낸다.
    per_hour = grid.groupby(["CELL_ID", "일자", "시간"], as_index=False)[["pop", "elderly"]].sum(min_count=1)
    cells = per_hour.groupby("CELL_ID", as_index=False)[["pop", "elderly"]].mean()
    cells["elderly_share"] = cells["elderly"] / cells["pop"].where(cells["pop"] > 0)
    # 대표 행정동: 경계에 걸친 격자는 낮 인구가 가장 많이 잡힌 행정동으로 둔다.
    by_dong = grid.groupby(["CELL_ID", "행정동코드"], as_index=False)["pop"].sum()
    cell_dong = by_dong.loc[by_dong.groupby("CELL_ID")["pop"].idxmax()].set_index("CELL_ID")["행정동코드"]
    cells["dong"] = cells["CELL_ID"].map(cell_dong)

    geometry = gpd.read_file(GRID_SHP)[["CELL_ID", "geometry"]].to_crs(5179)
    geometry["cx"] = geometry.geometry.centroid.x
    geometry["cy"] = geometry.geometry.centroid.y
    cells = cells.merge(geometry[["CELL_ID", "cx", "cy"]], on="CELL_ID", how="inner")
    night = pd.concat(night_frames, ignore_index=True).groupby("행정동코드", as_index=False)["night_pop"].mean()
    return cells, night


def load_population(rebuild: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = sorted(GRID_POP_DIR.glob("*.csv")) + sorted(DONG_POP_DIR.glob("*.csv"))
    cached = GRID_CACHE.exists() and "dong" in pd.read_csv(GRID_CACHE, nrows=0).columns
    if rebuild or not cached or any(is_stale(c, raw) for c in (GRID_CACHE, NIGHT_CACHE)):
        if not raw:
            raise SystemExit(f"생활인구 원자료({GRID_POP_DIR.name}, {DONG_POP_DIR.name})가 없어 격자 인구를 만들 수 없습니다. "
                             f"--rebuild를 빼거나, 대표 행정동(dong) 열이 있는 {GRID_CACHE.name}와 {NIGHT_CACHE.name}를 받으세요.")
        cells, night = build_population()
        cells.to_csv(GRID_CACHE, index=False, encoding="utf-8-sig")
        night.to_csv(NIGHT_CACHE, index=False, encoding="utf-8-sig")
        return cells, night
    print(f"  캐시 사용: {GRID_CACHE.name}, {NIGHT_CACHE.name} (다시 만들려면 --rebuild)")
    return pd.read_csv(GRID_CACHE, dtype={"dong": str}), pd.read_csv(NIGHT_CACHE, dtype={"행정동코드": str})


def dong_cars_per_1000(night: pd.DataFrame) -> pd.Series:
    """행정동별 주민(평일 새벽 생활인구) 1,000명당 자가용 승용차."""
    reg = pd.read_csv(REGISTRATION_CSV, encoding="cp949", dtype={"동_코드": str})
    reg = reg[reg["읍면동(행정동)"].str.strip() != "기타"]
    # 차종 4열(승용·승합·화물·특수)이 관용·자가용·영업용 순으로 반복되어 pandas가 '승용', '승용.1', '승용.2'로
    # 읽는다. '승용.1'(자가용 승용) 서울 합계 2,655,016대는 서울시 공표 자가용 승용 265.8만 대와 맞는다.
    cars = reg.groupby(reg["동_코드"].str[:8])["승용.1"].sum()
    residents = night.set_index("행정동코드")["night_pop"]
    return (cars / residents * 1000.0).dropna().rename("cars_per_1000")


def stop_ridership_table() -> pd.DataFrame:
    """노선번호×정류소 하루 평균 승차·하차와 가장 붐비는 시간의 승차."""
    ride = load_stop_ridership()
    board = ride.filter(like="board_")
    return pd.DataFrame({"노선명": ride["ride_key"], "NODE_ID": ride["NODE_ID"],
                         "boardings_per_day": board.sum(axis=1), "alightings_per_day": ride.filter(like="alight_").sum(axis=1),
                         "peak_hour_boardings": board.max(axis=1)})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rebuild", action="store_true", help="생활인구 원자료에서 격자·행정동 인구를 다시 만든다")
    args = parser.parse_args()

    print("[1/4] 격자 낮 시간 생활인구·고령 인구 추정...")
    cells, night = load_population(args.rebuild)
    print(f"  격자 {len(cells):,}개, 서울 낮 인구 합 {cells['pop'].sum():,.0f}, 고령 비율 {cells['elderly'].sum() / cells['pop'].sum():.1%}")
    cars = dong_cars_per_1000(night)
    print(f"  자가용 승용차 보유를 구한 행정동 {len(cars)}개 / 생활인구 행정동 {len(night)}개")

    print("[2/4] 정류소별 승하차...")
    ridership = stop_ridership_table()
    ridership.to_csv(TRANSIT / "stop_ridership.csv", index=False, encoding="utf-8-sig")

    print("[3/4] 노선 단위 수요 지표...")
    routes = pd.read_csv(TRANSIT / "routes.csv", dtype={"ROUTE_ID": str})
    stops = pd.read_csv(TRANSIT / "bus_stops.csv", dtype={"NODE_ID": str})
    route_stops = load_route_stops()
    weights = branch_stop_weights(route_stops[route_stops["ROUTE_ID"].isin(routes["ROUTE_ID"])])
    membership = route_stops[["ROUTE_ID", "NODE_ID"]]
    membership = membership[membership["ROUTE_ID"].isin(routes["ROUTE_ID"])].drop_duplicates()

    stop_xy = gpd.GeoSeries(gpd.points_from_xy(stops["X좌표"], stops["Y좌표"]), crs=4326).to_crs(5179)
    stops["sx"], stops["sy"] = stop_xy.x.to_numpy(), stop_xy.y.to_numpy()
    stops["no_alternative"] = (stops["dist_to_station_m"] > STATION_WALK_M) & ~stops["trunk_stop_within_200m"]
    cell_tree = cKDTree(cells[["cx", "cy"]].to_numpy())
    stops["dong"] = cells["dong"].to_numpy()[cell_tree.query(stops[["sx", "sy"]].to_numpy())[1]]
    stops["cars_per_1000"] = stops["dong"].map(cars)

    routes = routes.set_index("ROUTE_ID")
    rows = []
    for route_id, group in membership.merge(stops, on="NODE_ID", how="left").groupby("ROUTE_ID"):
        meta = routes.loc[route_id]
        hits = cell_tree.query_ball_point(group[["sx", "sy"]].dropna().to_numpy(), STOP_CATCHMENT_M)
        cell_idx = sorted({i for hit in hits for i in hit})
        catchment = cells.iloc[cell_idx]
        ride = ridership[(ridership["노선명"] == ridership_key(meta["노선명"])) & ridership["NODE_ID"].isin(group["NODE_ID"])].copy()
        ride["boardings_per_day"] *= [weights.get((route_id, n), 1.0) for n in ride["NODE_ID"]]
        ride = ride.merge(group[["NODE_ID", "no_alternative"]], on="NODE_ID", how="left")
        boardings = ride["boardings_per_day"].sum()
        rows.append({
            "ROUTE_ID": route_id, "노선명": meta["노선명"], "kind": meta["kind"], "length_km": meta["length_km"],
            "n_stops": group["NODE_ID"].nunique(),
            "boardings_per_day": boardings if len(ride) else np.nan,
            "boardings_per_km": boardings / meta["length_km"] if len(ride) and meta["length_km"] > 0 else np.nan,
            "catchment_pop_day": catchment["pop"].sum(),
            "catchment_elderly_day": catchment["elderly"].sum(),
            "catchment_elderly_share": catchment["elderly"].sum() / catchment["pop"].sum() if catchment["pop"].sum() > 0 else np.nan,
            "no_alt_stop_share": group["no_alternative"].mean(),
            "no_alt_boarding_share": (ride.loc[ride["no_alternative"].fillna(False), "boardings_per_day"].sum() / boardings) if boardings > 0 else np.nan,
            "median_dist_to_station_m": group["dist_to_station_m"].median(),
            "route_cars_per_1000": group["cars_per_1000"].mean(),
        })
    demand = pd.DataFrame(rows)
    demand.to_csv(TRANSIT / "route_demand.csv", index=False, encoding="utf-8-sig")

    print("[4/4] 요약")
    village = demand[demand["kind"] == "마을버스"]
    print(village[["boardings_per_day", "boardings_per_km", "catchment_elderly_day", "catchment_elderly_share",
                   "no_alt_stop_share", "no_alt_boarding_share", "median_dist_to_station_m", "route_cars_per_1000"]]
          .describe().round(3).to_string())
    print(f"  고령 비율과 승용차 보유의 순위상관: "
          f"{village[['catchment_elderly_share', 'route_cars_per_1000']].corr('spearman').iloc[0, 1]:.2f}")
    print(f"  승하차 매칭 안 된 마을버스 노선: {village['boardings_per_day'].isna().sum()}개")
    print(f"  저장: {TRANSIT}")


if __name__ == "__main__":
    main()
