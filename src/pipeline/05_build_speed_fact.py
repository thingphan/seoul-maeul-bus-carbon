"""관측속도(02)에 시간에 따라 변하는 변수를 붙여 속도 관측 팩트를 만든다.

1. 레이더 강수: 링크 중심점과 가장 가까운 HSR 격자의 1시간 누적 강수(precip_mm)
2. 시간대별 교통량: View-T 시간대 교통량과 AADT 대비 비율
3. 생활인구: 링크 중심점이 속한 250m 격자 인구(grid_pop)와 그 격자의 대표 행정동 인구(dong_pop)

시간에 따라 변하지 않는 도로 속성은 04의 링크 차원 표에 있고, 모델에서 LINK_ID로 결합한다.
생활인구는 팩트에 있는 날짜만 읽으므로, 기간을 늘리면 생활인구 원본도 그 기간을 포함해야 한다.

입력: data/interim/seoul_link_speed_5min_hourly.csv(02), data/raw/weather(03), View-T 교통량, 생활인구
출력: data/processed/seoul_link_speed_fact.csv(15,094,335행)
사용: python src/pipeline/05_build_speed_fact.py
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

FIVEMIN_PATH = INTERIM_DIR / "seoul_link_speed_5min_hourly.csv"
SEOUL_LINK_SHP = INTERIM_DIR / "seoul_link.shp"
GRID_LON_PATH = RAW_DIR / "weather" / "grid" / "hsr_lon.txt"
GRID_LAT_PATH = RAW_DIR / "weather" / "grid" / "hsr_lat.txt"
RADAR_DIR = RAW_DIR / "weather" / "radar"
GRID_CACHE_PATH = INTERIM_DIR / "weather_hsr_seoul_grid.npz"
LINK_GRID_MAP_PATH = INTERIM_DIR / "link_radar_grid_mapping.csv"
TRAFFIC_VOLUME_XLSX = RAW_DIR / "traffic" / "traffic_volume.xlsx"
GRID_MATCH_SHP = RAW_DIR / "traffic" / "match" / "match.shp"                 # 250m격자 CELL_ID -> geometry
GRID_POP_DIR = RAW_DIR / "traffic" / "250_LOCAL_RESD_202607"                 # 250m 격자 생활인구(일별)
DONG_POP_DIR = RAW_DIR / "traffic" / "250_LOCAL_RESD_ADMDONG_202607"         # 행정동 생활인구(일별)

OUT_CSV = PROCESSED_DIR / "seoul_link_speed_fact.csv"

SEOUL_BBOX = {"lon_min": 126.70, "lon_max": 127.25, "lat_min": 37.35, "lat_max": 37.75}


# ---------- 레이더 강수 ----------

def parse_kma_ascii_grid(path: Path) -> tuple[int, int, np.ndarray]:
    text = path.read_text(encoding="utf-8")
    data_lines = [line for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]
    values = re.findall(r"-?\d+(?:\.\d+)?", "\n".join(data_lines))
    if len(values) < 3:
        raise SystemExit(f"KMA 격자 파일을 파싱할 수 없습니다: {path}")
    width, height = int(values[0]), int(values[1])
    data = np.asarray(values[2:], dtype=np.float32)
    if data.size != width * height:
        raise SystemExit(f"{path.name} 값 개수 불일치")
    return width, height, data.reshape((height, width))


def load_or_build_seoul_grid() -> dict[str, np.ndarray]:
    if GRID_CACHE_PATH.exists():
        loaded = np.load(GRID_CACHE_PATH)
        return {key: loaded[key] for key in loaded.files}

    width, height, lon_grid = parse_kma_ascii_grid(GRID_LON_PATH)
    width_lat, height_lat, lat_grid = parse_kma_ascii_grid(GRID_LAT_PATH)
    if (width, height) != (width_lat, height_lat):
        raise SystemExit("lon/lat 격자 크기가 다릅니다.")

    mask = (
        (lon_grid >= SEOUL_BBOX["lon_min"]) & (lon_grid <= SEOUL_BBOX["lon_max"])
        & (lat_grid >= SEOUL_BBOX["lat_min"]) & (lat_grid <= SEOUL_BBOX["lat_max"])
    )
    rows, cols = np.where(mask)
    flat_indices = rows * width + cols
    GRID_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        GRID_CACHE_PATH, width=np.asarray(width), height=np.asarray(height), rows=rows, cols=cols,
        flat_indices=flat_indices, lon=lon_grid[rows, cols], lat=lat_grid[rows, cols],
    )
    return {"width": np.asarray(width), "height": np.asarray(height), "rows": rows, "cols": cols,
            "flat_indices": flat_indices, "lon": lon_grid[rows, cols], "lat": lat_grid[rows, cols]}


def build_link_grid_mapping(grid: dict[str, np.ndarray], target_link_ids: set) -> pd.DataFrame:
    if LINK_GRID_MAP_PATH.exists():
        cached = pd.read_csv(LINK_GRID_MAP_PATH, dtype={"LINK_ID": str})
        if set(cached["LINK_ID"]) == target_link_ids:
            return cached

    links = gpd.read_file(SEOUL_LINK_SHP)
    links["LINK_ID"] = links["LINK_ID"].astype(str)
    links = links[links["LINK_ID"].isin(target_link_ids)].copy()
    centroids = links.geometry.centroid
    wgs84 = gpd.GeoSeries(centroids, crs=links.crs).to_crs(epsg=4326)

    radar_lon, radar_lat = grid["lon"], grid["lat"]
    records = []
    for link_id, x, y in zip(links["LINK_ID"], wgs84.x, wgs84.y):
        distances = (radar_lon - x) ** 2 + (radar_lat - y) ** 2
        nearest = int(np.argmin(distances))
        records.append({
            "LINK_ID": link_id, "link_lon": x, "link_lat": y,
            "radar_row": int(grid["rows"][nearest]), "radar_col": int(grid["cols"][nearest]),
            "radar_flat_index": int(grid["flat_indices"][nearest]),
            "radar_lon": float(radar_lon[nearest]), "radar_lat": float(radar_lat[nearest]),
        })
    mapping = pd.DataFrame(records)
    mapping.to_csv(LINK_GRID_MAP_PATH, index=False, encoding="utf-8-sig")
    return mapping


def speed_row_to_tm(date_value, time_code_value) -> str:
    date = datetime.strptime(str(int(date_value)), "%Y%m%d")
    time_code = int(time_code_value)
    if time_code == 24:
        return (date + timedelta(days=1)).strftime("%Y%m%d") + "0000"
    return date.strftime("%Y%m%d") + f"{time_code:02d}00"


def raw_to_precip_mm(raw_values: np.ndarray) -> np.ndarray:
    precip = raw_values.astype(np.float32) / 100.0
    precip[raw_values < 0] = 0.0
    precip[raw_values <= -30000] = np.nan
    return precip


def build_precip(fact: pd.DataFrame) -> pd.DataFrame:
    target_link_ids = set(fact["LINK_ID"].unique())
    grid = load_or_build_seoul_grid()
    mapping = build_link_grid_mapping(grid, target_link_ids)
    flat_indices = mapping["radar_flat_index"].to_numpy(dtype=np.int64)
    link_ids = mapping["LINK_ID"].to_numpy()

    tms = sorted({speed_row_to_tm(d, t) for d, t in zip(fact["기준일"], fact["시간코드"])})
    records = []
    missing = []
    for tm in tms:
        radar_path = RADAR_DIR / f"hsr_pcph_acc60_{tm}.txt"
        if not radar_path.exists():
            missing.append(tm)
            continue
        _, _, radar_grid = parse_kma_ascii_grid(radar_path)
        flat_values = radar_grid.reshape(-1)[flat_indices]
        precip_values = raw_to_precip_mm(flat_values)
        for link_id, precip_mm in zip(link_ids, precip_values):
            records.append({"tm": tm, "LINK_ID": link_id, "precip_mm": float(precip_mm) if not np.isnan(precip_mm) else np.nan})

    print("레이더 처리 시각 수:", len(tms) - len(missing), "| 누락:", len(missing))
    if missing:
        print("누락 예시:", ", ".join(missing[:5]))
    return pd.DataFrame(records)


# ---------- 시간대별 교통량 ----------

def build_hourly_traffic_volume(seoul_ids: set) -> pd.DataFrame:
    tv_col_names = [
        "LEVEL55_LINK_ID", "ITS_LINK_ID", "road_rank_tv", "road_name_tv", "region_tv",
        "length_km_tv", "lanes_tv", "aadt",
    ] + [f"vol_h{h}" for h in range(1, 25)]
    tv = pd.read_excel(TRAFFIC_VOLUME_XLSX, sheet_name="Sheet0", header=6, skiprows=[7])
    tv.columns = tv_col_names
    tv["its_id_list"] = tv["ITS_LINK_ID"].astype(str).str.split(",")
    exp = tv.explode("its_id_list")
    exp["LINK_ID"] = exp["its_id_list"].str.strip()
    exp = exp[(exp["LINK_ID"] != "0") & exp["LINK_ID"].isin(seoul_ids)]

    hourly_cols = [f"vol_h{h}" for h in range(1, 25)]
    long = exp.melt(id_vars=["LINK_ID", "aadt"], value_vars=hourly_cols, var_name="hour_col", value_name="traffic_volume_hourly")
    long["시간코드"] = long["hour_col"].str.replace("vol_h", "", regex=False).astype("float32")
    long = long.groupby(["LINK_ID", "시간코드"], as_index=False).agg(
        traffic_volume_hourly=("traffic_volume_hourly", "mean"), aadt=("aadt", "mean"),
    )
    long["traffic_volume_ratio"] = (long["traffic_volume_hourly"] / long["aadt"]).replace([np.inf, -np.inf], np.nan)
    return long.drop(columns=["aadt"])


# ---------- 생활인구(격자·행정동) ----------

def build_population_features(fact: pd.DataFrame) -> pd.DataFrame:
    """(LINK_ID, 기준일, 시간코드)별 250m격자/행정동 생활인구. 시간코드-1=실제시(0~23시)."""
    link_pt = gpd.read_file(SEOUL_LINK_SHP)[["LINK_ID", "geometry"]]
    link_pt["LINK_ID"] = link_pt["LINK_ID"].astype(str)
    link_pt["geometry"] = link_pt.geometry.centroid
    link_pt = link_pt.to_crs("EPSG:5179")

    grid_cells = gpd.read_file(GRID_MATCH_SHP)[["CELL_ID", "geometry"]]
    link_cell = gpd.sjoin(link_pt, grid_cells, how="left", predicate="within")[["LINK_ID", "CELL_ID"]]
    print(f"  링크->격자 매칭: {link_cell['CELL_ID'].notna().sum():,} / {len(link_cell):,}")

    pop_dates = sorted(fact["기준일"].unique())

    grid_parts = []
    for d in pop_dates:
        part = pd.read_csv(
            GRID_POP_DIR / f"250_LOCAL_RESD_{d}.csv", encoding="cp949",
            usecols=["일자", "시간", "행정동코드", "250M격자", "생활인구합계"],
        )
        part["생활인구합계"] = pd.to_numeric(part["생활인구합계"], errors="coerce").fillna(0)
        grid_parts.append(part)
    grid_raw = pd.concat(grid_parts, ignore_index=True)

    dong_parts = []
    for d in pop_dates:
        part = pd.read_csv(
            DONG_POP_DIR / f"250_LOCAL_RESD_ADMDONG_{d}.csv", encoding="cp949",
            usecols=["일자", "시간", "행정동코드", "생활인구합계"],
        )
        part["생활인구합계"] = pd.to_numeric(part["생활인구합계"], errors="coerce").fillna(0)
        dong_parts.append(part)
    dong_pop_lookup = pd.concat(dong_parts, ignore_index=True)
    dong_pop_lookup.columns = ["기준일", "hour", "행정동코드", "dong_pop"]
    dong_pop_lookup["기준일"] = dong_pop_lookup["기준일"].astype(str)

    # 셀 -> 대표 행정동(인구 비중 최대) — 경계에 걸친 셀은 여러 행정동에 인구가 나뉘어 기록됨
    cell_dong_weight = grid_raw.groupby(["250M격자", "행정동코드"])["생활인구합계"].sum().reset_index()
    cell_to_dong = cell_dong_weight.loc[cell_dong_weight.groupby("250M격자")["생활인구합계"].idxmax()].set_index("250M격자")["행정동코드"]

    # 셀 단위 생활인구(격자,일자,시간) — 경계 분할 행은 합산해서 셀 전체 인구로 복원
    grid_pop_lookup = grid_raw.groupby(["250M격자", "일자", "시간"])["생활인구합계"].sum().reset_index()
    grid_pop_lookup.columns = ["CELL_ID", "기준일", "hour", "grid_pop"]
    grid_pop_lookup["기준일"] = grid_pop_lookup["기준일"].astype(str)

    link_cell["행정동코드"] = link_cell["CELL_ID"].map(cell_to_dong).astype("float64")
    dong_pop_lookup["행정동코드"] = dong_pop_lookup["행정동코드"].astype("float64")

    result = fact.copy()
    result["hour"] = result["시간코드"].astype(int) - 1
    result = result.merge(link_cell, on="LINK_ID", how="left", validate="many_to_one")
    result = result.merge(grid_pop_lookup, on=["CELL_ID", "기준일", "hour"], how="left", validate="many_to_one")
    result = result.merge(dong_pop_lookup, on=["행정동코드", "기준일", "hour"], how="left", validate="many_to_one")
    result["grid_pop"] = result["grid_pop"].astype("float32")
    result["dong_pop"] = result["dong_pop"].astype("float32")
    return result.drop(columns=["CELL_ID", "행정동코드", "hour"])


def main() -> None:
    fact = pd.read_csv(
        FIVEMIN_PATH,
        dtype={"LINK_ID": str, "기준일": str, "시간코드": str},
        encoding="utf-8-sig",
    )
    fact["평균속도"] = pd.to_numeric(fact["평균속도"], errors="coerce")
    fact["시간코드"] = pd.to_numeric(fact["시간코드"], errors="coerce").astype("float32")
    before = len(fact)
    print("기본 행 수 (02번 산출물):", before)

    print("[1/3] 레이더 강수 매칭 중...")
    precip = build_precip(fact)
    fact["tm"] = [speed_row_to_tm(d, t) for d, t in zip(fact["기준일"], fact["시간코드"])]
    fact = fact.merge(precip, on=["tm", "LINK_ID"], how="left")
    fact = fact.drop(columns=["tm"])
    assert len(fact) == before, "레이더 병합 후 행 수가 달라짐(1:1 매칭 깨짐)"

    print("[2/3] 시간대별 교통량 매칭 중...")
    seoul_ids = set(fact["LINK_ID"].unique())
    hourly_volume = build_hourly_traffic_volume(seoul_ids)
    fact = fact.merge(hourly_volume, on=["LINK_ID", "시간코드"], how="left")
    assert len(fact) == before, "교통량 병합 후 행 수가 달라짐(1:1 매칭 깨짐)"

    print("[3/3] 생활인구(격자/행정동) 매칭 중...")
    fact = build_population_features(fact)
    assert len(fact) == before, "생활인구 병합 후 행 수가 달라짐(1:1 매칭 깨짐)"

    fact.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    print()
    print("속도 팩트 테이블 저장:", OUT_CSV)
    print("행 수:", len(fact), "| 컬럼:", list(fact.columns))
    print("강수값 존재 행 수:", int(fact["precip_mm"].notna().sum()))
    print(f"traffic_volume_hourly 매칭 비율: {fact['traffic_volume_hourly'].notna().mean() * 100:.1f}%")
    print(f"grid_pop 매칭 비율: {fact['grid_pop'].notna().mean() * 100:.1f}%")
    print(f"dong_pop 매칭 비율: {fact['dong_pop'].notna().mean() * 100:.1f}%")


if __name__ == "__main__":
    main()
