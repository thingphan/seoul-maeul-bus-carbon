"""한 시점의 레이더 강수를 서울 전 링크(63,548개)에 붙인 스냅샷을 만든다.

링크 중심점마다 가장 가까운 HSR 레이더 격자를 미리 매핑해 두고(all_link_radar_grid_mapping.csv),
시점별 레이더 파일에서 그 격자값만 꺼내 강수(mm)로 바꾼다. 03은 이 파일을 강수 입력으로 쓰고,
04는 이 모듈의 함수를 직접 불러 기간 전체에 재사용한다.

입력: data/raw/weather/radar/hsr_pcph_acc60_{tm}.txt, data/interim/seoul_link.shp·seoul_link_attr.csv·seoul_link_dem_features.csv
출력: data/processed/weather_snapshot/weather_snapshot_{tm}.csv
사용: python src/models/weather_snapshot.py --tm 202607160800
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import geopandas as gpd
import numpy as np
import polars as pl
from scipy.spatial import cKDTree

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
INTERIM_DATA_DIR = PROJECT_ROOT / "data" / "interim"

SEOUL_LINK_SHP = INTERIM_DATA_DIR / "seoul_link.shp"
LINK_ATTR_PATH = INTERIM_DATA_DIR / "seoul_link_attr.csv"
DEM_PATH = INTERIM_DATA_DIR / "seoul_link_dem_features.csv"
GRID_CACHE_PATH = INTERIM_DATA_DIR / "weather_hsr_seoul_grid.npz"
ALL_LINK_GRID_MAP_PATH = INTERIM_DATA_DIR / "all_link_radar_grid_mapping.csv"
RADAR_DIR = RAW_DATA_DIR / "weather" / "radar"
SPEED_FACT_SCRIPT = PROJECT_ROOT / "src" / "pipeline" / "05_build_speed_fact.py"
SNAPSHOT_DIR = PROCESSED_DATA_DIR / "weather_snapshot"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="한 시점의 링크별 레이더 강수 스냅샷")
    parser.add_argument("--tm", required=True, help="레이더 시각 YYYYMMDDHHMM")
    return parser.parse_args()


def parse_kma_ascii_grid_fast(path: Path) -> np.ndarray:
    """기상청 ASCII 격자 파일을 (높이, 너비) 배열로 읽는다. 첫 두 수가 너비·높이, 이후가 격자값이다."""
    text = path.read_text(encoding="utf-8").replace(",", " ").replace("=", " ")
    lines = [line for line in text.splitlines() if line.strip() and not line.strip().startswith("#")]
    numbers = np.fromstring(" ".join(lines), dtype=np.float32, sep=" ")
    if numbers.size < 3:
        raise SystemExit(f"격자 파일을 읽을 수 없습니다: {path}")
    width, height = int(numbers[0]), int(numbers[1])
    values = numbers[2:]
    if values.size < width * height:
        raise SystemExit(f"격자 크기 불일치({path.name}): 예상 {width * height}, 실제 {values.size}")
    return values[:width * height].reshape(height, width)


def load_or_build_grid_mapping() -> pl.DataFrame:
    """링크 중심점별 최근접 레이더 격자 매핑을 읽는다. 없거나 링크 집합이 바뀌었으면 새로 만든다."""
    links_df = pl.read_csv(LINK_ATTR_PATH, columns=["LINK_ID"], schema_overrides={"LINK_ID": pl.Utf8})
    all_link_ids = set(links_df["LINK_ID"].to_list())

    if ALL_LINK_GRID_MAP_PATH.exists():
        cached = pl.read_csv(ALL_LINK_GRID_MAP_PATH, schema_overrides={"LINK_ID": pl.Utf8})
        if set(cached["LINK_ID"].to_list()) == all_link_ids:
            return cached

    print("링크-레이더 격자 매핑 생성")
    if not GRID_CACHE_PATH.exists():
        # 서울 범위 격자 캐시는 pipeline/05의 함수로 원본 격자(hsr_lon·lat.txt)에서 만든다
        spec = importlib.util.spec_from_file_location("speed_fact", SPEED_FACT_SCRIPT)
        speed_fact = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(speed_fact)
        speed_fact.load_or_build_seoul_grid()

    grid = np.load(GRID_CACHE_PATH)
    radar_points = np.column_stack([grid["lat"], grid["lon"]])
    tree = cKDTree(radar_points)

    links = gpd.read_file(SEOUL_LINK_SHP)
    links["LINK_ID"] = links["LINK_ID"].astype(str)
    centroids = links.geometry.centroid
    wgs84 = gpd.GeoSeries(centroids, crs=links.crs).to_crs(epsg=4326)
    coordinates = np.column_stack([wgs84.y.to_numpy(), wgs84.x.to_numpy()])
    _, nearest = tree.query(coordinates, k=1)

    mapping = pl.DataFrame({
        "LINK_ID": links["LINK_ID"].to_numpy(),
        "link_lon": wgs84.x.to_numpy(),
        "link_lat": wgs84.y.to_numpy(),
        "radar_row": grid["rows"][nearest],
        "radar_col": grid["cols"][nearest],
        "radar_flat_index": grid["flat_indices"][nearest],
        "radar_lon": grid["lon"][nearest],
        "radar_lat": grid["lat"][nearest],
    })
    mapping.write_csv(ALL_LINK_GRID_MAP_PATH)
    print("저장:", ALL_LINK_GRID_MAP_PATH)
    return mapping


def main() -> None:
    args = parse_args()
    radar_path = RADAR_DIR / f"hsr_pcph_acc60_{args.tm}.txt"
    if not radar_path.exists():
        raise SystemExit(f"레이더 파일이 없습니다: {radar_path}")

    mapping = load_or_build_grid_mapping()
    radar = parse_kma_ascii_grid_fast(radar_path).reshape(-1)
    raw = radar[mapping["radar_flat_index"].to_numpy()].astype(np.float32)
    # 원시값은 0.01mm 단위. -25000(관측영역 내 무강수) 등 음수는 0, -30000 이하(관측영역 밖)는 결측
    precip = raw / 100.0
    precip = np.where(raw < 0, 0.0, precip)
    precip = np.where(raw <= -30000, np.nan, precip)

    weather = mapping.select(["LINK_ID"]).with_columns([
        pl.lit(args.tm).alias("tm"),
        pl.Series("radar_raw_value", raw, dtype=pl.Float32),
        pl.Series("precip_mm", precip, dtype=pl.Float32)
    ])

    links = pl.read_csv(LINK_ATTR_PATH, schema_overrides={"LINK_ID": pl.Utf8})
    dem = pl.read_csv(DEM_PATH, schema_overrides={"LINK_ID": pl.Utf8})
    snapshot = links.join(dem, on="LINK_ID", how="left").join(weather, on="LINK_ID", how="left")

    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SNAPSHOT_DIR / f"weather_snapshot_{args.tm}.csv"
    snapshot.write_csv(out_path)
    print("저장:", out_path)
    print("링크 수:", snapshot.height, "| 강수 결측:", snapshot.select(pl.col("precip_mm").is_null().sum()).item())


if __name__ == "__main__":
    main()