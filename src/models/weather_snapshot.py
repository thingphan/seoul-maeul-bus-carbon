"""
[역할]
특정 기상 시각(tm)을 기준으로 서울시 전체 63,548개 도로 링크에 대한 정적(도로 속성, 지형 DEM) 정보 및 동적 날씨(강수량) 스냅샷 데이터를 고속으로 생성하는 모듈입니다.

[주요 기능]
- KMA 레이더 초고속 파싱: 기상청 HSR 레이더 텍스트 원본에서 주석 라인을 안전하게 필터링한 후 C-레벨에서 수치 데이터 배열로 고속 변환합니다.
- NumPy 기반 다이렉트 인덱싱: 별도의 데이터프레임 조인(Join) 과정 없이, 캐싱된 격자 매핑 배열을 활용해 6.3만 개 링크에 레이더 강수 데이터를 백터 단위로 즉각 할당합니다.
- Polars 멀티스레딩 데이터 처리: Pandas를 Rust 기반의 Polars 엔진으로 교체하여, 대용량 정적 데이터와 기상 정보의 병합 및 I/O 속도를 극대화했습니다.
- 멱등성(Idempotency) 내장: 동일한 시각(tm)에 대해 스크립트를 여러 번 중복 실행하더라도, 시스템 오류 없이 해당 시각의 파일만 안전하게 덮어쓰기(Overwrite)를 수행하여 정합성을 유지합니다.

[실행 방식]
CLI 환경에서 KMA 시각(YYYYMMDDHHMM)을 파라미터로 전달하여 단독 실행하거나, `src/models/04_batch_predict_all_link_speed.py`가 배치 예측 시 자동 호출합니다.
- 실행 예시: python src/models/weather_snapshot.py --tm 202607090800

[결과물 저장]
- 저장 위치: data/processed/weather_snapshot/weather_snapshot_{tm}.csv
- 산출물 내용: 특정 시각의 서울 전체 링크 정적 속성, DEM 고도/경사 데이터, 시간대별 레이더 강수량이 최종 결합된 스냅샷 파일.
"""

from __future__ import annotations

import argparse
import importlib.util
import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import polars as pl
from scipy.spatial import cKDTree

# 프로젝트 루트(src/models/의 두 단계 위)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tm", required=True, help="KMA 시각 YYYYMMDDHHMM")
    return parser.parse_args()


def parse_kma_ascii_grid_fast(path: Path) -> np.ndarray:
    """KMA 레이더 ASCII 그리드 파일 파싱 (오염된 데이터 정제 후 고속 파싱)"""
    try:
        # 1. 파일을 텍스트로 읽어옵니다.
        raw_text = path.read_text(encoding="utf-8")

        # 2. 파싱을 방해하는 쉼표(,)와 등호(=)를 공백으로 치환하여 데이터를 정제합니다.
        clean_text = raw_text.replace(",", " ").replace("=", " ")

        # 3. 주석('#')을 제외한 모든 유효 라인을 하나의 문자열로 합칩니다.
        lines = [line for line in clean_text.splitlines() if line.strip() and not line.strip().startswith('#')]
        final_text = " ".join(lines)
        
        # 4. 정제된 문자열을 np.fromstring을 사용하여 C-level에서 고속으로 1D 배열로 파싱합니다.
        all_numbers = np.fromstring(final_text, dtype=np.float32, sep=' ')

        if all_numbers.size < 3:
            raise ValueError(f"수치 데이터 부족. 격자 정보를 만들 수 없습니다. (데이터 개수: {all_numbers.size})")

        # 첫 두 숫자를 너비와 높이로 해석
        width, height = int(all_numbers[0]), int(all_numbers[1])
        expected_size = width * height
        
        # 실제 데이터는 세 번째 숫자부터 시작
        data_values = all_numbers[2:]

        if data_values.size < expected_size:
            raise ValueError(f"격자 크기 불일치. 예상: {expected_size}, 실제: {data_values.size}")

        # 예상 크기보다 데이터가 많을 경우, 예상 크기에 맞게 잘라냅니다.
        grid_data = data_values[:expected_size]

        return grid_data.reshape(height, width)

    except Exception as e:
        print(f"'{path.name}' 파일 처리 중 심각한 오류 발생: {e}")
        print("파일의 첫 5줄을 확인하세요:")
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i >= 5: break
                    print(f"  {i+1}: {line.strip()}")
        except Exception as read_e:
            print(f"  파일을 읽는 중 추가 오류 발생: {read_e}")
        raise SystemExit("스크립트를 중단합니다.")


def load_or_build_grid_mapping() -> pl.DataFrame:
    """Polars를 활용해 고속으로 매핑 테이블을 로드하거나 캐싱을 생성합니다."""
    links_df = pl.read_csv(LINK_ATTR_PATH, columns=["LINK_ID"], schema_overrides={"LINK_ID": pl.Utf8})
    all_link_ids = set(links_df["LINK_ID"].to_list())

    if ALL_LINK_GRID_MAP_PATH.exists():
        cached = pl.read_csv(ALL_LINK_GRID_MAP_PATH, schema_overrides={"LINK_ID": pl.Utf8})
        if set(cached["LINK_ID"].to_list()) == all_link_ids:
            return cached

    print("격자 매핑 테이블을 생성합니다. (최초 1회)")
    if not GRID_CACHE_PATH.exists():
        # 격자 캐시는 pipeline/05가 원본 격자(data/raw/weather/grid/hsr_lon·lat.txt)에서 만든다. 같은 함수를 재사용한다.
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
    print("전체 링크-레이더 격자 매핑 저장 완료:", ALL_LINK_GRID_MAP_PATH)
    return mapping


def main() -> None:
    args = parse_args()
    radar_path = RADAR_DIR / f"hsr_pcph_acc60_{args.tm}.txt"
    if not radar_path.exists():
        raise SystemExit(f"레이더 파일이 없습니다: {radar_path}")

    mapping = load_or_build_grid_mapping()
    radar = parse_kma_ascii_grid_fast(radar_path).reshape(-1)
    
    flat_indices = mapping["radar_flat_index"].to_numpy()
    raw = radar[flat_indices].astype(np.float32)
    
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
    
    snapshot = (
        links
        .join(dem, on="LINK_ID", how="left")
        .join(weather, on="LINK_ID", how="left")
    )

    WEATHER_SNAPSHOT_DIR = PROCESSED_DATA_DIR / "weather_snapshot"
    WEATHER_SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

    out_path = WEATHER_SNAPSHOT_DIR / f"weather_snapshot_{args.tm}.csv"
    snapshot.write_csv(out_path)
    
    print("전체 링크 기상 스냅샷 고속 저장 완료:", out_path)
    print("링크 수:", snapshot.height, "| 강수 결측:", snapshot.select(pl.col("precip_mm").is_null().sum()).item())


if __name__ == "__main__":
    main()