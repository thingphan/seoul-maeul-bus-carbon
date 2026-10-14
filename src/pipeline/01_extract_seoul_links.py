"""서울 행정경계와 교차하는 표준노드링크를 추출한다.

경계에 걸친 링크는 잘라내지 않고 형상 전체를 남긴다. 갱신일·비고 등 분석에 쓰지 않는 메타데이터 열은 뺀다.

입력: data/raw/road/MOCT_LINK.shp, data/raw/boundary/seoul_admin_boundary.geojson
출력:
  data/interim/seoul_link.shp        링크 형상(63,548개)
  data/interim/seoul_link_attr.csv   링크 속성(04가 링크 차원 표로 확장)
사용: python src/pipeline/01_extract_seoul_links.py
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
ROAD_RAW_DIR = RAW_DATA_DIR / "road"
BOUNDARY_RAW_DIR = RAW_DATA_DIR / "boundary"
INTERIM_DATA_DIR = PROJECT_ROOT / "data" / "interim"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

LINK_PATH = ROAD_RAW_DIR / "MOCT_LINK.shp"
BOUNDARY_PATH = BOUNDARY_RAW_DIR / "seoul_admin_boundary.geojson"
OUT_SHP = INTERIM_DATA_DIR / "seoul_link.shp"
OUT_CSV = INTERIM_DATA_DIR / "seoul_link_attr.csv"

DROP_META_COLS = ["UPDATEDATE", "REMARK", "HIST_TYPE", "HISTREMARK"]


def load_seoul_boundary(target_crs) -> "gpd.GeoSeries":
    boundary = gpd.read_file(BOUNDARY_PATH)
    if boundary.crs is None:
        boundary = boundary.set_crs("EPSG:4326")
    seoul_polygon = boundary.union_all()
    return gpd.GeoSeries([seoul_polygon], crs=boundary.crs).to_crs(target_crs)


def main() -> None:
    link = gpd.read_file(LINK_PATH)
    print("전체 링크 수:", link.shape)
    print("좌표계:", link.crs)

    seoul_boundary = load_seoul_boundary(link.crs)
    seoul_link = link[link.intersects(seoul_boundary.iloc[0])].copy()
    print("서울 추정 링크 수:", seoul_link.shape)

    INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    seoul_link.to_file(OUT_SHP, encoding="utf-8")

    attr = seoul_link.drop(columns="geometry")
    drop_cols = [c for c in DROP_META_COLS if c in attr.columns]
    attr = attr.drop(columns=drop_cols)
    attr.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    print("서울 링크 Shapefile 저장:", OUT_SHP)
    print("서울 링크 속성 CSV 저장:", OUT_CSV, "| 드롭한 메타 컬럼:", drop_cols)
    print("속성 컬럼 수:", len(attr.columns))


if __name__ == "__main__":
    main()
