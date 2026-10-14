"""전국 5분 단위 통행속도에서 서울 링크만 걸러 링크×날짜×시간 단위로 집계한다.

하루 약 7GB인 원본을 Polars LazyFrame으로 읽어 메모리에 전부 올리지 않고 필터·집계한다.
이미 집계된 날짜는 건너뛰고, 새 날짜만 기존 결과에 덧붙인다(같은 날짜를 다시 넣으면 그 날짜만 교체).

입력: data/raw/traffic/202607_5Min/YYYYMMDD_5Min.csv, data/interim/seoul_link_attr.csv(01)
출력: data/interim/seoul_link_speed_5min_hourly.csv
      LINK_ID, 기준일, 시간코드(00시=01 … 23시=24), 평균속도, sample_count_5min(시간당 5분 표본 수, 최대 12)
사용: python src/pipeline/02_aggregate_5min_speed.py
"""

import re
import sys
import time
from pathlib import Path

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INTERIM_DATA_DIR = PROJECT_ROOT / "data" / "interim"
SEOUL_LINK_ATTR_PATH = INTERIM_DATA_DIR / "seoul_link_attr.csv"
FIVEMIN_DIR = PROJECT_ROOT / "data" / "raw" / "traffic" / "202607_5Min"
FIVEMIN_GLOB = "*_5Min.csv"
OUT_CSV = INTERIM_DATA_DIR / "seoul_link_speed_5min_hourly.csv"
OUT_SCHEMA = {"LINK_ID": pl.String, "기준일": pl.String, "시간코드": pl.String,
              "평균속도": pl.Float32, "sample_count_5min": pl.Int32}


def read_seoul_link_ids() -> list[str]:
    if not SEOUL_LINK_ATTR_PATH.exists():
        sys.exit(f"서울 링크 속성 파일이 없습니다. 먼저 01을 실행하세요: {SEOUL_LINK_ATTR_PATH}")
    df = pl.read_csv(SEOUL_LINK_ATTR_PATH, columns=["LINK_ID"], schema_overrides={"LINK_ID": pl.String})
    seoul_ids = df["LINK_ID"].drop_nulls().to_list()
    print(f"서울 링크 {len(seoul_ids):,}개")
    return seoul_ids


def aggregate_one_file(path: Path, seoul_ids: list[str]) -> pl.DataFrame:
    """하루치 5분 속도 파일을 서울 링크×날짜×시간 평균속도와 표본 수로 집계한다."""
    start_time = time.time()
    lf = pl.scan_csv(
        path,
        schema_overrides={"LINKID": pl.String, "CREATDE": pl.String, "CREATHM": pl.Int32, "PASNGSPED": pl.Float32},
        null_values=["", "null", "NULL", "NaN"],
        ignore_errors=True,  # 형식이 깨진 일부 행은 건너뛴다
    )
    hourly = (
        lf.filter(pl.col("LINKID").is_in(seoul_ids))
        .with_columns([
            pl.col("LINKID").alias("LINK_ID"),
            pl.col("CREATDE").alias("기준일"),
            # HHMM의 시(00~23, 2400은 23으로)에 1을 더해 시간코드 01~24로 만든다
            pl.when(pl.col("CREATHM") // 100 >= 24).then(23).otherwise(pl.col("CREATHM") // 100)
              .add(1).cast(pl.String).str.zfill(2).alias("시간코드"),
            pl.col("PASNGSPED").alias("평균속도"),
        ])
        .group_by(["LINK_ID", "기준일", "시간코드"])
        .agg([pl.col("평균속도").mean().round(2), pl.len().cast(pl.Int32).alias("sample_count_5min")])
        .collect()
    )
    print(f"  {path.name}: {len(hourly):,}행, {time.time() - start_time:.1f}초")
    return hourly


def merge_and_save(new_dfs: list[pl.DataFrame]) -> pl.DataFrame:
    """새로 집계한 날짜를 기존 결과에 합쳐 저장한다. 같은 날짜가 이미 있으면 새 값으로 교체한다."""
    if not new_dfs:
        print("새로 집계할 날짜가 없습니다.")
        return pl.read_csv(OUT_CSV, schema_overrides=OUT_SCHEMA) if OUT_CSV.exists() else pl.DataFrame()

    new_data = pl.concat(new_dfs)
    new_dates = new_data["기준일"].unique().to_list()
    print(f"새로 집계한 날짜: {sorted(new_dates)}")
    final_data = new_data
    if OUT_CSV.exists():
        try:
            existing = pl.read_csv(OUT_CSV, schema_overrides=OUT_SCHEMA)
            # clone(): 기존 파일을 덮어쓰기 전에 읽은 데이터와의 연결을 끊는다
            final_data = pl.concat([existing.filter(~pl.col("기준일").is_in(new_dates)).clone(), new_data])
        except Exception as e:  # noqa: BLE001 — 기존 파일이 깨졌으면 새 집계로 다시 쓴다
            print(f"기존 결과를 읽지 못해 새 집계로 덮어씁니다: {e}")

    final_data = final_data.sort(["기준일", "시간코드", "LINK_ID"])
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    final_data.write_csv(OUT_CSV)
    print(f"저장: {OUT_CSV} ({len(final_data):,}행)")
    return final_data


def main() -> None:
    start_time = time.time()
    seoul_ids = read_seoul_link_ids()

    existing_dates: set[str] = set()
    if OUT_CSV.exists():
        try:
            dates = pl.read_csv(OUT_CSV, columns=["기준일"], schema_overrides={"기준일": pl.String})
            existing_dates = set(dates["기준일"].unique().to_list())
            print(f"이미 집계된 날짜 {len(existing_dates)}일")
        except Exception as e:  # noqa: BLE001
            print(f"기존 결과의 날짜 목록을 읽지 못했습니다: {e}")

    paths = sorted(FIVEMIN_DIR.glob(FIVEMIN_GLOB)) if FIVEMIN_DIR.exists() else []
    if not paths:
        sys.exit(f"5분 속도 원본이 없습니다: {FIVEMIN_DIR / FIVEMIN_GLOB}")
    print(f"5분 속도 원본 {len(paths)}개")

    processed, skipped = [], 0
    for path in paths:
        match = re.search(r"(\d{8})_5Min", path.name)
        if match and match.group(1) in existing_dates:
            skipped += 1
            continue
        processed.append(aggregate_one_file(path, seoul_ids))
    combined = merge_and_save(processed)

    print(f"\n처리 {len(paths) - skipped}일, 건너뜀 {skipped}일, 총 {time.time() - start_time:.1f}초")
    if not combined.is_empty():
        n_links = combined["LINK_ID"].n_unique()
        print(f"누적 {combined['기준일'].n_unique()}일, {len(combined):,}행, 관측 링크 {n_links:,}개"
              f"({n_links / len(seoul_ids):.2%}), 시간당 5분 표본 평균 {combined['sample_count_5min'].mean():.2f}(최대 12)")


if __name__ == "__main__":
    main()
