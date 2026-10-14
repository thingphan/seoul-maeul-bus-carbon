"""
[역할]
5분 단위 전국 교통소통 원본에서 서울 링크만 걸러 1시간 단위로 집계한다.

[변경사항]
1. 구 src/04_aggregate_5min_speed.py와 로직 동일(이미 올바른 grain — LINK_ID x
    기준일 x 시간코드 — 이라 변경 불필요), src/pipeline/으로 위치만 이동.

2. 기존 Pandas Chunking(단일 코어 순차 연산)의 극심한 I/O 병목 및 CPU 대기를 해결하기 위해, 
    Rust 기반의 멀티쓰레딩 쿼리 엔진인 'Polars'와 'LazyFrame(지연 연산)' 아키텍처를 도입
    - Pandas 청크 로직 전용 매개변수를 제거했습니다

3. 동일한 날짜에 대해 중복 실행되더라도 기존의 과거 날짜 데이터는 보존하고, 
    새로 입력된 날짜 데이터셋만 스마트하게 Overwrite(점진적 누적 업데이트)하는 멱등성이 내장

[출력]
data/interim/seoul_link_speed_5min_hourly.csv (다음 파이프라인 단계의 중간 재료)

[사용]
python src/pipeline/02_aggregate_5min_speed.py
"""

import os
import re
import sys
import time
from pathlib import Path
import polars as pl


# =====================================================================
# 1. 프로젝트 및 전역 데이터 디렉토리 설정
# =====================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
TRAFFIC_RAW_DIR = RAW_DATA_DIR / "traffic"
INTERIM_DATA_DIR = PROJECT_ROOT / "data" / "interim"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

SEOUL_LINK_ATTR_PATH = INTERIM_DATA_DIR / "seoul_link_attr.csv"
FIVEMIN_DIR = TRAFFIC_RAW_DIR / "202607_5Min"
FIVEMIN_GLOB = "*_5Min.csv"
OUT_CSV = INTERIM_DATA_DIR / "seoul_link_speed_5min_hourly.csv"

# =====================================================================
# 2. 핵심 함수 정의
# =====================================================================
def read_seoul_link_ids() -> list[str]:
    """
    서울 도로 속성 테이블로부터 서울시 표준 도로 링크 ID 리스트를 읽어옵니다.
    """
    # 경로 교차 지원 예외 처리
    target_path = SEOUL_LINK_ATTR_PATH
    if not target_path.exists():
        target_path = PROCESSED_DATA_DIR / "seoul_link_attr.csv"

    if not target_path.exists():
        print(f"서울 도로 속성 파일을 찾을 수 없습니다: {SEOUL_LINK_ATTR_PATH}")
        sys.exit(1)
        
    try:
        df = pl.read_csv(
            target_path,
            columns=["LINK_ID"],
            schema_overrides={"LINK_ID": pl.String}
        )
        seoul_ids = df["LINK_ID"].drop_nulls().to_list()
        print(f"서울시 표준 링크 {len(seoul_ids):,}개 식별자 로딩 완료.")
        return seoul_ids
    except Exception as e:
        print(f"서울 도로 속성 파일 파싱 중 에러 발생: {e}")
        sys.exit(1)


def aggregate_one_file(path: Path, seoul_ids: list[str]) -> pl.DataFrame:
    """
    Polars 지연 연산(LazyFrame)을 활용하여 하루 약 7GB 분량의 전국 5분 속도 원천 데이터를
    서울 영역 필터링 -> 시간코드(01~24) 매핑 -> 1시간 집계 데이터프레임으로 변환합니다.
    """
    start_time = time.time()
    
    # 5분 데이터 스키마 타입 강제 설정 (타입 파싱 오류 및 메모리 낭비 차단)
    schema_overrides = {
        "LINKID": pl.String,
        "CREATDE": pl.String,
        "CREATHM": pl.Int32,
        "PASNGSPED": pl.Float32
    }
    
    # scan_csv를 사용하여 디스크 전체 로딩 없이 지연 쿼리 플랜 구성
    lf = pl.scan_csv(
        path,
        schema_overrides=schema_overrides,
        null_values=["", "null", "NULL", "NaN"],
        ignore_errors=True  # 일부 비정상적인 행이 있어도 프로세스가 죽지 않도록 허용
    )
    
    # 지연 연산 체인 빌드 (Rust 쿼리 최적화기가 최적의 CPU 코어 분산 및 프로젝션 푸시다운 수행)
    lf_aggregated = (
        lf
        # 1. 서울시에 있는 도로 링크만 즉시 공간 필터링 (메모리 사용량 급감)
        .filter(pl.col("LINKID").is_in(seoul_ids))
        # 2. 필요한 데이터 형태 표준화 및 시간코드 연산
        .with_columns([
            pl.col("LINKID").alias("LINK_ID"),
            pl.col("CREATDE").alias("기준일"),
            # HHMM 시각에서 시간을 분할(CREATHM // 100). 자정(2400) 예외 시 23으로 클리핑 후 +1 하여 01~24 시간코드 도출
            pl.when(pl.col("CREATHM") // 100 >= 24)
              .then(23)
              .otherwise(pl.col("CREATHM") // 100)
              .add(1)
              .cast(pl.String)
              .str.zfill(2)
              .alias("시간코드"),
            pl.col("PASNGSPED").alias("평균속도")
        ])
        # 3. 1시간 키 기준(LINK_ID, 기준일, 시간코드) 그룹바이 집계
        .group_by(["LINK_ID", "기준일", "시간코드"])
        .agg([
            # 속도 평균 및 소수점 둘째 자리 반올림
            pl.col("평균속도").mean().round(2),
            # 시간당 5분 샘플 개수 집계 (최대 12)
            pl.len().cast(pl.Int32).alias("sample_count_5min")
        ])
    )
    
    # collect()를 호출하는 순간 전체 멀티쓰레딩 병렬 집계 쿼리가 동작합니다.
    df_result = lf_aggregated.collect()
    
    elapsed = time.time() - start_time
    print(f"집계 완료: {path.name} | 집계 행 수: {len(df_result):,}건 | 연산 시간: {elapsed:.2f}초")
    return df_result

def merge_and_save(new_dfs: list[pl.DataFrame]) -> None:
    """
    새로 집계된 일자별 데이터들과 기존의 seoul_link_speed_5min_hourly.csv를 스마트 병합합니다.
    보유한 과거 이력 날짜 정보는 안전하게 보존하고, 새로 추가된 날짜 데이터에 대해서만 덮어쓰기를 수행합니다.
    """
    if not new_dfs:
        print("새롭게 가공된 원천 속도 데이터가 없습니다. 프로세스를 종료합니다.")
        if OUT_CSV.exists():
            return pl.read_csv(OUT_CSV, schema_overrides={
                "LINK_ID": pl.String, "기준일": pl.String, "시간코드": pl.String,
                "평균속도": pl.Float32, "sample_count_5min": pl.Int32
            })
        return pl.DataFrame()
    
    # 1. 신규 데이터 일괄 결합
    new_data = pl.concat(new_dfs)
    new_dates = new_data["기준일"].unique().to_list()
    print(f"금회 파이프라인에서 새롭게 가공된 날짜: {new_dates}")
    
    # 2. 기존 가공 누적 파일 점검 및 결합
    if OUT_CSV.exists():
        print(f"기존 누적 파일이 발견되었습니다: {OUT_CSV.name}")
        try:
            existing_data = pl.read_csv(
                OUT_CSV,
                schema_overrides={
                    "LINK_ID": pl.String,
                    "기준일": pl.String,
                    "시간코드": pl.String,
                    "평균속도": pl.Float32,
                    "sample_count_5min": pl.Int32
                }
            )
            # 신규 날짜와 겹치는 과거 데이터만 격리 후 덮어쓰기 (파일 잠금 해제를 위해 clone() 호출)
            preserved_data = existing_data.filter(~pl.col("기준일").is_in(new_dates)).clone()
            final_data = pl.concat([preserved_data, new_data])
            print(f"기존 데이터(보존 날짜)와 금회 가공 데이터를 멱등성 있게 병합했습니다.")
        except Exception as e:
            print(f"기존 데이터 파일 로드 중 에러가 발생하여 새 데이터로 덮어씁니다: {e}")
            final_data = new_data
    else:
        final_data = new_data
        
    # 3. 날짜, 시간코드, 링크별로 깔끔하게 데이터 정렬
    final_data = final_data.sort(["기준일", "시간코드", "LINK_ID"])
    
    # 4. 최종 데이터 물리 디렉토리 저장
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    final_data.write_csv(OUT_CSV)
    print(f"최종 가공 속도 팩트 재료 누적 저장 완료: {OUT_CSV}")
    print(f"최종 파일 행 크기: {len(final_data):,} 행")
    return final_data


# =====================================================================
# 3. 메인 파이프라인 컨트롤러
# =====================================================================
def main() -> None:
    print("=================================================================")
    print(" 🚀  SEOUL ECO-ROUTING: POLARS 5-MIN SPEED AGGREGATOR (STEP 02)")
    print("=================================================================")
    
    start_time = time.time()
    seoul_ids = read_seoul_link_ids()

    # 1. 기존 누적 파일에 있는 기 가공 날짜 동적 파악
    existing_dates = set()
    if OUT_CSV.exists():
        try:
            df_existing_dates = pl.read_csv(OUT_CSV, columns=["기준일"], schema_overrides={"기준일": pl.String})
            existing_dates = set(df_existing_dates["기준일"].unique().to_list())
            print(f"📂 기존 파일에 이미 가공 완료된 날짜: {sorted(list(existing_dates))}")
        except Exception as e:
            print(f"⚠️ 기존 가공 날짜 목록 파악 실패: {e}")
    
    # 2. 수집 대상 원천 5분 속도 파일 리스트 수집
    paths = sorted(FIVEMIN_DIR.glob(FIVEMIN_GLOB)) if FIVEMIN_DIR.exists() else []
    if not paths:
        print(f"❌ 5분 데이터 원본 파일이 폴더에 없습니다: {FIVEMIN_DIR / FIVEMIN_GLOB}")
        print("💡 data/raw/traffic/202607_5Min/ 폴더 안에 YYYYMMDD_5Min.csv 파일을 넣어주세요.")
        sys.exit(1)
        
    print(f"📂 발견된 원천 5분 데이터 파일 수: {len(paths)}개")
    for p in paths:
        print(f"  • {p.name} ({p.stat().st_size / (1024**3):.2f} GB)")
        
    # 3. 신규 날짜에 대해서만 필터링하여 집계 수행 (중복 스킵 핵심 로직)
    processed_dfs = []
    skipped_count = 0
    for p in paths:
        # 파일명에서 8자리 날짜 코드 추출
        match = re.search(r"(\d{8})_5Min", p.name)
        file_date = match.group(1) if match else None
        
        if file_date and file_date in existing_dates:
            print(f"  • {p.name} ➔ [SKIP] 이미 누적 데이터에 가공된 날짜이므로 집계를 스킵합니다.")
            skipped_count += 1
            continue
            
        print(f"  • {p.name} ({p.stat().st_size / (1024**3):.2f} GB) ➔ [PROCESS] 가공을 시작합니다.")
        df_file_agg = aggregate_one_file(p, seoul_ids)
        processed_dfs.append(df_file_agg)
        
    # 4. 점진적 누적 업데이트 및 최종 병합 저장
    combined = merge_and_save(processed_dfs)
    
    total_elapsed = time.time() - start_time
    print("=================================================================")
    print(f" 🎉  전체 교통 속도 가공 파이프라인 가속화 완료! (총 소요 시간: {total_elapsed:.2f}초)")
    print("=================================================================")

    if combined is not None and not combined.is_empty():
        print()
        print("이번 실행에서 처리한 일자 수:", len(paths) - skipped_count)
        print("저장 파일에 누적된 전체 일자 수:", combined["기준일"].n_unique())
        print("서울 링크 수:", len(seoul_ids))
        print("누적 (링크, 날짜, 시간코드) 행 수:", len(combined))
        print("누적 고유 LINK_ID 수:", combined["LINK_ID"].n_unique(), f"({combined['LINK_ID'].n_unique() / len(seoul_ids) * 100:.2f}%)")
        print("시간당 5분 샘플 수 평균:", round(combined["sample_count_5min"].mean(), 2), "(최대 12)")
        print("저장 파일:", OUT_CSV)


if __name__ == "__main__":
    main()