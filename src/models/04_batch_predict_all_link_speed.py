"""저장된 A/B/C 모델로 기간 전체의 전 링크 시간별 속도를 예측한다.

03을 시점마다 실행하는 대신, 속도 팩트를 한 번만 읽어 시간대별 교통량·생활인구 프로필을 만들고
정적 변수를 재사용한다. 레이더 원본이 없는 시각은 건너뛰고, 이미 있는 출력은 --overwrite가 없으면 유지한다.

출력: data/processed/all_link_speed_prediction_{YYYYMMDDHHMM}.csv(시점별)
사용: python src/models/04_batch_predict_all_link_speed.py --start-date 20260707 --end-date 20260720
"""

from __future__ import annotations

import argparse
import importlib.util
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = PROJECT_ROOT / "data" / "processed"
RAW_RADAR = PROJECT_ROOT / "data" / "raw" / "weather" / "radar"
ARTIFACTS = PROJECT_ROOT / "src" / "models" / "artifacts"
PREDICTOR_PATH = PROJECT_ROOT / "src" / "models" / "03_predict_all_link_speed.py"
WEATHER_BUILDER_PATH = PROJECT_ROOT / "src" / "models" / "weather_snapshot.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"모듈을 불러올 수 없습니다: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="기간 전체 전 링크 속도 예측 배치")
    parser.add_argument("--start-date", default="20260707", help="시작일 YYYYMMDD")
    parser.add_argument("--end-date", default="20260720", help="종료일 YYYYMMDD")
    parser.add_argument("--overwrite", action="store_true", help="이미 있는 동일 시각 출력도 다시 만듭니다")
    return parser.parse_args()


def make_dynamic_profiles(fact_path: Path) -> pd.DataFrame:
    """팩트를 한 번만 읽어 링크×시간×주말여부별 동적 피처 평균을 만든다."""
    date_col, hour_col = "기준일", "시간코드"
    value_cols = ["traffic_volume_hourly", "traffic_volume_ratio", "grid_pop", "dong_pop"]
    sums: pd.DataFrame | None = None
    counts: pd.DataFrame | None = None

    for chunk in pd.read_csv(
        fact_path, usecols=["LINK_ID", date_col, hour_col, *value_cols], dtype={"LINK_ID": str, date_col: str},
        encoding="utf-8-sig", chunksize=500_000,
    ):
        dates = pd.to_datetime(chunk[date_col], format="%Y%m%d", errors="coerce")
        chunk[hour_col] = pd.to_numeric(chunk[hour_col], errors="coerce").astype("Int8")
        chunk["is_weekend"] = (dates.dt.dayofweek >= 5).astype("int8")
        for col in value_cols:
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce")
        grouped_sum = chunk.groupby(["LINK_ID", hour_col, "is_weekend"], observed=True)[value_cols].sum(min_count=1)
        grouped_count = chunk.groupby(["LINK_ID", hour_col, "is_weekend"], observed=True)[value_cols].count()
        sums = grouped_sum if sums is None else sums.add(grouped_sum, fill_value=0)
        counts = grouped_count if counts is None else counts.add(grouped_count, fill_value=0)

    if sums is None or counts is None:
        raise RuntimeError("속도 팩트에서 동적 프로필을 만들 수 없습니다.")
    profile = sums.divide(counts.where(counts > 0)).reset_index()
    profile = profile.rename(columns={hour_col: "시간코드"})
    profile["시간코드"] = profile["시간코드"].astype(int)
    return profile


def available_timestamps(start: pd.Timestamp, end: pd.Timestamp) -> list[str]:
    pattern = re.compile(r"hsr_pcph_acc60_(\d{12})\.txt$")
    found = {
        match.group(1)
        for path in RAW_RADAR.glob("hsr_pcph_acc60_*.txt")
        if (match := pattern.match(path.name))
    }
    requested = pd.date_range(start, end + pd.Timedelta(hours=23), freq="h").strftime("%Y%m%d%H00").tolist()
    return [tm for tm in requested if tm in found]


def weather_for_timestamp(weather_module, mapping: pd.DataFrame, tm: str) -> pd.DataFrame:
    path = RAW_RADAR / f"hsr_pcph_acc60_{tm}.txt"
    radar = weather_module.parse_kma_ascii_grid_fast(path).reshape(-1)
    raw = radar[mapping["radar_flat_index"].to_numpy(dtype=np.int64)]
    precip = raw.astype("float32") / 100.0
    precip[raw < 0] = 0.0
    precip[raw <= -30000] = np.nan
    return pd.DataFrame({"LINK_ID": mapping["LINK_ID"].to_numpy(), "precip_mm": precip})


def main() -> None:
    args = parse_args()
    try:
        start = pd.to_datetime(args.start_date, format="%Y%m%d")
        end = pd.to_datetime(args.end_date, format="%Y%m%d")
    except ValueError as exc:
        raise SystemExit("날짜는 YYYYMMDD 형식이어야 합니다.") from exc
    if end < start:
        raise SystemExit("종료일은 시작일보다 빠를 수 없습니다.")

    predictor = load_module("all_link_predictor", PREDICTOR_PATH)
    weather_module = load_module("weather_snapshot_builder", WEATHER_BUILDER_PATH)
    required = [PROCESSED / "seoul_link_dimension.csv", PROCESSED / "seoul_link_speed_fact.csv",
                PROCESSED / "model_a_lookup.parquet", PROCESSED / "neighbor_speed_lookup.parquet",
                ARTIFACTS / "model_nb.pkl", ARTIFACTS / "model_stat.pkl"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("필수 파일이 없습니다:\n- " + "\n- ".join(missing))

    print("[1/5] 모델·정적 링크 피처 로드...")
    builder = predictor.load_builder_module()
    dimension = predictor.load_dimension(builder)
    with open(ARTIFACTS / "model_nb.pkl", "rb") as file:
        nb_artifact = pickle.load(file)
    with open(ARTIFACTS / "model_stat.pkl", "rb") as file:
        stat_artifact = pickle.load(file)
    nb_features = predictor.model_feature_columns(nb_artifact)
    stat_features = predictor.model_feature_columns(stat_artifact)

    print("[2/5] 14일 동적 시간대 프로필 집계 (팩트 파일 1회 읽기)...")
    profiles = make_dynamic_profiles(PROCESSED / "seoul_link_speed_fact.csv")
    print(f"      프로필 행: {len(profiles):,}")

    print("[3/5] Model A·이웃 조회표 및 레이더 격자 매핑 로드...")
    model_a = pd.read_parquet(PROCESSED / "model_a_lookup.parquet")
    neighbor = pd.read_parquet(PROCESSED / "neighbor_speed_lookup.parquet")
    # 날씨 모듈은 Polars DataFrame을 돌려주므로 이후 pandas/numpy 처리에 맞게 변환한다.
    mapping = weather_module.load_or_build_grid_mapping().select(["LINK_ID", "radar_flat_index"]).to_pandas()
    timestamps = available_timestamps(start, end)
    requested_count = int((end - start).days + 1) * 24
    print(f"[4/5] 예측 대상: {len(timestamps)}/{requested_count}개 시간대 (레이더 없는 시각 자동 제외)")

    completed = 0
    for index, tm_text in enumerate(timestamps, start=1):
        output = PROCESSED / f"all_link_speed_prediction_{tm_text}.csv"
        if output.exists() and not args.overwrite:
            completed += 1
            print(f"[{index}/{len(timestamps)}] {tm_text} 기존 파일 유지")
            continue
        tm = pd.to_datetime(tm_text, format="%Y%m%d%H%M")
        hour_code = tm.hour + 1
        weekend = int(tm.dayofweek >= 5)
        weather = weather_for_timestamp(weather_module, mapping, tm_text)
        profile = profiles[(profiles["시간코드"] == hour_code) & (profiles["is_weekend"] == weekend)]

        frame = dimension.merge(weather, on="LINK_ID", how="left", validate="one_to_one").merge(
            profile.drop(columns=["시간코드", "is_weekend"]), on="LINK_ID", how="left", validate="one_to_one"
        )
        frame["시간코드"] = np.float32(hour_code)
        frame["dow"] = pd.Categorical([tm.dayofweek] * len(frame), categories=np.arange(7, dtype="int32"))
        frame["is_weekend"] = np.int8(weekend)
        frame["hour_sin"] = np.float32(np.sin(2 * np.pi * hour_code / 24))
        frame["hour_cos"] = np.float32(np.cos(2 * np.pi * hour_code / 24))
        frame["avg_speed_hourly"] = pd.to_numeric(frame[f"avg_speed_h{tm.hour}"], errors="coerce")

        a_now = model_a[(model_a["시간코드"] == hour_code) & (model_a["is_weekend"] == weekend)][
            ["LINK_ID", "pred_speed"]
        ].rename(columns={"pred_speed": "model_a_speed"})
        nb_now = neighbor[(neighbor["시간코드"] == hour_code) & (neighbor["is_weekend"] == weekend)][
            ["LINK_ID", builder.NEIGHBOR_COL]
        ]
        frame = frame.merge(a_now, on="LINK_ID", how="left", validate="one_to_one").merge(
            nb_now, on="LINK_ID", how="left", validate="one_to_one"
        )
        for col in set(nb_features + stat_features):
            if col not in frame.columns:
                frame[col] = np.nan
        predictor.align_model_categories(frame, nb_artifact)
        predictor.align_model_categories(frame, stat_artifact)

        predictor.predict_speed(frame, builder, nb_artifact, stat_artifact, nb_features, stat_features)
        keep = [col for col in predictor.output_columns(builder) if col in frame.columns]
        frame[keep].to_csv(output, index=False, encoding="utf-8-sig")
        completed += 1
        print(f"[{index}/{len(timestamps)}] {tm_text} 완료 | 평균 {frame['speed_kmh'].mean():.2f} km/h")

    missing_timestamps = requested_count - len(timestamps)
    print(f"[5/5] 완료: {completed}개 파일 | 레이더 부재로 제외: {missing_timestamps}개")


if __name__ == "__main__":
    main()
