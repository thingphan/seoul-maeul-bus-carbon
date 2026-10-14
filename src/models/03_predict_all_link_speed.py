"""저장된 Model A/B/C 아티팩트로 특정 시각의 서울 전 링크 속도를 예측한다.

재학습 없이 예측만 하는 배포용 단계다. 이 스크립트는 이미
학습·저장된 아래 파일만 사용한다.

* data/processed/model_a_lookup.parquet, neighbor_speed_lookup.parquet
* src/models/artifacts/model_nb.pkl, model_stat.pkl

교통량·생활인구처럼 미래 시각에 실측값이 없는 변수는 같은 시간대·평일/주말의 과거 평균을 쓴다.

출력: data/processed/all_link_speed_prediction_{tm}.csv(63,548개 링크의 speed_kmh, tier)
사용:
    python src/models/03_predict_all_link_speed.py --tm 202607160800                 # 레이더 강수(weather_snapshot.py 결과)
    python src/models/03_predict_all_link_speed.py --tm 202607160800 --precip-mm 0   # 강수 시나리오
"""

from __future__ import annotations

import argparse
import importlib.util
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = PROJECT_ROOT / "data" / "processed"
ARTIFACTS = PROJECT_ROOT / "src" / "models" / "artifacts"
BUILDER_PATH = PROJECT_ROOT / "src" / "models" / "02_build_model_bc.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="저장된 A/B/C 모델로 서울 전 링크 속도를 예측합니다.")
    parser.add_argument("--tm", required=True, help="예측 시각 YYYYMMDDHHMM")
    weather_group = parser.add_mutually_exclusive_group()
    weather_group.add_argument(
        "--weather-file", type=Path,
        help="LINK_ID, precip_mm 컬럼을 가진 링크별 강수 CSV. 생략 시 표준 weather 스냅샷을 사용합니다.",
    )
    weather_group.add_argument(
        "--precip-mm", type=float,
        help="전 링크에 적용할 균일 강수량(mm). 날씨 시나리오 비교용입니다.",
    )
    parser.add_argument("--output", type=Path, help="출력 CSV 경로")
    return parser.parse_args()


def load_builder_module():
    """학습 스크립트의 상수와 파생 피처 함수를 그대로 재사용한다."""
    spec = importlib.util.spec_from_file_location("model_bc_builder", BUILDER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"학습 스크립트를 불러올 수 없습니다: {BUILDER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_tm(tm: str) -> pd.Timestamp:
    if len(tm) != 12 or not tm.isdigit():
        raise SystemExit("--tm은 YYYYMMDDHHMM 형식이어야 합니다. 예: 202607160800")
    try:
        return pd.to_datetime(tm, format="%Y%m%d%H%M")
    except ValueError as exc:
        raise SystemExit(f"올바르지 않은 시각입니다: {tm}") from exc


def load_dimension(builder) -> pd.DataFrame:
    """학습 때 쓴 링크 차원 테이블을 읽고 정적 파생 피처를 더한다."""
    path = PROCESSED / "seoul_link_dimension.csv"
    if not path.exists():
        raise SystemExit(f"링크 차원 테이블이 없습니다: {path}")

    dtype = {col: str for col in builder.CATEGORICAL_COLS if col != "c_its"}
    dtype.update({"LINK_ID": str, "F_NODE": str, "T_NODE": str, "C-ITS": str, "ROAD_NAME": str})
    dimension = pd.read_csv(path, dtype=dtype, encoding="utf-8-sig")
    if "C-ITS" in dimension.columns:
        dimension = dimension.rename(columns={"C-ITS": "c_its"})

    # 학습 때 숫자로 다룬 열은 동일하게 숫자형으로 통일한다.
    numeric = set(builder.NUMERIC_COLS + builder.EXTERNAL_CONTEXT_COLS + builder.SPARSE_NUMERIC_COLS
                  + builder.TOPO_COLS_PRUNED + builder.POPULATION_COLS)
    numeric.update(c for c in builder.VIEWT_COLS if c != "avg_speed_hourly")
    numeric.update(f"avg_speed_h{hour}" for hour in range(24))
    for col in numeric & set(dimension.columns):
        dimension[col] = pd.to_numeric(dimension[col], errors="coerce")

    corridor = builder.build_corridor_and_cbd(dimension)
    turn = builder.build_turn_features(dimension)
    return dimension.merge(corridor, on="LINK_ID", how="left", validate="one_to_one").merge(
        turn, on="LINK_ID", how="left", validate="one_to_one"
    )


def load_weather(args: argparse.Namespace, link_ids: pd.Series) -> pd.DataFrame:
    if args.precip_mm is not None:
        if args.precip_mm < 0:
            raise SystemExit("--precip-mm은 0 이상이어야 합니다.")
        return pd.DataFrame({"LINK_ID": link_ids, "precip_mm": np.float32(args.precip_mm)})

    path = args.weather_file or PROCESSED / "weather_snapshot" / f"weather_snapshot_{args.tm}.csv"
    if not path.exists():
        raise SystemExit(
            f"링크별 강수 파일이 없습니다: {path}\n"
            "python src/models/weather_snapshot.py --tm <시각>으로 레이더 스냅샷을 만들거나, 강수 시나리오라면 --precip-mm 0처럼 지정하세요."
        )
    weather = pd.read_csv(path, usecols=["LINK_ID", "precip_mm"], dtype={"LINK_ID": str}, encoding="utf-8-sig")
    weather["precip_mm"] = pd.to_numeric(weather["precip_mm"], errors="coerce")
    return weather.drop_duplicates("LINK_ID")


def load_dynamic_profile(hour_code: int, is_weekend: int) -> pd.DataFrame:
    """미래 시각에도 쓸 수 있게 과거 동일 시간대·주말 여부의 평균 프로필을 만든다.

    B/C 모델이 학습 때 사용한 교통량·생활인구 피처는 특정 시각의 라이브 값이 없으므로,
    해당 링크의 동일 시간대 평균을 쓴다. 관측 이력이 없는 링크의 결측은 XGBoost가
    학습 때와 마찬가지로 결측값으로 처리한다.
    """
    fact_path = PROCESSED / "seoul_link_speed_fact.csv"
    value_cols = ["traffic_volume_hourly", "traffic_volume_ratio", "grid_pop", "dong_pop"]
    if not fact_path.exists():
        raise SystemExit(f"속도 팩트 테이블이 없습니다: {fact_path}")

    sums: pd.DataFrame | None = None
    counts: pd.DataFrame | None = None
    usecols = ["LINK_ID", "기준일", "시간코드", *value_cols]
    for chunk in pd.read_csv(
        fact_path, usecols=usecols, dtype={"LINK_ID": str, "기준일": str},
        encoding="utf-8-sig", chunksize=500_000,
    ):
        chunk["시간코드"] = pd.to_numeric(chunk["시간코드"], errors="coerce")
        dates = pd.to_datetime(chunk["기준일"], format="%Y%m%d", errors="coerce")
        mask = (chunk["시간코드"] == hour_code) & ((dates.dt.dayofweek >= 5) == bool(is_weekend))
        selected = chunk.loc[mask, ["LINK_ID", *value_cols]]
        if selected.empty:
            continue
        for col in value_cols:
            selected[col] = pd.to_numeric(selected[col], errors="coerce")
        grouped_sum = selected.groupby("LINK_ID")[value_cols].sum(min_count=1)
        grouped_count = selected.groupby("LINK_ID")[value_cols].count()
        sums = grouped_sum if sums is None else sums.add(grouped_sum, fill_value=0)
        counts = grouped_count if counts is None else counts.add(grouped_count, fill_value=0)

    if sums is None or counts is None:
        raise SystemExit("요청한 시간대·주말 여부에 해당하는 과거 프로필 행이 없습니다.")
    profile = sums.divide(counts.where(counts > 0)).reset_index()
    return profile


def prepare_features(builder, dimension: pd.DataFrame, tm: pd.Timestamp, weather: pd.DataFrame) -> pd.DataFrame:
    hour_code = tm.hour + 1  # 프로젝트 시간코드: 00시=1, ..., 23시=24
    is_weekend = int(tm.dayofweek >= 5)
    frame = dimension.merge(weather, on="LINK_ID", how="left", validate="one_to_one")
    profile = load_dynamic_profile(hour_code, is_weekend)
    frame = frame.merge(profile, on="LINK_ID", how="left", validate="one_to_one")

    frame["시간코드"] = np.float32(hour_code)
    # 학습 데이터에는 월~일 전체가 있어 category index가 0~6이었다. 한 시각만
    # 예측할 때도 같은 전체 범주를 선언해야 XGBoost가 범주 코드를 동일하게 해석한다.
    frame["dow"] = pd.Categorical(
        [tm.dayofweek] * len(frame), categories=np.arange(7, dtype="int32")
    )
    frame["is_weekend"] = np.int8(is_weekend)
    frame["hour_sin"] = np.float32(np.sin(2 * np.pi * hour_code / 24))
    frame["hour_cos"] = np.float32(np.cos(2 * np.pi * hour_code / 24))

    actual_hour = tm.hour
    hour_profile_col = f"avg_speed_h{actual_hour}"
    if hour_profile_col not in frame.columns:
        raise SystemExit(f"차원 테이블에 필요한 시간대 속도 프로필이 없습니다: {hour_profile_col}")
    frame["avg_speed_hourly"] = pd.to_numeric(frame[hour_profile_col], errors="coerce")

    for col in builder.CATEGORICAL_COLS:
        if col not in frame.columns:
            raise SystemExit(f"차원 테이블에 모델 필수 범주형 피처가 없습니다: {col}")
        frame[col] = frame[col].astype("category")
    return frame


def model_feature_columns(artifact: dict) -> list[str]:
    """학습된 Booster가 기억한 순서를 우선한다.

    초기 아티팩트에는 ``feature_cols`` 메타데이터의 이웃속도 열 위치가 실제 XGBoost
    학습 입력과 다르게 저장된 경우가 있다. Booster의 feature_names는 학습에 사용된
    진짜 순서이므로, 예측에는 항상 그것을 쓴다.
    """
    names = artifact["model"].get_booster().feature_names
    return list(names) if names else list(artifact["feature_cols"])


def align_model_categories(frame: pd.DataFrame, artifact: dict) -> None:
    """Booster에 저장된 범주 목록·타입과 DataFrame을 정확히 맞춘다."""
    categories = artifact["model"].get_booster().get_categories(export_to_arrow=True).to_arrow()
    for name, values in categories:
        if values is None or name not in frame.columns:
            continue
        # values는 pyarrow Array다. to_pylist()는 int32를 Python int로 바꿔 버리므로
        # numpy 배열을 유지해 dow 같은 범주의 index dtype도 학습 당시와 맞춘다.
        frame[name] = pd.Categorical(frame[name], categories=values.to_numpy(zero_copy_only=False))


def output_columns(builder) -> list[str]:
    return ["LINK_ID", "F_NODE", "T_NODE", "LENGTH", "ROAD_NAME", "MAX_SPD", "ROAD_RANK",
            "slope_pct", "precip_mm", "시간코드", "is_weekend", builder.NEIGHBOR_COL,
            "speed_kmh", "tier", "speed_source", "prediction_confidence"]


def predict_speed(frame: pd.DataFrame, builder, nb_artifact: dict, stat_artifact: dict,
                  nb_features: list[str], stat_features: list[str]) -> None:
    """A 조회값이 있으면 A, 없고 이웃속도가 있으면 B, 둘 다 없으면 C로 속도를 채운다(frame에 열 추가).

    예측속도는 3km/h와 링크 제한속도 사이로 제한한다.
    """
    is_a = frame["model_a_speed"].notna()
    is_b = ~is_a & frame[builder.NEIGHBOR_COL].notna()
    is_c = ~is_a & ~is_b
    speed = pd.Series(np.nan, index=frame.index, dtype="float32")
    speed.loc[is_a] = frame.loc[is_a, "model_a_speed"].to_numpy(dtype="float32")
    speed.loc[is_b] = nb_artifact["model"].predict(frame.loc[is_b, nb_features])
    speed.loc[is_c] = stat_artifact["model"].predict(frame.loc[is_c, stat_features])
    max_speed = pd.to_numeric(frame["MAX_SPD"], errors="coerce").fillna(110).clip(lower=3)
    frame["speed_kmh"] = speed.clip(lower=3).clip(upper=max_speed).round(3)
    frame["tier"] = np.select([is_a, is_b], ["A_lookup", "B_neighbor_model"], default="C_static_model")
    frame["speed_source"] = np.select([is_a, is_b], ["model_a_lookup", "model_nb"], default="model_stat")
    frame["prediction_confidence"] = np.select([is_a, is_b], ["high", "medium"], default="low")


def main() -> None:
    args = parse_args()
    tm = validate_tm(args.tm)
    builder = load_builder_module()

    required = [PROCESSED / "model_a_lookup.parquet", PROCESSED / "neighbor_speed_lookup.parquet",
                ARTIFACTS / "model_nb.pkl", ARTIFACTS / "model_stat.pkl"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("필수 모델/조회표 파일이 없습니다:\n- " + "\n- ".join(missing))

    with open(ARTIFACTS / "model_nb.pkl", "rb") as file:
        nb_artifact = pickle.load(file)
    with open(ARTIFACTS / "model_stat.pkl", "rb") as file:
        stat_artifact = pickle.load(file)
    nb_features = model_feature_columns(nb_artifact)
    stat_features = model_feature_columns(stat_artifact)

    print("[1/4] 링크 정적 피처 구성...")
    dimension = load_dimension(builder)
    weather = load_weather(args, dimension["LINK_ID"])
    frame = prepare_features(builder, dimension, tm, weather)

    hour_code = tm.hour + 1
    is_weekend = int(tm.dayofweek >= 5)
    print("[2/4] Model A·이웃속도 조회...")
    model_a = pd.read_parquet(PROCESSED / "model_a_lookup.parquet")
    a_now = model_a[(model_a["시간코드"] == hour_code) & (model_a["is_weekend"] == is_weekend)][
        ["LINK_ID", "pred_speed", "source"]
    ].rename(columns={"pred_speed": "model_a_speed", "source": "model_a_source"})
    neighbor = pd.read_parquet(PROCESSED / "neighbor_speed_lookup.parquet")
    neighbor_now = neighbor[(neighbor["시간코드"] == hour_code) & (neighbor["is_weekend"] == is_weekend)][
        ["LINK_ID", builder.NEIGHBOR_COL]
    ]
    frame = frame.merge(a_now, on="LINK_ID", how="left", validate="one_to_one").merge(
        neighbor_now, on="LINK_ID", how="left", validate="one_to_one"
    )

    for feature in set(nb_features + stat_features):
        if feature not in frame.columns:
            frame[feature] = np.nan
    missing_nb = set(nb_features) - set(frame.columns)
    missing_stat = set(stat_features) - set(frame.columns)
    if missing_nb or missing_stat:
        raise SystemExit(f"모델 입력 피처 구성 실패: {sorted(missing_nb | missing_stat)}")
    align_model_categories(frame, nb_artifact)
    align_model_categories(frame, stat_artifact)

    print("[3/4] B/C 모델 예측...")
    predict_speed(frame, builder, nb_artifact, stat_artifact, nb_features, stat_features)

    output = args.output or PROCESSED / f"all_link_speed_prediction_{args.tm}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    keep = [col for col in output_columns(builder) if col in frame.columns]
    frame[keep].to_csv(output, index=False, encoding="utf-8-sig")

    print(f"[4/4] 저장 완료: {output}")
    print(f"링크 수: {len(frame):,} | 속도 결측: {int(frame['speed_kmh'].isna().sum())}")
    print(frame["tier"].value_counts().to_string())
    print(f"평균 속도: {frame['speed_kmh'].mean():.2f} km/h")


if __name__ == "__main__":
    main()
