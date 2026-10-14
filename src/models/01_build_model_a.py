"""관측 링크(70.75%)를 위한 (LINK_ID, 시간코드, is_weekend) 조회표를 만든다.

notebooks/03_model_a_history.ipynb에서 시간 분할(과거 11일 학습 -> 미래 3일
평가, 진짜 미래예측 시나리오)로 검증된 설계를 그대로 가져온다:
  - 조회표(link x hour x 주중/주말) 단독: R2=0.9249
  - + LightGBM(사전계획 피처: lag_24h/lag_168h 등, 실시간 피처 제외): R2=0.9276
  둘의 차이가 +0.0027뿐이라 학습 비용 없이 조회표만 쓴다.
  - 요일을 7단계로 세분하면 셀당 표본이 부족해 과적합된다(같은 노트북에서
    T-data 실험으로 이미 확인) -> 주중/주말 2단계만 쓴다.

배포 시점엔 라이브 피드가 없어 lag_1h/roll_3h(직전 시간 실측) 같은 실시간
피처를 쓸 수 없다 -> "사전계획" 시나리오만 해당하므로 위 결과가 정확히 우리
상황이다.

시간코드는 T-data 규칙(1~24, 00시대=01 ... 23시대=24)을 그대로 쓴다 -
notebooks/03_model_a_history.ipynb와 동일 키라 검증된 성능이 그대로 적용된다
(notebooks/02_derived_features_lightgbm.ipynb 등 다른 곳에서 쓰는
"시간코드-1=실제시(0~23시)" 변환과는 다른 키이니 섞어 쓰지 않는다).

결측 조합(관측이 빠진 링크x시간x주중주말)은 상위 키로 폴백한다:
  link x hour x weekend -> link x hour -> link -> 전체 평균

출력: data/processed/model_a_lookup.parquet
  컬럼: LINK_ID, 시간코드(1~24), is_weekend(0/1), pred_speed, n_obs, source

사용: python src/models/01_build_model_a.py
"""
from __future__ import annotations

from itertools import product
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FACT_PATH = PROCESSED_DIR / "seoul_link_speed_fact.csv"
OUT_PATH = PROCESSED_DIR / "model_a_lookup.parquet"

TARGET = "평균속도"


def main() -> None:
    fact = pd.read_csv(
        FACT_PATH, dtype={"LINK_ID": str, "기준일": str}, encoding="utf-8-sig",
        usecols=["LINK_ID", "기준일", "시간코드", TARGET],
    )
    fact[TARGET] = pd.to_numeric(fact[TARGET], errors="coerce")
    fact["시간코드"] = pd.to_numeric(fact["시간코드"], errors="coerce").astype(int)
    fact = fact.dropna(subset=[TARGET])

    date = pd.to_datetime(fact["기준일"], format="%Y%m%d")
    fact["is_weekend"] = (date.dt.dayofweek >= 5).astype("int8")

    global_mean = fact[TARGET].mean()
    lut_link = fact.groupby("LINK_ID")[TARGET].mean()
    lut_link_hour = fact.groupby(["LINK_ID", "시간코드"])[TARGET].mean()
    fine = fact.groupby(["LINK_ID", "시간코드", "is_weekend"])[TARGET].agg(pred_speed="mean", n_obs="count")

    link_ids = fact["LINK_ID"].unique()
    full_index = pd.MultiIndex.from_tuples(
        list(product(link_ids, range(1, 25), [0, 1])), names=["LINK_ID", "시간코드", "is_weekend"]
    )
    lookup = fine.reindex(full_index).reset_index()

    fallback_hour = lookup.set_index(["LINK_ID", "시간코드"]).index.map(lut_link_hour)
    fallback_link = lookup["LINK_ID"].map(lut_link)
    source = pd.Series("link_hour_weekend", index=lookup.index)
    need_fallback = lookup["pred_speed"].isna()
    source[need_fallback & fallback_hour.notna()] = "link_hour"
    lookup["pred_speed"] = lookup["pred_speed"].fillna(pd.Series(fallback_hour, index=lookup.index))
    need_fallback = lookup["pred_speed"].isna()
    source[need_fallback & fallback_link.notna()] = "link"
    lookup["pred_speed"] = lookup["pred_speed"].fillna(fallback_link)
    need_fallback = lookup["pred_speed"].isna()
    source[need_fallback] = "global_mean"
    lookup["pred_speed"] = lookup["pred_speed"].fillna(global_mean)

    lookup["pred_speed"] = lookup["pred_speed"].astype("float32")
    lookup["n_obs"] = lookup["n_obs"].fillna(0).astype("int32")
    lookup["source"] = source

    lookup.to_parquet(OUT_PATH, index=False)

    print("Model A 조회표 저장:", OUT_PATH)
    print("행 수(링크 x 시간코드 x 주중주말):", len(lookup), "| 관측 링크 수:", len(link_ids))
    print("전체 평균:", round(global_mean, 2), "km/h")
    print("폴백 비율:")
    print((lookup["source"].value_counts(normalize=True) * 100).round(2).to_string())


if __name__ == "__main__":
    main()
