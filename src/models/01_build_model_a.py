"""Model A: 관측 이력이 있는 링크의 (LINK_ID, 시간코드, 평일/주말) 평균속도 조회표를 만든다.

조합에 관측이 없으면 링크×시간 → 링크 → 전체 평균 순으로 대체한다.
notebooks/03의 시간 순서 분할(과거 기간 학습 → 이후 3일 평가)에서 조회표만으로 R² 0.925를 얻었고,
과거 패턴 변수를 더한 LightGBM(0.928)과 차이가 작아 학습 없는 조회표를 택했다.
요일별(7단계) 구분은 칸당 표본이 부족해 과적합되어 평일/주말 2단계만 쓴다.

입력: data/processed/seoul_link_speed_fact.csv(pipeline/05)
출력: data/processed/model_a_lookup.parquet
      LINK_ID, 시간코드(1~24), is_weekend(0/1), pred_speed, n_obs, source(사용한 대체 단계)
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
