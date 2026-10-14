"""탄소 효율 × 교통약자 의존도 네 유형 분류 규칙(06 분류, 08 도로망 거리 민감도가 함께 쓴다)."""

from __future__ import annotations

import numpy as np
import pandas as pd

DEPENDENCE = {
    "catchment_elderly_share": "고령 비율",
    "catchment_elderly_day": "고령 인구",
    "no_alt_boarding_share": "대체수단 없음",
}
CLASS_LABELS = {1: "1 증편·지원 우선", 2: "2 소형 차량 전환", 3: "3 현행 유지", 4: "4 노선 조정 검토"}


def assign_classes(data: pd.DataFrame, dependence_cols: list[str] | None = None) -> pd.DataFrame:
    """dependence_score(구성요소 백분위 순위 평균)와 class_id·class를 붙인다.

    의존도는 중앙값 이상이면 높음, 탄소 효율은 승객 1명·1km당 CO2(bus_g_per_pkm)가 중앙값 이하이면 높음.
    """
    cols = dependence_cols or list(DEPENDENCE)
    out = data.copy()
    out["dependence_score"] = out[cols].rank(pct=True).mean(axis=1)
    high_dep = out["dependence_score"] >= out["dependence_score"].median()
    efficient = out["bus_g_per_pkm"] <= out["bus_g_per_pkm"].median()
    out["class_id"] = np.select([high_dep & efficient, high_dep & ~efficient, ~high_dep & efficient], [1, 2, 3], 4)
    out["class"] = out["class_id"].map(CLASS_LABELS)
    return out
