"""네 유형(중앙값 규칙) 대신 군집분석을 쓸 수 있는지 확인한다.

K-평균의 실루엣, 가우시안 혼합의 BIC, HDBSCAN의 군집 수·잡음 비율, 그리고 수동 네 유형과의 일치도(ARI)를 본다.
군집이 뚜렷하지 않으면(실루엣 낮음, 최적 군집 수 불명확, 잡음 비율 높음) 해석이 쉬운 중앙값 규칙을 쓴다.

입력: data/processed/transit/route_priority.csv(06), route_demand.csv(02), route_feasibility.csv(03)
사용: python src/transit/checks/cluster_check.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN, AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[3]
TRANSIT = PROJECT_ROOT / "data" / "processed" / "transit"

BASE = ["catchment_elderly_share", "catchment_elderly_day", "no_alt_boarding_share", "bus_g_per_pkm"]
RICH = BASE + ["route_cars_per_1000", "boardings_per_day", "slope_mean", "one_lane_share", "day_speed_kmh", "trips_per_day"]


def load() -> pd.DataFrame:
    read = lambda name: pd.read_csv(TRANSIT / name, dtype={"ROUTE_ID": str})  # noqa: E731
    pr = read("route_priority.csv")
    pr = pr.merge(read("route_demand.csv")[["ROUTE_ID", "route_cars_per_1000"]], on="ROUTE_ID", how="left")
    feas = read("route_feasibility.csv")[["ROUTE_ID", "slope_mean", "one_lane_share", "day_speed_kmh"]]
    return pr.merge(feas, on="ROUTE_ID", how="left")


def run(pr: pd.DataFrame, features: list[str], label: str) -> None:
    X = StandardScaler().fit_transform(pr[features].fillna(pr[features].median()))
    manual = pr["class_id"].to_numpy()
    print(f"\n=== {label} ({len(features)}개 변수) ===")
    sil = {k: silhouette_score(X, KMeans(k, n_init=50, random_state=0).fit(X).labels_) for k in range(2, 7)}
    print("  K-평균 실루엣(1에 가까울수록 뚜렷):", {k: round(v, 3) for k, v in sil.items()})
    bic = {k: GaussianMixture(k, n_init=10, random_state=0).fit(X).bic(X) for k in range(1, 7)}
    print("  가우시안 혼합 BIC(낮을수록 좋음):", {k: round(v) for k, v in bic.items()}, "→ 최적 k =", min(bic, key=bic.get))
    km4 = KMeans(4, n_init=50, random_state=0).fit(X).labels_
    ward4 = AgglomerativeClustering(4, linkage="ward").fit_predict(X)
    print(f"  네 유형과 일치도(ARI, 1=완전 일치, 0=무작위): K-평균 {adjusted_rand_score(manual, km4):.2f}, "
          f"계층(Ward) {adjusted_rand_score(manual, ward4):.2f}")
    for size in (5, 10, 15):
        hdb = HDBSCAN(min_cluster_size=size, copy=True).fit_predict(X)
        n_clusters = len(set(hdb) - {-1})
        print(f"  HDBSCAN(min_cluster_size={size}): 군집 {n_clusters}개, 잡음 {np.mean(hdb == -1):.0%}, "
              f"네 유형과 ARI {adjusted_rand_score(manual, hdb):.2f}")


def main() -> None:
    pr = load()
    run(pr, BASE, "네 유형 분류와 같은 변수")
    run(pr, RICH, "변수 확장(승용차 보유·이용량·경사·골목·속도·운행횟수)")


if __name__ == "__main__":
    main()
