"""마을버스 노선 유형을 데이터 군집으로 찾고, 기준을 정해 나눈 네 칸(06)과 얼마나 겹치는지 확인한다.

입력 (data/processed/transit/)
  route_priority.csv(06), route_demand.csv(02), route_feasibility.csv(03)

처리
  1. 변수 10개를 표준화한다: 고령 비율, 고령 인구, 대체수단 없는 승차 비중, 승객 1명·1km당 CO2,
     노선 주변 승용차 보유, 하루 승차, 평균 경사, 1차로 비중, 낮 평균 속도, 하루 운행횟수.
  2. 군집이 뚜렷한지: K-평균 k=2~6 실루엣 계수(1에 가까울수록 잘 갈라짐), 가우시안 혼합 BIC.
  3. k=4 K-평균과 네 칸의 일치도(ARI, 1=같음, 0=무작위 수준). 계층 군집(Ward)도 함께 본다.
  4. 유형 이름은 군집별 표준화 중앙값으로 정한 규칙에 따라 차례로 붙인다:
       하루 승차가 가장 큰 군집 → 대량 수송형
       남은 군집 중 평균 경사가 가장 큰 군집 → 고지대 골목형
       남은 군집 중 승용차 보유가 가장 큰 군집 → 평지·승용차형
       나머지 → 한산형

출력 (data/processed/transit/)
  route_typology.csv   노선 1행: 군집 번호, 유형 이름, 네 칸
  typology_profile.csv 유형별 노선 수와 변수 중앙값

사용: python src/transit/08_route_typology.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRANSIT = PROJECT_ROOT / "data" / "processed" / "transit"
FEATURES = {
    "catchment_elderly_share": "고령 비율",
    "catchment_elderly_day": "고령 인구",
    "no_alt_boarding_share": "대체수단 없는 승차",
    "bus_g_per_pkm": "승객km당 CO2",
    "route_cars_per_1000": "승용차 보유",
    "boardings_per_day": "하루 승차",
    "slope_mean": "평균 경사",
    "one_lane_share": "1차로 비중",
    "day_speed_kmh": "낮 속도",
    "trips_per_day": "하루 운행",
}
NAMING_RULES = [("boardings_per_day", "대량 수송형"), ("slope_mean", "고지대 골목형"),
                ("route_cars_per_1000", "평지·승용차형")]
REST_NAME = "한산형"
K = 4
SEED = 0


def name_clusters(z_median: pd.DataFrame) -> dict[int, str]:
    names, left = {}, list(z_median.index)
    for col, name in NAMING_RULES:
        pick = z_median.loc[left, col].idxmax()
        names[pick] = name
        left.remove(pick)
    names.update({c: REST_NAME for c in left})
    return names


def main() -> None:
    data = pd.read_csv(TRANSIT / "route_priority.csv", dtype={"ROUTE_ID": str})
    demand = pd.read_csv(TRANSIT / "route_demand.csv", dtype={"ROUTE_ID": str})[["ROUTE_ID", "route_cars_per_1000"]]
    feas = pd.read_csv(TRANSIT / "route_feasibility.csv", dtype={"ROUTE_ID": str})[
        ["ROUTE_ID", "slope_mean", "one_lane_share", "day_speed_kmh"]]
    data = data.merge(demand, on="ROUTE_ID", how="left").merge(feas, on="ROUTE_ID", how="left")
    cols = list(FEATURES)
    missing = data[cols].isna().sum()
    if missing.any():
        print("  결측(중앙값으로 채움):", missing[missing > 0].to_dict())
    x = StandardScaler().fit_transform(data[cols].fillna(data[cols].median()))

    print("[1/3] 군집이 뚜렷한지...")
    for k in range(2, 7):
        labels = KMeans(k, n_init=50, random_state=SEED).fit_predict(x)
        print(f"  K-평균 k={k}: 실루엣 {silhouette_score(x, labels):.3f}")
    bic = {k: GaussianMixture(k, n_init=10, random_state=SEED).fit(x).bic(x) for k in range(1, 7)}
    print(f"  가우시안 혼합 BIC 최소 k = {min(bic, key=bic.get)} ({ {k: round(v) for k, v in bic.items()} })")

    print(f"[2/3] k={K} 군집과 네 칸 비교...")
    data["cluster"] = KMeans(K, n_init=50, random_state=SEED).fit_predict(x)
    ward = AgglomerativeClustering(K, linkage="ward").fit_predict(x)
    print(f"  네 칸과 일치도(ARI): K-평균 {adjusted_rand_score(data['class_id'], data['cluster']):.2f}, "
          f"Ward {adjusted_rand_score(data['class_id'], ward):.2f} | K-평균과 Ward 사이 "
          f"{adjusted_rand_score(data['cluster'], ward):.2f}")

    print("[3/3] 유형 이름 붙이기...")
    z = pd.DataFrame(x, columns=cols).assign(cluster=data["cluster"].to_numpy()).groupby("cluster").median()
    data["type"] = data["cluster"].map(name_clusters(z))
    data[["ROUTE_ID", "노선명", "cluster", "type", "class", *cols]].to_csv(
        TRANSIT / "route_typology.csv", index=False, encoding="utf-8-sig")
    profile = data.groupby("type")[cols].median().rename(columns=FEATURES)
    profile.insert(0, "노선수", data["type"].value_counts())
    profile.to_csv(TRANSIT / "typology_profile.csv", encoding="utf-8-sig")

    print(profile.round(2).T.to_string())
    print("\n  유형 × 네 칸(노선 수):")
    print(pd.crosstab(data["type"], data["class"]).to_string())


if __name__ == "__main__":
    main()
