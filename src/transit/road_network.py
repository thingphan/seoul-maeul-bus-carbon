"""표준노드링크 도로망 불러오기와 최단 거리·노인 보행 시간 계산(01 맵매칭, 09·11 보행 공용)."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LINK_SHP = PROJECT_ROOT / "data" / "interim" / "seoul_link.shp"
DIMENSION_CSV = PROJECT_ROOT / "data" / "processed" / "seoul_link_dimension.csv"


def load_network(extra_cols: tuple[str, ...] = ()) -> tuple[gpd.GeoDataFrame, pd.Index, np.ndarray]:
    """(링크 표, 노드 ID 색인, 노드 좌표 배열). 링크 표에는 LINK_ID, geometry, F_NODE, T_NODE, LENGTH가 있다."""
    links = gpd.read_file(LINK_SHP)[["LINK_ID", "geometry"]]
    links["LINK_ID"] = links["LINK_ID"].astype(str)
    dimension = pd.read_csv(
        DIMENSION_CSV, usecols=["LINK_ID", "F_NODE", "T_NODE", "LENGTH", *extra_cols],
        dtype={"LINK_ID": str, "F_NODE": str, "T_NODE": str},
    )
    links = links.merge(dimension, on="LINK_ID", how="inner").reset_index(drop=True)
    nodes = pd.Index(pd.unique(pd.concat([links["F_NODE"], links["T_NODE"]])))
    # 노드 좌표 = 그 노드가 처음 나오는 링크의 시작점(F_NODE) 또는 끝점(T_NODE). 링크 순서대로 F, T를 번갈아 본다.
    geoms = links.geometry.to_numpy()
    start, end = shapely.get_point(geoms, 0), shapely.get_point(geoms, -1)
    ends = pd.DataFrame({
        "node": np.column_stack([links["F_NODE"], links["T_NODE"]]).ravel(),
        "x": np.column_stack([shapely.get_x(start), shapely.get_x(end)]).ravel(),
        "y": np.column_stack([shapely.get_y(start), shapely.get_y(end)]).ravel(),
    }).drop_duplicates("node").set_index("node")
    node_xy = ends.loc[nodes, ["x", "y"]].to_numpy()
    return links, nodes, node_xy


def edge_graph(u: np.ndarray, v: np.ndarray, w: np.ndarray, n_nodes: int) -> csr_matrix:
    """방향 간선 (u→v, 비용 w)으로 희소 그래프를 만든다. 같은 노드 쌍의 간선이 여럿이면 가장 싼 것만 남긴다."""
    edges = pd.DataFrame({"u": u, "v": v, "w": w})
    edges = edges[edges["u"] != edges["v"]].sort_values("w").drop_duplicates(["u", "v"])
    return csr_matrix((edges["w"], (edges["u"], edges["v"])), shape=(n_nodes, n_nodes))


def nearest_source_cost(graph: csr_matrix, source_nodes: np.ndarray, source_cost: np.ndarray,
                        limit: float = np.inf) -> np.ndarray:
    """모든 노드에서 '가장 가까운 출발점'까지의 최소 비용(출발점마다 시작 비용을 더한다).

    graph는 '노드 → 출발점' 방향으로 비용이 정의된 그래프다. 출발점이 많을 때 한 번의 Dijkstra로 풀기 위해
    가상 노드 하나를 두고, 전치 그래프에서 가상 노드 → 각 출발점 노드(비용 = 시작 비용)를 잇는다.
    """
    n = graph.shape[0]
    table = pd.DataFrame({"node": source_nodes, "cost": source_cost}).groupby("node")["cost"].min()
    reverse = graph.T.tocoo()
    rows = np.r_[reverse.row, np.full(len(table), n)]
    cols = np.r_[reverse.col, table.index.to_numpy()]
    data = np.r_[reverse.data, table.to_numpy() + 1e-9]  # 비용 0 간선은 희소 행렬에서 사라지므로 아주 작은 값을 더한다
    extended = csr_matrix((data, (rows, cols)), shape=(n + 1, n + 1))
    return dijkstra(extended, directed=True, indices=n, limit=limit)[:n]


ELDERLY_SPEED_FACTOR = 0.75   # 노인 보행속도 = 일반 보행속도 × 0.75(평지 약 1.05m/s)
SLOPE_CLIP_PCT = 12.0         # 03과 같은 경사 상한(짧은 링크의 DEM 잡음 억제)


def tobler_kmh(grade: np.ndarray | float) -> np.ndarray:
    """Tobler 보행속도[km/h]. grade = 걷는 방향 경사(비율, 오르막 +)."""
    return 6.0 * np.exp(-3.5 * np.abs(np.asarray(grade) + 0.05))


def elderly_mps(grade: np.ndarray | float) -> np.ndarray:
    return tobler_kmh(grade) * ELDERLY_SPEED_FACTOR / 3.6


def link_grade(links: pd.DataFrame) -> np.ndarray:
    """링크 F→T 방향 경사(비율), ±SLOPE_CLIP_PCT로 제한. links에 slope_pct 열이 있어야 한다."""
    return links["slope_pct"].fillna(0).clip(-SLOPE_CLIP_PCT, SLOPE_CLIP_PCT).to_numpy() / 100.0


def elderly_walk_graph(fi: np.ndarray, ti: np.ndarray, length: np.ndarray, grade: np.ndarray, n_nodes: int) -> csr_matrix:
    """노인 보행 시간[초] 그래프. F→T로 걸으면 경사 grade, T→F로 걸으면 −grade."""
    return edge_graph(np.r_[fi, ti], np.r_[ti, fi], np.r_[length / elderly_mps(grade), length / elderly_mps(-grade)], n_nodes)
