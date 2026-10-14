"""Model B·C: 관측이 없는 링크의 속도를 XGBoost로 추정하고 검증한다.

- B(이웃속도 포함, 61개 변수): 1칸 이웃(없으면 2칸 이웃) 관측 링크의 속도를 변수로 쓴다.
- C(이웃속도 제외, 60개 변수): 이웃 관측도 없는 링크용 모델이다.

이웃속도는 '이웃 링크의 (시간코드, 평일/주말) 과거 평균'으로 정의한다(Model A 조회표와 같은 방식).
같은 시각의 이웃 실측값은 미래 시점에 얻을 수 없으므로, 학습과 예측에 같은 정의를 써서
학습·서빙 불일치를 막는다. 따라서 검증 성능이 곧 실제 적용 시 기대 성능이다.

  --mode validate  도로 단위 3-fold GroupKFold. fold마다 이웃속도 조회표를 학습 도로만으로 다시 만들어 누수를 막는다(저장 없음)
  --mode final     전체 자료로 B·C를 학습해 src/models/artifacts/에 저장하고, 이웃속도 조회표를 data/processed/에 저장(기본값)

입력: data/processed/seoul_link_dimension.csv(pipeline/04), seoul_link_speed_fact.csv(pipeline/05),
      model_a_lookup.parquet(01, final 모드), data/raw/road/MOCT_NODE.shp
사용:
  python src/models/02_build_model_bc.py --mode validate
  python src/models/02_build_model_bc.py --mode final
"""
from __future__ import annotations

import argparse
import gc
import pickle
from collections import defaultdict
from itertools import product
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.sparse import csr_matrix
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "src" / "models" / "artifacts"

DIMENSION_PATH = PROCESSED_DIR / "seoul_link_dimension.csv"
FACT_PATH = PROCESSED_DIR / "seoul_link_speed_fact.csv"
MODEL_A_PATH = PROCESSED_DIR / "model_a_lookup.parquet"
MOCT_NODE_SHP = RAW_DIR / "road" / "MOCT_NODE.shp"

TARGET = "평균속도"
RANDOM_STATE = 42
N_FOLDS = 3
BUCKET_HOURS = list(range(1, 25))  # 시간코드 그대로(1~24), Model A와 동일 키

NUMERIC_COLS = [
    "LANES", "MAX_SPD", "LENGTH", "시간코드", "elev_mean_m", "elev_delta_m",
    "slope_pct", "abs_slope_pct", "precip_mm",
]
CATEGORICAL_COLS = ["ROAD_RANK", "ROAD_TYPE", "ROAD_USE", "MULTI_LINK", "CONNECT",
                    "REST_VEH", "REST_W", "REST_H", "c_its"]
EXTERNAL_CONTEXT_COLS = ["bus_lane_ratio", "school_zone_count", "accident_hotspot_count", "speed_bump_count"]
SPARSE_NUMERIC_COLS = ["traffic_volume_aadt", "traffic_volume_hourly", "traffic_volume_ratio"]
TOPO_COLS_PRUNED = [
    "f_node_node_in_degree", "f_node_incoming_max_spd_mean", "f_node_incoming_lanes_mean",
    "f_node_node_out_degree", "f_node_outgoing_max_spd_mean", "f_node_outgoing_lanes_mean",
    "t_node_node_in_degree", "t_node_incoming_max_spd_mean", "t_node_incoming_lanes_mean",
    "t_node_node_out_degree", "t_node_outgoing_max_spd_mean", "t_node_outgoing_lanes_mean",
    "node_complexity", "downstream_speed_delta", "slope_pct_clipped",
    "slope_length_m", "max_spd_x_lanes", "length_x_node_complexity",
]
POPULATION_COLS = ["grid_pop", "dong_pop"]
VIEWT_COLS = ["annual_avg_speed", "evening_peak_speed", "avg_speed_hourly", "congest_freq", "congest_time"]
CORRIDOR_COLS = ["rn_links", "rn_len", "rn_lanes", "rn_maxspd", "dist_cbd"]
NEIGHBOR_COL = "neighbor_speed_bucket"

FEATURE_COLS = (
    NUMERIC_COLS + CATEGORICAL_COLS
    + ["dow", "is_weekend", "hour_sin", "hour_cos"] + EXTERNAL_CONTEXT_COLS + SPARSE_NUMERIC_COLS
    + [NEIGHBOR_COL]
    + TOPO_COLS_PRUNED + POPULATION_COLS + VIEWT_COLS + CORRIDOR_COLS + ["is_cross_t"]
)
STATIC_COLS = [c for c in FEATURE_COLS if c != NEIGHBOR_COL]
CBD_X, CBD_Y = 198000.0, 552000.0  # 서울시청 부근 좌표(EPSG:5179)


# ---------------------------------------------------------------- 데이터/피처 ----

def load_dimension_and_fact() -> tuple[pd.DataFrame, pd.DataFrame]:
    dim_dtype = {c: str for c in CATEGORICAL_COLS if c != "c_its"}
    dim_dtype.update({"LINK_ID": str, "F_NODE": str, "T_NODE": str, "C-ITS": str, "ROAD_NAME": str})
    # avg_speed_hourly는 차원 표의 avg_speed_h0~23 중 그 행의 시간 값을 assemble_features가 골라 만든다
    viewt_raw_cols = [c for c in VIEWT_COLS if c != "avg_speed_hourly"]
    dim_cols = (
        ["LINK_ID", "F_NODE", "T_NODE", "ROAD_NAME"] + [c for c in CATEGORICAL_COLS if c != "c_its"] + ["C-ITS"]
        + ["LANES", "MAX_SPD", "LENGTH", "elev_mean_m", "elev_delta_m", "slope_pct", "abs_slope_pct"]
        + EXTERNAL_CONTEXT_COLS + ["traffic_volume_aadt"] + TOPO_COLS_PRUNED + viewt_raw_cols
        + [f"avg_speed_h{h}" for h in range(24)]
    )
    dimension = pd.read_csv(DIMENSION_PATH, dtype=dim_dtype, usecols=dim_cols, encoding="utf-8-sig")
    dimension = dimension.rename(columns={"C-ITS": "c_its"})
    for col in ["LANES", "MAX_SPD", "LENGTH", "elev_mean_m", "elev_delta_m", "slope_pct", "abs_slope_pct"] + TOPO_COLS_PRUNED + viewt_raw_cols + [f"avg_speed_h{h}" for h in range(24)]:
        dimension[col] = pd.to_numeric(dimension[col], errors="coerce")

    fact = pd.read_csv(
        FACT_PATH, dtype={"LINK_ID": str, "기준일": str}, encoding="utf-8-sig",
        usecols=["LINK_ID", "기준일", "시간코드", TARGET, "precip_mm", "traffic_volume_hourly",
                 "traffic_volume_ratio", "grid_pop", "dong_pop", "sample_count_5min"],
    )
    for col in [TARGET, "precip_mm", "traffic_volume_hourly", "traffic_volume_ratio",
                "grid_pop", "dong_pop", "sample_count_5min"]:
        fact[col] = pd.to_numeric(fact[col], errors="coerce")
    fact = fact.dropna(subset=[TARGET])
    # 조회표의 시간코드(int)와 자료형을 맞춘다. float과 결합하면 매칭이 조용히 실패한다
    fact["시간코드"] = pd.to_numeric(fact["시간코드"], errors="coerce").astype(int)
    return dimension, fact


def build_corridor_and_cbd(dimension: pd.DataFrame) -> pd.DataFrame:
    valid_name = dimension["ROAD_NAME"].notna() & ~dimension["ROAD_NAME"].isin(["", "-", "None"])
    rn = dimension[valid_name].groupby("ROAD_NAME").agg(
        rn_links=("LINK_ID", "size"), rn_len=("LENGTH", "sum"),
        rn_lanes=("LANES", "mean"), rn_maxspd=("MAX_SPD", "mean"),
    ).reset_index()
    corridor = dimension[["LINK_ID", "ROAD_NAME", "LANES", "MAX_SPD", "LENGTH"]].merge(rn, on="ROAD_NAME", how="left")
    corridor["rn_links"] = corridor["rn_links"].fillna(1)
    corridor["rn_len"] = corridor["rn_len"].fillna(corridor["LENGTH"])
    corridor["rn_lanes"] = corridor["rn_lanes"].fillna(corridor["LANES"])
    corridor["rn_maxspd"] = corridor["rn_maxspd"].fillna(corridor["MAX_SPD"])
    corridor = corridor[["LINK_ID"] + CORRIDOR_COLS[:-1]]

    link_centroid = gpd.read_file(INTERIM_DIR / "seoul_link.shp")[["LINK_ID", "geometry"]]
    link_centroid["LINK_ID"] = link_centroid["LINK_ID"].astype(str)
    link_centroid["geometry"] = link_centroid.geometry.centroid
    link_centroid = link_centroid.to_crs("EPSG:5179")
    dist_cbd = pd.DataFrame({
        "LINK_ID": link_centroid["LINK_ID"],
        "dist_cbd": np.sqrt((link_centroid.geometry.x - CBD_X) ** 2 + (link_centroid.geometry.y - CBD_Y) ** 2),
    })
    out = corridor.merge(dist_cbd, on="LINK_ID", how="left")
    for col in CORRIDOR_COLS:
        out[col] = out[col].astype("float32")
    return out[["LINK_ID"] + CORRIDOR_COLS]


def build_turn_features(dimension: pd.DataFrame) -> pd.DataFrame:
    node_info = gpd.read_file(MOCT_NODE_SHP)[["NODE_ID", "NODE_TYPE"]]
    node_info["NODE_ID"] = node_info["NODE_ID"].astype(str)
    node_info["NODE_TYPE"] = pd.to_numeric(node_info["NODE_TYPE"], errors="coerce")
    node_type_lookup = node_info.set_index("NODE_ID")["NODE_TYPE"]

    t_type = node_type_lookup.reindex(dimension["T_NODE"]).to_numpy()
    return pd.DataFrame({
        "LINK_ID": dimension["LINK_ID"],
        "is_cross_t": (t_type == 101).astype("float32"),
    })


def build_link_graph(dimension: pd.DataFrame) -> tuple[dict, dict]:
    """전체 63,548개 링크 위상(F_NODE/T_NODE 인접) 기준 1홉/2홉 이웃."""
    node_links = defaultdict(set)
    for row in dimension[["LINK_ID", "F_NODE", "T_NODE"]].itertuples(index=False):
        node_links[row.F_NODE].add(row.LINK_ID)
        node_links[row.T_NODE].add(row.LINK_ID)
    link_neighbors = defaultdict(set)
    for _, links in node_links.items():
        for l in links:
            link_neighbors[l].update(links - {l})

    link_neighbors_2hop = {}
    for l, one_hop in link_neighbors.items():
        two_hop = set()
        for n in one_hop:
            two_hop |= link_neighbors.get(n, set())
        two_hop -= one_hop
        two_hop.discard(l)
        link_neighbors_2hop[l] = two_hop
    return link_neighbors, link_neighbors_2hop


def build_bucket_lookup(fact_subset: pd.DataFrame, link_ids: np.ndarray) -> pd.DataFrame:
    """(LINK_ID, 시간코드, is_weekend) 평균. 비면 링크×시간 → 링크 → 전체 평균으로 대체(01과 같은 방식)."""
    global_mean = fact_subset[TARGET].mean()
    lut_link = fact_subset.groupby("LINK_ID")[TARGET].mean()
    lut_link_hour = fact_subset.groupby(["LINK_ID", "시간코드"])[TARGET].mean()
    fine = fact_subset.groupby(["LINK_ID", "시간코드", "is_weekend"])[TARGET].mean()

    full_index = pd.MultiIndex.from_tuples(
        list(product(link_ids, BUCKET_HOURS, [0, 1])), names=["LINK_ID", "시간코드", "is_weekend"]
    )
    lookup = fine.reindex(full_index).rename("pred_speed").reset_index()
    hour_fallback = lookup.set_index(["LINK_ID", "시간코드"]).index.map(lut_link_hour)
    lookup["pred_speed"] = lookup["pred_speed"].fillna(pd.Series(hour_fallback, index=lookup.index))
    lookup["pred_speed"] = lookup["pred_speed"].fillna(lookup["LINK_ID"].map(lut_link))
    lookup["pred_speed"] = lookup["pred_speed"].fillna(global_mean)
    lookup["pred_speed"] = lookup["pred_speed"].astype("float32")
    return lookup


def build_neighbor_bucket_feature(bucket_lookup: pd.DataFrame, link_neighbors: dict, link_neighbors_2hop: dict,
                                  all_link_ids: np.ndarray, allowed_links: set) -> pd.DataFrame:
    """링크×시간코드×평일/주말 이웃속도를 만든다.

    all_link_ids는 값을 만들어 줄 대상(평가·미관측 링크 포함 전부), allowed_links는 이웃 소스로 인정할
    링크(학습 또는 관측 링크)다. 둘을 같게 넘기면 평가·미관측 링크가 결과에서 빠지므로 반드시 구분한다.
    """
    link_ids_sorted = np.sort(np.array(list(all_link_ids)))
    link_idx = {l: i for i, l in enumerate(link_ids_sorted)}
    n_links = len(link_ids_sorted)

    def adjacency(neighbor_map: dict) -> csr_matrix:
        rows, cols = [], []
        for l in link_ids_sorted:
            i = link_idx[l]
            for n in neighbor_map.get(l, ()):
                if n in allowed_links:
                    rows.append(i)
                    cols.append(link_idx[n])
        return csr_matrix((np.ones(len(rows), dtype="float32"), (rows, cols)), shape=(n_links, n_links))

    A1 = adjacency(link_neighbors)
    A2 = adjacency(link_neighbors_2hop)

    bucket_cols = [f"{h}_{w}" for h in BUCKET_HOURS for w in (0, 1)]
    wide = bucket_lookup.copy()
    wide["bucket"] = wide["시간코드"].astype(str) + "_" + wide["is_weekend"].astype(str)
    wide = wide.pivot(index="LINK_ID", columns="bucket", values="pred_speed").reindex(link_ids_sorted)[bucket_cols]
    speed_mat = wide.to_numpy(dtype="float32")

    deg1 = np.asarray(A1.sum(axis=1)).ravel()
    deg2 = np.asarray(A2.sum(axis=1)).ravel()
    with np.errstate(invalid="ignore", divide="ignore"):
        mean1 = (A1 @ speed_mat) / deg1[:, None]
        mean2 = (A2 @ speed_mat) / deg2[:, None]
    mean1[deg1 == 0] = np.nan
    mean2[deg2 == 0] = np.nan
    fallback = np.where(np.isnan(mean1), mean2, mean1)

    out = pd.DataFrame(fallback, index=link_ids_sorted, columns=bucket_cols)
    out = out.reset_index().rename(columns={"index": "LINK_ID"})
    long = out.melt(id_vars="LINK_ID", var_name="bucket", value_name=NEIGHBOR_COL)
    split = long["bucket"].str.split("_", expand=True)
    long["시간코드"] = split[0].astype(int)
    long["is_weekend"] = split[1].astype(int)
    return long[["LINK_ID", "시간코드", "is_weekend", NEIGHBOR_COL]]


def assemble_features(dimension: pd.DataFrame, fact: pd.DataFrame) -> pd.DataFrame:
    df = fact.merge(dimension.drop(columns=["ROAD_NAME"]), on="LINK_ID", how="left")

    date = pd.to_datetime(df["기준일"], format="%Y%m%d")
    df["dow"] = date.dt.weekday.astype("category")
    df["is_weekend"] = (date.dt.weekday >= 5).astype("int8")
    hour_code = df["시간코드"].astype("float32")
    df["hour_sin"] = np.sin(2 * np.pi * hour_code / 24).astype("float32")
    df["hour_cos"] = np.cos(2 * np.pi * hour_code / 24).astype("float32")

    actual_hour = (df["시간코드"].astype(int) - 1).clip(0, 23).to_numpy()
    hour_mat = dimension.set_index("LINK_ID")[[f"avg_speed_h{h}" for h in range(24)]]
    hour_mat = hour_mat.reindex(df["LINK_ID"]).to_numpy(dtype="float32")
    df["avg_speed_hourly"] = hour_mat[np.arange(len(df)), actual_hour]

    corridor = build_corridor_and_cbd(dimension)
    df = df.merge(corridor, on="LINK_ID", how="left")
    turn = build_turn_features(dimension)
    df = df.merge(turn, on="LINK_ID", how="left")

    for col in CATEGORICAL_COLS:
        df[col] = df[col].astype("category")
    return df


# ---------------------------------------------------------------- 학습/평가 ----

def make_and_fit(X: pd.DataFrame, y: pd.Series, sample_weight: np.ndarray) -> xgb.XGBRegressor:
    model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=300, learning_rate=0.05,
                             max_depth=8, tree_method="hist", enable_categorical=True,
                             random_state=RANDOM_STATE, verbosity=0)
    model.fit(X, y, sample_weight=sample_weight)
    return model


def evaluate(y_true, pred) -> dict:
    return {"rmse": mean_squared_error(y_true, pred) ** 0.5,
            "mae": mean_absolute_error(y_true, pred), "r2": r2_score(y_true, pred)}


def eval_bucket(y_true, pred, mask) -> dict:
    if mask.sum() < 30:
        return {"n": int(mask.sum()), "rmse": float("nan"), "mae": float("nan"), "r2": float("nan")}
    return {"n": int(mask.sum()), "rmse": mean_squared_error(y_true[mask], pred[mask]) ** 0.5,
            "mae": mean_absolute_error(y_true[mask], pred[mask]), "r2": r2_score(y_true[mask], pred[mask])}


def run_validate(df: pd.DataFrame, link_neighbors: dict, link_neighbors_2hop: dict, all_link_ids: np.ndarray) -> None:
    groups = df["LINK_ID"].values
    gkf = GroupKFold(n_splits=N_FOLDS)
    rows = []

    for fold_i, (train_idx, test_idx) in enumerate(gkf.split(df, groups=groups)):
        train_links = set(df["LINK_ID"].iloc[train_idx].unique())
        test_links = set(df["LINK_ID"].iloc[test_idx].unique())
        assert len(train_links & test_links) == 0

        bucket_lookup = build_bucket_lookup(df.iloc[train_idx][["LINK_ID", "시간코드", "is_weekend", TARGET]], np.sort(np.array(list(train_links))))
        neighbor_long = build_neighbor_bucket_feature(bucket_lookup, link_neighbors, link_neighbors_2hop, all_link_ids, train_links)
        merged = df[["LINK_ID", "시간코드", "is_weekend"]].merge(neighbor_long, on=["LINK_ID", "시간코드", "is_weekend"], how="left")

        Xf = df[STATIC_COLS].copy()
        Xf[NEIGHBOR_COL] = merged[NEIGHBOR_COL].to_numpy(dtype="float32")
        yf = df[TARGET]
        w = df["sample_count_5min"].to_numpy(dtype="float32")

        X_train, X_test = Xf.iloc[train_idx], Xf.iloc[test_idx]
        y_train, y_test = yf.iloc[train_idx], yf.iloc[test_idx]
        w_train = w[train_idx]

        model_nb = make_and_fit(X_train, y_train, w_train)
        pred_nb = model_nb.predict(X_test)
        model_stat = make_and_fit(X_train[STATIC_COLS], y_train, w_train)
        pred_stat = model_stat.predict(X_test[STATIC_COLS])

        has_nb = X_test[NEIGHBOR_COL].notna().to_numpy()
        pred_split = np.where(has_nb, pred_nb, pred_stat)
        y_test_arr = y_test.to_numpy()

        r_single = evaluate(y_test_arr, pred_nb)
        r_split = evaluate(y_test_arr, pred_split)
        print(f"[fold{fold_i}] 단일모델 R2={r_single['r2']:.3f} MAE={r_single['mae']:.3f} | "
              f"분리모델 R2={r_split['r2']:.3f} MAE={r_split['mae']:.3f} | 이웃있음(B) 비율={has_nb.mean():.1%}")
        b_split = eval_bucket(y_test_arr, pred_split, has_nb)
        c_split = eval_bucket(y_test_arr, pred_split, ~has_nb)
        c_single = eval_bucket(y_test_arr, pred_nb, ~has_nb)
        print(f"          B(분리) R2={b_split['r2']:.3f} n={b_split['n']:,} | "
              f"C 단일={c_single['r2']:.3f} -> 분리={c_split['r2']:.3f} n={c_split['n']:,}")
        rows.append({"fold": fold_i, **{f"single_{k}": v for k, v in r_single.items()},
                     **{f"split_{k}": v for k, v in r_split.items()}})
        del bucket_lookup, neighbor_long, merged, Xf, X_train, X_test
        gc.collect()

    summary = pd.DataFrame(rows)
    print("\n=== 3-fold 평균 ===")
    print(summary[["split_rmse", "split_mae", "split_r2"]].mean().round(4).to_string())


def run_final(df: pd.DataFrame, link_neighbors: dict, link_neighbors_2hop: dict, all_link_ids: np.ndarray) -> None:
    model_a = pd.read_parquet(MODEL_A_PATH)
    observed_links = set(model_a["LINK_ID"].unique())
    bucket_lookup = model_a[["LINK_ID", "시간코드", "is_weekend", "pred_speed"]]

    neighbor_long = build_neighbor_bucket_feature(bucket_lookup, link_neighbors, link_neighbors_2hop, all_link_ids, observed_links)
    neighbor_long.to_parquet(PROCESSED_DIR / "neighbor_speed_lookup.parquet", index=False)
    print("이웃속도 lookup 저장:", PROCESSED_DIR / "neighbor_speed_lookup.parquet", "| 행 수:", len(neighbor_long))
    print(f"이웃 있는 링크(B) 비율: {neighbor_long.groupby('LINK_ID')[NEIGHBOR_COL].apply(lambda s: s.notna().any()).mean():.1%}")

    merged = df[["LINK_ID", "시간코드", "is_weekend"]].merge(
        neighbor_long, on=["LINK_ID", "시간코드", "is_weekend"], how="left")
    Xf = df[STATIC_COLS].copy()
    Xf[NEIGHBOR_COL] = merged[NEIGHBOR_COL].to_numpy(dtype="float32")
    yf = df[TARGET]
    w = df["sample_count_5min"].to_numpy(dtype="float32")

    print(f"\n전체데이터 학습: {len(Xf):,}행")
    model_nb = make_and_fit(Xf, yf, w)
    model_stat = make_and_fit(Xf[STATIC_COLS], yf, w)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODELS_DIR / "model_nb.pkl", "wb") as f:
        pickle.dump({"model": model_nb, "feature_cols": FEATURE_COLS, "categorical_cols": CATEGORICAL_COLS}, f)
    with open(MODELS_DIR / "model_stat.pkl", "wb") as f:
        pickle.dump({"model": model_stat, "feature_cols": STATIC_COLS, "categorical_cols": CATEGORICAL_COLS}, f)
    print("저장:", MODELS_DIR / "model_nb.pkl", "/", MODELS_DIR / "model_stat.pkl")


def main() -> None:
    parser = argparse.ArgumentParser(description="Model B·C 학습과 검증")
    parser.add_argument("--mode", choices=["validate", "final"], default="final",
                        help="validate: 도로 단위 GroupKFold 검증, final: 전체 학습과 저장")
    args = parser.parse_args()

    print("데이터 로드 중...")
    dimension, fact = load_dimension_and_fact()
    df = assemble_features(dimension, fact)
    print(f"행 수: {len(df):,} | 관측 링크: {df['LINK_ID'].nunique():,}")

    link_neighbors, link_neighbors_2hop = build_link_graph(dimension)
    all_link_ids = dimension["LINK_ID"].to_numpy()

    if args.mode == "validate":
        run_validate(df, link_neighbors, link_neighbors_2hop, all_link_ids)
    else:
        run_final(df, link_neighbors, link_neighbors_2hop, all_link_ids)


if __name__ == "__main__":
    main()
