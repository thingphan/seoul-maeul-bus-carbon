"""노선 유형 구분과 승하차 데이터 연결 규칙(00·01·02·06 공통).

마을버스는 이름 규칙(예: '한글+두 자리 숫자') 대신 승하차 데이터(OA-12913)의 공식 교통수단 분류
'서울마을버스'로 고른다. 이름 규칙으로는 강남06-1, 은평08-2 같은 파생 노선과 서대문02대·02소 같은
분기 노선이 빠진다.

정류소 파일(OA-1095)은 서대문02·09를 '대'(큰 순환)·'소'(작은 순환) 두 노선으로 나누지만, 승하차
데이터는 '서대문02' 하나로 집계한다. 그래서 승하차를 찾을 때는 끝의 '대'·'소'를 뗀 이름을 쓰고,
두 노선이 함께 서는 정류장의 승하차는 TOPIS 정류장별 운행횟수(OA-21220) 비율로 나눈다.

자율주행 노선은 이름에 'A+숫자'(예: 동작A01)가 들어간다.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from route_data import RIDERSHIP_CSV, load_trip_counts

VILLAGE_TYPE = "서울마을버스"
BRANCH_SUFFIXES = ("대", "소")


@lru_cache(maxsize=1)
def village_route_numbers() -> frozenset[str]:
    ride = pd.read_csv(RIDERSHIP_CSV, encoding="cp949", usecols=["노선번호", "교통수단타입명"], dtype=str)
    return frozenset(ride.loc[ride["교통수단타입명"] == VILLAGE_TYPE, "노선번호"].str.strip())


def ridership_key(name: str) -> str:
    """정류소 파일의 노선명 → 승하차 데이터의 노선번호(분기 노선은 끝의 '대'·'소'를 뗀다)."""
    name = str(name).strip()
    villages = village_route_numbers()
    if name not in villages and name.endswith(BRANCH_SUFFIXES) and name[:-1] in villages:
        return name[:-1]
    return name


def classify_route(name: pd.Series) -> pd.Series:
    name = name.astype(str).str.strip()
    is_village = name.map(ridership_key).isin(village_route_numbers())
    return pd.Series(
        np.select([name.str.contains(r"A\d", regex=True), is_village], ["자율주행", "마을버스"], "일반"),
        index=name.index,
    )


def branch_stop_weights(route_stops: pd.DataFrame) -> pd.Series:
    """(ROUTE_ID, NODE_ID)별 승하차 배분 비율. 분기 노선이 함께 서는 정류장만 1 미만이 된다.

    route_stops에는 ROUTE_ID, 노선명, NODE_ID 열이 있어야 한다. 같은 승하차 노선번호를 나눠 쓰는
    노선들 사이에서, 정류장별 평균 운행횟수 비율로 나눈다. 운행횟수가 없으면 똑같이 나눈다.
    """
    frame = route_stops[["ROUTE_ID", "노선명", "NODE_ID"]].drop_duplicates(["ROUTE_ID", "NODE_ID"]).copy()
    frame["key"] = frame["노선명"].map(ridership_key)
    shared = frame.groupby("key")["ROUTE_ID"].transform("nunique") > 1
    frame["weight"] = 1.0
    if shared.any():
        trips = load_trip_counts()
        trips = trips[trips["노선_ID"].isin(set(frame.loc[shared, "ROUTE_ID"]))]
        stop_trips = (trips.groupby(["노선_ID", "정류장_ID"])["버스운행횟수"].mean()
                      .rename("trips").rename_axis(["ROUTE_ID", "NODE_ID"]))
        sub = frame.loc[shared].join(stop_trips, on=["ROUTE_ID", "NODE_ID"])
        group_sum = sub.groupby(["key", "NODE_ID"])["trips"].transform("sum")
        group_n = sub.groupby(["key", "NODE_ID"])["ROUTE_ID"].transform("nunique")
        sub["weight"] = np.where(group_sum > 0, sub["trips"].fillna(0) / group_sum, 1.0 / group_n)
        frame.loc[shared, "weight"] = sub["weight"]
    return frame.set_index(["ROUTE_ID", "NODE_ID"])["weight"]
