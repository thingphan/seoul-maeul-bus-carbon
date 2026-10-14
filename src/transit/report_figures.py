"""마을버스 분석 보고서용 그림(PNG)을 만든다. 01~10 산출물만 읽는다.

디자인: 그림마다 결론을 말하는 제목·설명·출처를 넣고, 강조할 대상만 색을 쓰며 범례 대신 직접 라벨을 단다.
글꼴은 Pretendard(--font-dir 또는 REPORT_FONT_DIR 폴더의 .otf/.ttf)를 쓰고, 없으면 맑은 고딕을 쓴다.

출력: --out 폴더(기본 docs/report/images, 보고서 본문 docs/report/maeul_bus_report.md가 쓰는 위치)에 PNG.
      fig00 한눈에 보기, fig01~fig07 본문 그림, figA1~figA5 부록 그림
사용: python src/transit/report_figures.py [--out 폴더] [--font-dir Pretendard 폴더]
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import geopandas as gpd
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.ticker import NullLocator
from shapely.geometry import LineString, box

from route_data import load_route_stops

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRANSIT = PROJECT_ROOT / "data" / "processed" / "transit"
ROUTE_PATH_CSV = PROJECT_ROOT / "data" / "raw" / "traffic" / "bus_route_path" / "route_paths.csv"
BOUNDARY = PROJECT_ROOT / "data" / "raw" / "boundary" / "seoul_admin_boundary.geojson"

# 색: 글자는 무채색, 데이터는 강조할 대상만 파랑, 나머지는 옅은 회색(색각 이상에도 구분되는 조합).
INK, INK2, MUTED = "#111827", "#4b5563", "#9ca3af"
GRID, LINE, CARD, BORDER = "#eef0f3", "#d1d5db", "#f5f6f8", "#e5e7eb"
BLUE, BLUE_DARK, BLUE_LIGHT, BLUE_PALE = "#2a78d6", "#184f95", "#86b6ef", "#e8f1fc"
DIM = "#cbd5e1"
CLASS_COLORS = {1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a", 4: "#9ca3af"}
CLASS_NAMES = {1: "① 증편·지원 우선", 2: "② 소형 차량 전환", 3: "③ 현행 유지", 4: "④ 노선 조정 검토"}
SOURCE = "자료: 서울 열린데이터광장, TOPIS, 국가교통정보센터, 서울 생활인구 · 분석: 본 연구"
DPI = 220
W = 9.0  # 그림 기본 너비(인치)


# ---------------------------------------------------------------- 공통 ----

def setup_style(font_dir: Path | None) -> None:
    for folder in filter(None, [font_dir, os.environ.get("REPORT_FONT_DIR")]):
        for path in Path(folder).glob("*.[ot]tf"):
            fm.fontManager.addfont(str(path))
    family = "Pretendard" if any(f.name == "Pretendard" for f in fm.fontManager.ttflist) else "Malgun Gothic"
    print(f"  글꼴: {family}")
    plt.rcParams.update({
        "font.family": family, "axes.unicode_minus": False, "font.size": 9.5,
        "axes.edgecolor": LINE, "axes.labelcolor": INK2, "axes.linewidth": 0.8, "axes.labelsize": 9,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white", "text.color": INK,
    })


def canvas(height: float, title: str, subtitle: str = "", source: str = SOURCE, width: float = W):
    """제목·설명·출처를 둔 그림. 본문 영역 (left, bottom, right, top)을 그림 비율로 돌려준다."""
    fig = plt.figure(figsize=(width, height))
    left = 0.34 / width
    fig.text(left, 1 - 0.26 / height, title, fontsize=15, fontweight="bold", color=INK, va="top")
    used = 0.26 + 0.32
    if subtitle:
        fig.text(left, 1 - used / height, subtitle, fontsize=9.8, color=INK2, va="top", linespacing=1.45)
        used += 0.24 * (subtitle.count("\n") + 1) + 0.06
    if source:
        fig.text(left, 0.14 / height, source, fontsize=7.8, color=MUTED, va="bottom")
    top = 1 - (used + 0.10) / height
    bottom = (0.42 if source else 0.16) / height
    return fig, (left, bottom, 1 - left, top)


def axes_in(fig, rect, pad=(0.0, 0.0, 0.0, 0.0)):
    """본문 영역 안에 축을 둔다. pad = (왼쪽, 아래, 오른쪽, 위) 여백(인치)."""
    left, bottom, right, top = rect
    w, h = fig.get_size_inches()
    return fig.add_axes([left + pad[0] / w, bottom + pad[1] / h,
                         right - left - (pad[0] + pad[2]) / w, top - bottom - (pad[1] + pad[3]) / h])


def clean(ax, grid: str | None = "y") -> None:
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(LINE)
    if grid:
        ax.grid(axis=grid, color=GRID, linewidth=0.9)
        ax.set_axisbelow(True)
    ax.tick_params(length=0, pad=5)


def hbar(ax, y: float, value: float, height: float, color: str, r_px: float = 5.0, left: float = 0.0) -> None:
    """가로 막대: 값 쪽 끝만 둥글게, 기준선 쪽은 각지게."""
    ax.figure.canvas.draw()
    bbox = ax.get_window_extent()
    rx = r_px / (bbox.width / np.ptp(ax.get_xlim()))
    ry = r_px / (bbox.height / np.ptp(ax.get_ylim()))
    ax.add_patch(FancyBboxPatch((left, y - height / 2), value, height, boxstyle=f"round,pad=0,rounding_size={rx}",
                                mutation_aspect=ry / rx, facecolor=color, edgecolor="none"))
    ax.add_patch(Rectangle((left, y - height / 2), min(2 * rx, value), height, facecolor=color, edgecolor="none"))


def vbar(ax, x: float, value: float, width: float, color: str, r_px: float = 4.0) -> None:
    ax.figure.canvas.draw()
    bbox = ax.get_window_extent()
    rx = r_px / (bbox.width / np.ptp(ax.get_xlim()))
    ry = r_px / (bbox.height / np.ptp(ax.get_ylim()))
    ax.add_patch(FancyBboxPatch((x - width / 2, 0), width, value, boxstyle=f"round,pad=0,rounding_size={rx}",
                                mutation_aspect=ry / rx, facecolor=color, edgecolor="none"))
    ax.add_patch(Rectangle((x - width / 2, 0), width, min(2 * ry, value), facecolor=color, edgecolor="none"))


def card(ax, x, y, w, h, face=CARD, edge="none", radius=0.03, lw=1.0, transform=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}", facecolor=face,
                                edgecolor=edge, linewidth=lw, transform=transform or ax.transAxes))


def pill(ax, x, y, text, face, color="white", size=8.5, transform=None):
    ax.text(x, y, text, fontsize=size, fontweight="bold", color=color, va="center", ha="left",
            transform=transform or ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.35,rounding_size=0.8", facecolor=face, edgecolor="none"))


def save(fig, out: Path, name: str) -> None:
    fig.savefig(out / name, dpi=DPI, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print(f"  {name}")


def load_tables() -> dict[str, pd.DataFrame]:
    read = lambda name: pd.read_csv(TRANSIT / name, dtype={"ROUTE_ID": str})  # noqa: E731
    carbon = read("route_carbon_breakeven.csv")
    return {
        "carbon": carbon[~carbon["load_qa_flag"]].copy(),
        "priority": read("route_priority.csv"),
        "demand": read("route_demand.csv"),
        "hourly": read("route_hourly_load.csv"),
        "ops": read("route_operations.csv"),
        "speed": pd.read_csv(TRANSIT / "ai_speed_validation.csv"),
        "feas": read("route_feasibility.csv"),
        "removal": read("route_removal_access.csv"),
        "removal_cells": read("route_removal_cells.csv"),
    }


def tier_shares() -> tuple[pd.Series, pd.Series]:
    """(마을버스 노선 길이 기준 등급 비중 %, 서울 전체 링크 수 기준 등급 비중 %)."""
    links = pd.read_csv(TRANSIT / "route_links.csv", dtype={"ROUTE_ID": str, "LINK_ID": str})
    links = links[links["kind"] == "마을버스"]
    tier = pd.read_parquet(TRANSIT / "link_speed_weekday_profile.parquet", columns=["LINK_ID", "tier"]).drop_duplicates("LINK_ID")
    merged = links.merge(tier, on="LINK_ID")
    route_share = merged.groupby("tier")["LENGTH"].sum() / merged["LENGTH"].sum() * 100
    return route_share, tier["tier"].value_counts(normalize=True) * 100


def reduction_numbers(t: dict) -> dict[str, float]:
    c, p, o, h = t["carbon"], t["priority"], t["ops"], t["hourly"]
    return {
        "electric": (c["bus_co2_kg_per_day"].sum() - c["bus_co2_kg_per_day_all_electric"].sum()) / 1000,
        "offpeak": o["van_offpeak_cut_vs_current_kg_per_day"].sum() / 1000,
        "offpeak_size": o["van_offpeak_cut_kg_per_day"].sum() / 1000,
        "class2": p["van_co2_cut_vs_current_kg_per_day"].sum() / 1000,
        "class2_size": p["van_co2_cut_kg_per_day"].sum() / 1000,
        "low_share": h.loc[h["low_load"], "trips"].sum() / h["trips"].sum(),
        "class2_routes": int(p.loc[p["class_id"] == 2, "van_feasible"].sum()),
    }


def year_t(day_t: float) -> str:
    year = day_t * 365
    return f"연 약 {round(year, -2 if year >= 1000 else -1):,.0f}톤"


# ---------------------------------------------------------------- 그림 ----

def fig00_key_numbers(t: dict, out: Path) -> None:
    """한눈에 보기: 보고서 첫머리의 숫자 4개."""
    c, r = t["carbon"], t["removal"]
    red = reduction_numbers(t)
    high = r.loc[r["class_id"].isin([1, 2]), "net_elderly_affected"].median()
    low = r.loc[r["class_id"].isin([3, 4]), "net_elderly_affected"].median()
    ai = 100 - tier_shares()[0]["A_lookup"]
    tiles = [
        ("탄소", f"{int((c['bus_vs_car'] < 1).sum())}", f"/{len(c)}",
         f"노선에서 승객 1명당 탄소가\n승용차보다 적다({c['bus_g_per_pkm'].median():.0f}g 대 {c['car_g_per_pkm'].median():.0f}g)"),
        ("차 크기 · 시간", f"{red['low_share']:.0%}", "",
         f"차 안 3명 미만인 운행만 작은 전기차로\n바꿔도 전기화 효과의 절반 가까이\n({year_t(red['offpeak'])})"),
        ("어르신", f"{high / low:.0f}배", "",
         "의존 높은 노선을 없앨 때 더 걷게\n되는 어르신(의존 낮은 노선 대비)"),
        ("AI", f"{ai:.0f}%", "", "AI가 채운 골목길 구간(95%가 1차로)\nAI 오차를 감안해도 결론 유지"),
    ]
    fig = plt.figure(figsize=(11, 2.25))
    for i, (tag, big, small, label) in enumerate(tiles):
        ax = fig.add_axes([0.005 + i * 0.25, 0.03, 0.24, 0.94])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        card(ax, 0.0, 0.0, 1.0, 1.0, face="white", edge=BORDER, radius=0.06, lw=1.2)
        ax.scatter([0.105], [0.84], s=28, color=BLUE)
        ax.text(0.15, 0.84, tag, fontsize=9, fontweight="semibold", color=INK2, va="center")
        num = ax.text(0.08, 0.66, big, fontsize=30, fontweight="heavy", color=INK, va="top")
        if small:
            fig.canvas.draw()
            bb = num.get_window_extent().transformed(ax.transAxes.inverted())
            ax.text(bb.x1 + 0.01, bb.y0 + 0.03, small, fontsize=15, fontweight="bold", color=MUTED, va="bottom")
        ax.text(0.08, 0.36, label, fontsize=9, color=INK2, va="top", linespacing=1.5)
    save(fig, out, "fig00_key_numbers.png")


def fig01_flow(out: Path) -> None:
    fig, rect = canvas(5.3, "분석 흐름: 데이터 → AI·물리 모델 → 분석 → 결과·정책",
                       "관측이 없는 골목길 속도는 AI가 채우고, 같은 길·속도·경사로 버스와 승용차의 탄소를 계산했다", source="")
    ax = axes_in(fig, rect)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 60)
    ax.axis("off")
    steps = [
        ("01", "데이터", [("버스 경로·정류장", "서울시 버스 API"), ("승하차·운행횟수", "교통카드, TOPIS"),
                         ("생활인구·자동차 등록", "250m 격자, 행정동"), ("도로 속도 + 예측 자료", "강수·교통량·시설 등 13종")]),
        ("02", "AI·물리 모델", [("AI 속도 추정", "관측 없는 도로 29%를\nXGBoost로 채움"),
                              ("차량별 에너지 모델", "같은 길·속도·경사로\n버스 3종, 승용차 5종")]),
        ("03", "분석", [("승객 1명·1km당 탄소", "버스 대 승용차, 손익분기"), ("시간대·차 크기별 감축", "언제, 어떤 차를 보낼까"),
                       ("어르신 의존도", "노선을 없앨 때 보행 피해")]),
        ("04", "결과·정책", [("네 유형 분류", "탄소 효율 × 의존도"), ("노선 카드", "노선별 처방"), ("정책 제안", "저상 소형차·DRT, 평가 기준")]),
    ]
    col_w, gap = 22.6, 3.2
    for k, (num, head, items) in enumerate(steps):
        x = k * (col_w + gap)
        card(ax, x, 12, col_w, 46, face=CARD, radius=1.6, transform=ax.transData)
        ax.text(x + 1.8, 54.2, num, fontsize=10, fontweight="heavy", color=BLUE, va="center")
        ax.text(x + 6.0, 54.2, head, fontsize=10.5, fontweight="bold", color=INK, va="center")
        slot = 40.0 / len(items)
        for i, (title, sub) in enumerate(items):
            top = 50 - i * slot
            if i:
                ax.plot([x + 1.8, x + col_w - 1.8], [top + 0.8, top + 0.8], color=BORDER, lw=1)
            is_ai = title == "AI 속도 추정"
            ax.text(x + 1.8, top - 1.4, title, fontsize=9.6, fontweight="bold", color=BLUE_DARK if is_ai else INK, va="top")
            ax.text(x + 1.8, top - 5.4, sub, fontsize=8.6, color=INK2, va="top", linespacing=1.4)
            if is_ai:
                pill(ax, x + 15.4, top - 2.6, "AI", BLUE, size=7.5, transform=ax.transData)
        if k < 3:
            ax.annotate("", xy=(x + col_w + gap - 0.6, 35), xytext=(x + col_w + 0.6, 35),
                        arrowprops=dict(arrowstyle="-|>,head_width=0.35,head_length=0.6", color=MUTED, lw=1.4))
    card(ax, 25.8, 0, 74.2, 9, face=BLUE_PALE, radius=1.6, transform=ax.transData)
    ax.text(27.6, 4.5, "검증", fontsize=9.6, fontweight="bold", color=BLUE_DARK, va="center")
    ax.text(33.0, 4.5, "AI 정확도(도로 단위 분할) · 실제 마을버스 속도 비교 · AI 오차·탑승 인원·버스 크기·정류장 거리를 바꿔도 결론 유지",
            fontsize=8.5, color=INK2, va="center")
    save(fig, out, "fig01_flow.png")
# 검증 성능: A는 notebooks/03_model_a_history.ipynb(시간 순서 분할, docs/SPEED_MODEL.md). B·C는 src/models/02_build_model_bc.py
# --mode validate(도로 단위 3-fold GroupKFold)를 등급별로 집계한 값(src/transit/checks/validate_bc_tiers.py, 전체 R² 0.7226).
TIERS = [
    ("A_lookup", "A 관측 이력", "과거 같은 시간대 평균", "R² 0.92 · 평균 오차 2.3km/h", DIM),
    ("B_neighbor_model", "B AI 추정", "이웃 도로 관측 있음", "R² 0.72 · 평균 오차 4.6km/h", BLUE),
    ("C_static_model", "C AI 추정", "이웃 관측도 없음", "R² 0.31 · 평균 오차 5.7km/h", BLUE_DARK),
]


def fig02_ai_tiers(out: Path) -> None:
    route_share, city_share = tier_shares()
    fig, rect = canvas(3.9, "마을버스 노선의 13%는 AI가 채운 골목길",
                       "속도 정보 등급별 비중. AI가 채운 마을버스 구간의 95%가 1차로 골목이다")
    ax = axes_in(fig, rect, pad=(1.75, 0.95, 0.15, 0.05))
    ax.set_xlim(0, 100)
    ax.set_ylim(1.6, -0.6)
    ax.axis("off")
    for row, (label, shares) in enumerate((("서울 전체 도로", city_share), ("마을버스 노선", route_share))):
        ax.text(-2, row, label, ha="right", va="center", fontsize=10, fontweight="bold" if row else "medium",
                color=INK if row else INK2)
        left = 0.0
        for key, _, _, _, color in TIERS:
            value = shares[key]
            ax.add_patch(Rectangle((left, row - 0.28), value - 0.4, 0.56, facecolor=color, edgecolor="none"))
            if value > 6:
                ax.text(left + value / 2, row, f"{value:.1f}%", ha="center", va="center", fontsize=9.5, fontweight="bold",
                        color=INK if color == DIM else "white")
            left += value
        tail = sum(shares[k] for k, *_ in TIERS[1:])
        ax.text(101, row, f"AI {tail:.0f}%", ha="left", va="center", fontsize=9.5, fontweight="bold", color=BLUE_DARK)
    legend = fig.add_axes([rect[0] + 1.75 / W, rect[1], rect[2] - rect[0] - 1.9 / W, 0.62 / 3.9])
    legend.axis("off")
    for i, (_, name, desc, metric, color) in enumerate(TIERS):
        x = i / 3
        legend.add_patch(Rectangle((x, 0.62), 0.018, 0.22, facecolor=color, transform=legend.transAxes))
        legend.text(x + 0.03, 0.73, f"{name} · {desc}", fontsize=8.8, fontweight="bold", color=INK, va="center")
        legend.text(x + 0.03, 0.30, metric, fontsize=8.5, color=INK2, va="center")
    save(fig, out, "fig02_ai_tiers.png")


def figA1_co2_per_km(t: dict, out: Path) -> None:
    c, p = t["carbon"], t["priority"]
    cars = [("현재 구성", c["car_g_per_km"].median(), True), ("휘발유", c["car_휘발유_g_per_km"].median(), False),
            ("경유", c["car_경유_g_per_km"].median(), False), ("LPG", c["car_LPG_g_per_km"].median(), False),
            ("하이브리드", c["car_하이브리드_g_per_km"].median(), False), ("전기", c["car_전기_g_per_km"].median(), False)]
    buses = [("현재 구성", c["bus_g_per_km"].median(), True), ("경유", c["bus_diesel_g_per_km"].median(), False),
             ("CNG", c["bus_cng_g_per_km"].median(), False), ("전기", c["bus_ev_g_per_km"].median(), False),
             ("소형 전기 승합차", p["van_g_per_km"].median(), False)]
    fig, rect = canvas(4.1, "차 1대로만 보면 마을버스가 승용차보다 더 낸다",
                       "1km 주행당 CO₂(g, 230개 노선 중앙값). '현재 구성'은 서울의 실제 차량 구성 가중")
    half = (rect[2] - rect[0]) / 2
    for k, (title, rows) in enumerate((("승용차 1대", cars), ("마을버스 1대", buses))):
        sub = (rect[0] + k * half, rect[1], rect[0] + (k + 1) * half, rect[3])
        ax = axes_in(fig, sub, pad=(1.15, 0.05, 0.35, 0.35))
        ax.set_xlim(0, 620)
        ax.set_ylim(len(cars) - 0.4, -0.6)
        ax.axis("off")
        ax.text(-150, -0.95, title, fontsize=10.5, fontweight="bold", color=INK, va="bottom")
        for i, (label, value, highlight) in enumerate(rows):
            ax.text(-18, i, label, ha="right", va="center", fontsize=9.2,
                    fontweight="bold" if highlight else "regular", color=INK if highlight else INK2)
            hbar(ax, i, value, 0.56, BLUE if highlight else DIM)
            ax.text(value + 12, i, f"{value:,.0f}g", va="center", fontsize=9.2, color=INK,
                    fontweight="bold" if highlight else "regular")
    save(fig, out, "figA1_co2_per_km.png")


def fig03_load_scatter(t: dict, out: Path) -> None:
    c = t["carbon"]
    worse = c["bus_vs_car"] >= 1
    be = c["breakeven_load"].median()
    med = c["avg_load"].median()
    fig, rect = canvas(5.0, f"차 안에 평균 {be:.2f}명만 있어도 승용차와 같다. 실제는 {med:.1f}명",
                       "노선별 차 안 평균 인원과 승객 1명·1km당 CO₂(230개 노선, 세로축 로그 눈금)")
    ax = axes_in(fig, rect, pad=(0.55, 0.45, 0.1, 0.0))
    clean(ax, "y")
    ax.set_yscale("log")
    ax.set_xlim(0, 18)
    ax.set_ylim(20, 1500)
    bus_km, car_pkm = c["bus_g_per_km"].median(), c["car_g_per_pkm"].median()
    x = np.linspace(0.3, 18, 300)
    ax.plot(x, bus_km / x, color=LINE, lw=1.6, zorder=1)
    ax.axhline(car_pkm, color=INK2, lw=1.1, zorder=1)
    ax.scatter(c.loc[~worse, "avg_load"], c.loc[~worse, "bus_g_per_pkm"], s=26, color=BLUE, alpha=0.55,
               edgecolor="white", linewidth=0.6, zorder=3)
    ax.scatter(c.loc[worse, "avg_load"], c.loc[worse, "bus_g_per_pkm"], s=34, color=CLASS_COLORS[2],
               edgecolor="white", linewidth=0.7, zorder=4)
    ax.scatter([be], [car_pkm], s=70, facecolor="white", edgecolor=INK, linewidth=1.6, zorder=5)
    ax.annotate(f"손익분기 {be:.2f}명", (be, car_pkm), xytext=(2.6, car_pkm * 2.6), fontsize=9.5, fontweight="bold",
                color=INK, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.9))
    ax.text(17.9, car_pkm * 1.07, f"승용차 승객 1명 {car_pkm:.0f}g", ha="right", va="bottom", fontsize=9, color=INK2)
    ax.axvline(med, color=BLUE, lw=1.0, alpha=0.5, zorder=1)
    ax.text(med + 0.25, 1300, f"실제 중앙 {med:.1f}명", fontsize=9, fontweight="bold", color=BLUE_DARK, va="top")
    ax.text(17.9, 150, f"승용차보다 적은 노선 {int((~worse).sum())}개", ha="right", fontsize=9.5, fontweight="bold",
            color=BLUE_DARK)
    ax.text(0.9, 1300, f"승용차보다 많은 노선 {int(worse.sum())}개", fontsize=9.5, fontweight="bold", color=CLASS_COLORS[2], va="top")
    ax.text(17.9, 46, "곡선: 버스 1km 탄소 ÷ 차 안 인원", fontsize=8.5, color=MUTED, ha="right", va="bottom")
    ax.set_yticks([50, 100, 200, 500, 1000], ["50g", "100g", "200g", "500g", "1,000g"])
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_xticks(range(0, 19, 3), [f"{v}명" for v in range(0, 19, 3)])
    ax.set_xlabel("차 안 평균 인원(재차인원)")
    save(fig, out, "fig03_load_vs_carbon.png")


def fig04_hourly(t: dict, out: Path) -> None:
    h = t["hourly"].copy()
    h["low_trips"] = h["trips"].where(h["low_load"], 0.0)
    g = h.groupby("hour").agg(pkm=("pkm", "sum"), vkm=("vkm", "sum"), trips=("trips", "sum"), low=("low_trips", "sum"))
    g["load"] = g["pkm"] / g["vkm"]
    g["low_share"] = g["low"] / g["trips"] * 100
    be = t["carbon"]["breakeven_load"].median()
    evening = h[h["hour"].isin(range(20, 23))]
    evening_share = evening["low_trips"].sum() / evening["trips"].sum()
    fig, rect = canvas(5.6, f"저녁엔 운행 4번 중 1번({evening_share:.0%})이 차 안 3명 미만",
                       "손님이 없는 게 아니라 큰 차가 필요 없는 시간이다. 이런 운행도 1회에 13명 안팎이 짧게 타고 내린다")
    left, bottom, right, top = rect
    mid = bottom + (top - bottom) * 0.50
    a1 = axes_in(fig, (left, mid, right, top), pad=(0.45, 0.25, 0.1, 0.25))
    a2 = axes_in(fig, (left, bottom, right, mid), pad=(0.45, 0.35, 0.1, 0.3))
    for ax in (a1, a2):
        clean(ax, "y")
        ax.set_xlim(5.4, 22.6)
        ax.axvspan(19.5, 22.6, color=BLUE_PALE, zorder=0, lw=0)
    a1.set_ylim(0, 11.5)
    a1.fill_between(g.index, g["load"], color=BLUE, alpha=0.10, lw=0)
    a1.plot(g.index, g["load"], color=BLUE, lw=2.2, zorder=3)
    a1.scatter(g.index, g["load"], s=22, color=BLUE, edgecolor="white", linewidth=1.2, zorder=4)
    a1.axhline(be, color=INK2, lw=1)
    a1.text(5.6, be + 0.35, f"손익분기 {be:.2f}명", fontsize=8.6, color=INK2)
    for hour in (8, 18, 21):
        a1.text(hour, g.loc[hour, "load"] + 0.75, f"{g.loc[hour, 'load']:.1f}명", ha="center", fontsize=9,
                fontweight="bold", color=INK)
    a1.text(5.45, 12.3, "차 안 평균 인원", fontsize=9.5, fontweight="bold", color=INK, va="bottom")
    a1.set_xticks([])
    a1.set_yticks([0, 5, 10], ["0", "5", "10명"])
    a2.set_ylim(0, 62)
    for hour, value in g["low_share"].items():
        vbar(a2, hour, value, 0.56, BLUE if hour >= 20 else DIM)
    for hour in (6, 12, 21):
        a2.text(hour, g.loc[hour, "low_share"] + 2.2, f"{g.loc[hour, 'low_share']:.0f}%", ha="center", fontsize=9,
                fontweight="bold", color=INK)
    a2.text(5.45, 66, "차 안 평균 3명 미만인 운행의 비중", fontsize=9.5, fontweight="bold", color=INK, va="bottom")
    a2.set_yticks([0, 20, 40, 60], ["0", "20", "40", "60%"])
    a2.set_xticks(range(6, 23, 2), [f"{x}시" for x in range(6, 23, 2)])
    save(fig, out, "fig04_hourly_load.png")


def fig05_reduction(t: dict, out: Path) -> None:
    red = reduction_numbers(t)
    fig, rect = canvas(3.6, "운행 16%만 작은 전기차로 바꿔도 전기화 효과의 절반 가까이",
                       "하루 CO₂ 감축(지금 차량 구성 대비, 각 방법을 따로 했을 때). 연 = 하루 평균 × 365")
    ax = axes_in(fig, rect, pad=(3.05, 0.1, 0.1, 0.1))
    rows = [("경유·CNG 1,229대를 모두 전기버스로", red["electric"], "버스 1,229대 교체", False),
            ("차 안 3명 미만 시간대만 소형 전기차로", red["offpeak"],
             f"운행의 {red['low_share']:.0%}만 · 차 크기 효과 {red['offpeak_size']:.1f}톤", True),
            ("② 유형 노선 차량을 소형 전기차로", red["class2"],
             f"{red['class2_routes']}개 노선 · 차 크기 효과 {red['class2_size']:.1f}톤", False)]
    ax.set_xlim(0, 26)
    ax.set_ylim(len(rows) - 0.45, -0.55)
    ax.axis("off")
    for i, (label, value, note, highlight) in enumerate(rows):
        ax.text(-0.5, i, label, ha="right", va="center", fontsize=9.6, fontweight="bold" if highlight else "regular",
                color=INK if highlight else INK2)
        hbar(ax, i, value, 0.46, BLUE if highlight else DIM)
        ax.text(value + 0.35, i - 0.08, f"하루 {value:.1f}톤", va="bottom", fontsize=10, fontweight="bold", color=INK)
        ax.text(value + 0.35, i + 0.02, f"{year_t(value)} · {note}", va="top", fontsize=8.6, color=INK2)
    save(fig, out, "fig05_reduction_options.png")


def fig06_quadrant(t: dict, out: Path) -> None:
    p = t["priority"]
    n2 = int((p["class_id"] == 2).sum())
    fig, rect = canvas(5.4, f"② 유형 {n2}개 노선: 어르신 의존은 높은데 손님에 비해 차가 크다",
                       "노선별 교통약자 의존도와 승객 1명·1km당 CO₂(가로·세로 선은 230개 노선의 중앙값, 세로축 로그 눈금)")
    ax = axes_in(fig, rect, pad=(0.62, 0.48, 0.1, 0.3))
    clean(ax, None)
    ax.spines["bottom"].set_visible(False)
    ax.set_yscale("log")
    ax.set_xlim(0, 1)
    ax.set_ylim(22, 1100)
    xm, ym = p["dependence_score"].median(), p["bus_g_per_pkm"].median()
    ax.add_patch(Rectangle((xm, ym), 1 - xm, 1100 - ym, facecolor="#fdf0ea", edgecolor="none", zorder=0))
    ax.axvline(xm, color=LINE, lw=1, zorder=1)
    ax.axhline(ym, color=LINE, lw=1, zorder=1)
    for cid, color in CLASS_COLORS.items():
        d = p[p["class_id"] == cid]
        ax.scatter(d["dependence_score"], d["bus_g_per_pkm"], s=30, color=color, edgecolor="white", linewidth=0.7,
                   alpha=0.95 if cid == 2 else 0.8, zorder=3)
    corners = {1: (0.985, 24, "right", "bottom"), 2: (0.985, 1050, "right", "top"),
               3: (0.015, 24, "left", "bottom"), 4: (0.015, 1050, "left", "top")}
    for cid, (x, y, ha, va) in corners.items():
        n = int((p["class_id"] == cid).sum())
        ax.text(x, y, f"{CLASS_NAMES[cid]} {n}개", ha=ha, va=va, fontsize=9.6, fontweight="bold",
                color="#b4441a" if cid == 2 else CLASS_COLORS[cid] if cid != 4 else INK2, zorder=5)
    ax.set_yticks([30, 100, 300, 1000], ["30g", "100g", "300g", "1,000g"])
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["낮음", "", "", "", "높음"])
    ax.set_xlabel("교통약자 의존도(고령 비율·고령 인구·대체수단 없는 승차 비중의 순위 평균) →")
    ax.text(-0.075, 1.04, "승객 1명·1km당 CO₂ (위로 갈수록 손님에 비해 차가 큼)", transform=ax.transAxes, fontsize=9,
            color=INK2, va="bottom")
    save(fig, out, "fig06_quadrant.png")


def figA2_elderly_cars(t: dict, out: Path) -> None:
    p = t["priority"].merge(t["demand"][["ROUTE_ID", "route_cars_per_1000"]], on="ROUTE_ID")
    rho = p[["catchment_elderly_share", "route_cars_per_1000"]].corr("spearman").iloc[0, 1]
    two = p["class_id"] == 2
    fig, rect = canvas(4.6, "어르신이 많은 노선일수록 동네에 승용차가 적다",
                       f"노선 주변 고령 비율과 주민 1,000명당 자가용 승용차(순위상관 {rho:.2f}). 주황 = ② 소형 차량 전환 유형")
    ax = axes_in(fig, rect, pad=(0.55, 0.45, 0.1, 0.0))
    clean(ax, "y")
    ax.scatter(p.loc[~two, "catchment_elderly_share"] * 100, p.loc[~two, "route_cars_per_1000"], s=26, color=DIM,
               edgecolor="white", linewidth=0.6, zorder=2)
    ax.scatter(p.loc[two, "catchment_elderly_share"] * 100, p.loc[two, "route_cars_per_1000"], s=32,
               color=CLASS_COLORS[2], edgecolor="white", linewidth=0.7, zorder=3)
    ax.set_xticks(range(10, 35, 5), [f"{v}%" for v in range(10, 35, 5)])
    ax.set_yticks(range(150, 500, 100), [f"{v}대" for v in range(150, 500, 100)])
    ax.set_xlabel("정류장 300m 안 평일 낮 고령(65세 이상) 비율")
    save(fig, out, "figA2_elderly_vs_cars.png")


def figA3_map(t: dict, out: Path) -> None:
    p = t["priority"]
    paths = pd.read_csv(ROUTE_PATH_CSV, dtype={"ROUTE_ID": str}).sort_values(["ROUTE_ID", "no"])
    paths = paths[paths["ROUTE_ID"].isin(p["ROUTE_ID"])]
    lines = paths.groupby("ROUTE_ID")[["gpsX", "gpsY"]].apply(lambda g: LineString(g.to_numpy()))
    routes = gpd.GeoDataFrame({"ROUTE_ID": lines.index}, geometry=lines.to_numpy(), crs=4326).to_crs(5179)
    routes = routes.merge(p[["ROUTE_ID", "class_id"]], on="ROUTE_ID")
    seoul = gpd.read_file(BOUNDARY).to_crs(5179)
    fig, rect = canvas(8.8, "네 유형의 노선은 서울 어디에 있나",
                       "② 소형 차량 전환 유형은 동작·은평·강북·서대문·성북처럼 언덕이 많은 자치구에 모여 있다")
    left, bottom, right, top = rect
    cw, ch = (right - left) / 2, (top - bottom) / 2
    for k, cid in enumerate((1, 2, 3, 4)):
        col, row = k % 2, k // 2
        ax = fig.add_axes([left + col * cw, top - (row + 1) * ch, cw * 0.98, ch * 0.92])
        seoul.plot(ax=ax, color=CARD, edgecolor="white", linewidth=0.7)
        routes[routes["class_id"] != cid].plot(ax=ax, color="#e2e5ea", linewidth=0.35)
        routes[routes["class_id"] == cid].plot(ax=ax, color=CLASS_COLORS[cid], linewidth=1.15)
        ax.set_axis_off()
        n = int((routes["class_id"] == cid).sum())
        ax.text(0.0, 1.0, f"{CLASS_NAMES[cid]}", transform=ax.transAxes, fontsize=10.5, fontweight="bold", color=INK, va="top")
        ax.text(0.0, 0.93, f"{n}개 노선", transform=ax.transAxes, fontsize=9, color=INK2, va="top")
    save(fig, out, "figA3_route_map.png")


def figA5_speed_check(t: dict, out: Path) -> None:
    s = t["speed"]
    fig, rect = canvas(4.0, "분석에 쓴 속도는 실제 마을버스 속도와 비슷한 수준",
                       "시간대별 속도(km/h). 낮에는 1km/h 안팎으로 맞고, 저녁 20~22시는 분석 속도가 약 2.5km/h 빠르다")
    ax = axes_in(fig, rect, pad=(0.5, 0.3, 0.1, 0.0))
    clean(ax, "y")
    ax.plot(s["hour"], s["observed_kmh"], color=INK2, lw=2.2)
    ax.plot(s["hour"], s["model_kmh"], color=BLUE, lw=2.2)
    ax.text(22.15, s["observed_kmh"].iloc[-1], "실측 마을버스\n(TOPIS, 10월)", fontsize=8.6, fontweight="bold",
            color=INK2, va="center")
    ax.text(22.15, s["model_kmh"].iloc[-1], "분석 입력\n(관측 + AI, 7월)", fontsize=8.6, fontweight="bold",
            color=BLUE_DARK, va="center")
    ax.set_xlim(6.8, 24.2)
    ax.set_ylim(13, 20.5)
    ax.set_xticks(range(7, 23, 3), [f"{x}시" for x in range(7, 23, 3)])
    ax.set_yticks([14, 16, 18, 20], ["14", "16", "18", "20km/h"])
    save(fig, out, "figA5_speed_check.png")


CARDS = [
    ("서대문01", "fig07_card_seodaemun01.png", "언덕 위 어르신 노선, 작은 저상 전기차로 계속 달리게"),
    ("관악04", "figA4_card_gwanak04.png", "붐비는 구간이 있는 어르신 노선, 차 크기는 그대로 지원"),
]
CELL_RAMP = LinearSegmentedColormap.from_list("ramp", ["#dbe9fb", "#86b6ef", "#2a78d6", "#0d366b"])


def fig_route_card(t: dict, out: Path, name: str, filename: str, headline: str) -> None:
    p, c = t["priority"], t["carbon"]
    row = p[p["노선명"] == name].iloc[0]
    rid = row["ROUTE_ID"]
    feas = t["feas"].set_index("ROUTE_ID").loc[rid]
    rem = t["removal"].set_index("ROUTE_ID").loc[rid]
    car = c.set_index("ROUTE_ID").loc[rid]

    path = pd.read_csv(ROUTE_PATH_CSV, dtype={"ROUTE_ID": str})
    path = path[path["ROUTE_ID"] == rid].sort_values("no")
    line = gpd.GeoSeries([LineString(path[["gpsX", "gpsY"]].to_numpy())], crs=4326).to_crs(5179)
    rs = load_route_stops()
    rs = rs[rs["ROUTE_ID"] == rid].drop_duplicates("NODE_ID")
    stops = gpd.GeoSeries(gpd.points_from_xy(rs["X좌표"], rs["Y좌표"]), crs=4326).to_crs(5179)
    grid = pd.read_csv(TRANSIT / "grid_population_daytime.csv", usecols=["CELL_ID", "cx", "cy"])
    cells = t["removal_cells"][t["removal_cells"]["ROUTE_ID"] == rid].merge(grid, on="CELL_ID")
    cells = cells[cells["added_min_net"] >= 1]
    squares = gpd.GeoDataFrame(cells, geometry=[box(x - 125, y - 125, x + 125, y + 125)
                                                for x, y in zip(cells["cx"], cells["cy"])], crs=5179)
    seoul = gpd.read_file(BOUNDARY).to_crs(5179)

    fig = plt.figure(figsize=(11, 6.0))
    fig.text(0.012, 0.975, name, fontsize=20, fontweight="heavy", color=INK, va="top")
    fig.text(0.125, 0.958, CLASS_NAMES[2], fontsize=9, fontweight="bold", color="white", va="top",
             bbox=dict(boxstyle="round,pad=0.4,rounding_size=0.8", facecolor=CLASS_COLORS[2], edgecolor="none"))
    fig.text(0.012, 0.895, headline, fontsize=11, color=INK2, va="top")

    ax = fig.add_axes([0.0, 0.08, 0.47, 0.76])
    xmin, ymin, xmax, ymax = line.total_bounds
    pad = max(xmax - xmin, ymax - ymin) * 0.12 + 300
    seoul.plot(ax=ax, color=CARD, edgecolor="white", linewidth=1.0)
    if len(squares):
        squares.plot(ax=ax, column=squares["added_min_net"].clip(upper=15), cmap=CELL_RAMP, vmin=1, vmax=15,
                     linewidth=0.6, edgecolor="white")
    line.plot(ax=ax, color=CLASS_COLORS[2], linewidth=2.6)
    stops.plot(ax=ax, color="white", markersize=16, edgecolor=INK, linewidth=1.0, zorder=4)
    ax.set_xlim(xmin - pad, xmax + pad)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_axis_off()
    cax = fig.add_axes([0.05, 0.055, 0.37, 0.018])
    sm = plt.cm.ScalarMappable(cmap=CELL_RAMP, norm=plt.Normalize(1, 15))
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.outline.set_visible(False)
    cb.set_ticks([1, 5, 10, 15], labels=["1분", "5분", "10분", "15분+"])
    cax.tick_params(labelsize=8, length=0, colors=INK2)
    fig.text(0.05, 0.085, "노선을 없애면 늘어나는 어르신 보행 시간(250m 격자, 도로망 기준)", fontsize=8.4, color=INK2)

    tiles = [("하루 승차", f"{row['boardings_per_day']:,.0f}명", f"{row['trips_per_day']:.0f}회 운행 · {row['length_km']:.1f}km"),
             ("승객 1명·1km", f"{row['bus_g_per_pkm']:.0f}g", f"같은 길 승용차 {car['car_g_per_pkm']:.0f}g"),
             ("고령 비율", f"{row['catchment_elderly_share']:.0%}", f"대체수단 없는 승차 {row['no_alt_boarding_share']:.0%}"),
             ("평균 경사", f"{feas['slope_mean']:.1f}%", f"1차로 {feas['one_lane_share']:.0%} · AI 구간 {feas['unobserved_share']:.0%}")]
    for i, (label, value, sub) in enumerate(tiles):
        tx = fig.add_axes([0.50 + i * 0.125, 0.66, 0.117, 0.20])
        tx.axis("off")
        card(tx, 0, 0, 1, 1, face="white", edge=BORDER, radius=0.08, lw=1.1)
        tx.text(0.1, 0.80, label, fontsize=8.5, color=INK2, va="center")
        tx.text(0.1, 0.48, value, fontsize=16, fontweight="heavy", color=INK, va="center")
        tx.text(0.1, 0.17, sub, fontsize=7.4, color=MUTED, va="center")

    ah = fig.add_axes([0.53, 0.33, 0.45, 0.24])
    hourly = t["hourly"][t["hourly"]["ROUTE_ID"] == rid].set_index("hour")
    clean(ah, "y")
    ah.set_xlim(5.4, 22.6)
    ah.set_ylim(0, max(hourly["avg_load"].max() * 1.3, 4))
    for hour, value in hourly["avg_load"].items():
        vbar(ah, hour, value, 0.58, BLUE if value >= 3 else CLASS_COLORS[2])
    ah.axhline(3, color=INK2, lw=0.9)
    ah.set_xticks(range(6, 23, 4), [f"{x}시" for x in range(6, 23, 4)])
    ah.tick_params(labelsize=8)
    fig.text(0.51, 0.60, "시간대별 차 안 평균 인원(명) · 가로선 3명", fontsize=9.2, fontweight="bold", color=INK)

    if row["van_feasible"]:
        action = ("저상 소형 전기차로 바꾸면", f"하루 {row['van_co2_cut_vs_current_kg_per_day']:.0f}kg · "
                  f"{year_t(row['van_co2_cut_vs_current_kg_per_day'] / 1000)} 감축")
    else:
        action = ("가장 붐비는 구간이 1회 평균 " f"{row['avg_peak_load_per_trip']:.0f}명", "소형차로는 부족 · 크기 유지, 전기화·지원")
    harm = ("노선을 없애면 어르신", f"{rem['simple_elderly_over_10min_new']:,.0f}~{rem['net_elderly_over_10min_new']:,.0f}명이 "
            f"10분 넘게 걷는다")
    for i, (face, edge, color, (small, big)) in enumerate(((BLUE_PALE, "none", BLUE_DARK, action),
                                                           ("#fdf0ea", "none", "#b4441a", harm))):
        bx = fig.add_axes([0.51 + i * 0.245, 0.07, 0.235, 0.17])
        bx.axis("off")
        card(bx, 0, 0, 1, 1, face=face, edge=edge, radius=0.08)
        bx.text(0.07, 0.68, small, fontsize=8.8, color=INK2, va="center")
        bx.text(0.07, 0.33, big, fontsize=10.2, fontweight="bold", color=color, va="center")
    save(fig, out, filename)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "docs" / "report" / "images")
    parser.add_argument("--font-dir", type=Path, default=None, help="Pretendard 글꼴 파일(.otf/.ttf)이 있는 폴더")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    setup_style(args.font_dir)
    tables = load_tables()
    print(f"그림 저장: {args.out}")
    fig00_key_numbers(tables, args.out)
    fig01_flow(args.out)
    fig02_ai_tiers(args.out)
    fig03_load_scatter(tables, args.out)
    fig04_hourly(tables, args.out)
    fig05_reduction(tables, args.out)
    fig06_quadrant(tables, args.out)
    figA1_co2_per_km(tables, args.out)
    figA2_elderly_cars(tables, args.out)
    figA3_map(tables, args.out)
    figA5_speed_check(tables, args.out)
    for name, filename, headline in CARDS:
        fig_route_card(tables, args.out, name, filename, headline)


if __name__ == "__main__":
    main()
