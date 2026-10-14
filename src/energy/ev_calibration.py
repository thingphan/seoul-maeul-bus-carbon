"""전기차(VT-CPEM) 파라미터를 공인 전비·주행거리로 보정한다.

휘발유·하이브리드 승용차(ice_fuel.py, hev_fuel.py)는 공인 도심·고속 연비가 각 시험주기 평균속도의
정속 주행에서 재현되도록 시간당 부가 소비(가감속 등)를 넣어 보정한다. 전기차도 같은 기준을 맞추기
위해, 링크 정속 근사의 VT-CPEM에 '보조전력 + 부가 전력'(시간에 비례하는 소비)을 두고 공기저항계수와
부가 전력 두 값을 공인값 두 개로 역산한다. 두 값 모두 1km당 에너지에 선형으로 들어가므로 2×2
연립방정식으로 풀린다.

    1km당 전력[Wh] = [m·g·Crr(u) + ½·ρ·Cd·A·u²] / (η·3.6) + (P_aux + P_extra) / v
                     (u: m/s, v: km/h, 평지)

공인 도심·고속 값이 따로 없는 전기버스(현대 카운티 일렉트릭)는 calibrate_county_electric에서
정속 주행거리로 Cd를, 복합 주행거리로 부가 전력을 차례로 역산한다.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy.optimize import brentq

from .link_energy import VTCPEMParameters, rolling_resistance_coefficient

CITY_KMH, HWY_KMH = 34.1, 77.7  # 도심·고속 시험주기 평균속도(승용차 보정과 같은 값)

# 현대 카운티 일렉트릭(마을버스 사양, 제조사 공개값): 배터리 128kWh, 73km/h 정속 338km, 복합 303km.
# 공차중량·전면적·공기저항계수는 공개 자료가 없어 질량 6t·보조전력 1kW에서 Cd를 역산하고,
# 전면적은 전폭 약 2.0m × 전고 약 2.7m로 둔다.
COUNTY_BATTERY_KWH = 128.0
COUNTY_STEADY_RANGE_KM, COUNTY_STEADY_KMH = 338.0, 73.0
COUNTY_COMBINED_RANGE_KM = 303.0
COUNTY_CERT = VTCPEMParameters(mass_kg=6000.0, frontal_area_m2=5.4, auxiliary_power_w=1000.0)


def per_km_terms(params: VTCPEMParameters, kmh: float) -> tuple[float, float, float]:
    """(Cd 계수항, 부가전력 계수항, 상수항) — 1km당 Wh가 이 세 항의 선형결합이 되도록 분해한다."""
    u = kmh / 3.6
    rolling = params.mass_kg * params.gravity_m_s2 * float(rolling_resistance_coefficient(np.array([u]))[0])
    aero_per_cd = 0.5 * params.air_density_kg_m3 * params.frontal_area_m2 * u * u
    eff = params.total_efficiency
    return aero_per_cd / eff / 3.6, 1.0 / kmh, rolling / eff / 3.6 + params.auxiliary_power_w / kmh


def calibrate_two_point(params: VTCPEMParameters, points: list[tuple[float, float]]) -> tuple[VTCPEMParameters, float]:
    """정속 두 점 (속도 km/h, 1km당 Wh)을 재현하는 (공기저항계수, 부가 전력 W)를 구한다.

    부가 전력은 보조전력에 더해 돌려준다(시간에 비례하는 소비로 같은 자리에 들어가기 때문).
    """
    a, rhs = [], []
    for kmh, wh_per_km in points:
        cd_term, extra_term, const = per_km_terms(params, kmh)
        a.append([cd_term, extra_term])
        rhs.append(wh_per_km - const)
    cd, extra_w = np.linalg.solve(np.array(a), np.array(rhs))
    if cd <= 0:
        raise ValueError(f"보정 실패: Cd={cd:.3g}")
    return replace(params, drag_coefficient=float(cd)), float(extra_w)


def combined_wh_per_km(params: VTCPEMParameters, extra_w: float, city_kmh: float, hwy_kmh: float,
                       city_share: float = 0.55) -> float:
    """도심·고속 정속 근사를 연비(km/kWh) 기준 가중 조화평균해 복합 1km당 Wh를 낸다."""
    def wh(kmh: float) -> float:
        cd_term, extra_term, const = per_km_terms(params, kmh)
        return params.drag_coefficient * cd_term + extra_w * extra_term + const
    km_per_kwh = city_share * 1000.0 / wh(city_kmh) + (1 - city_share) * 1000.0 / wh(hwy_kmh)
    return 1000.0 / km_per_kwh


def calibrate_county_electric() -> tuple[VTCPEMParameters, float]:
    """카운티 일렉트릭의 (Cd를 넣은 인증 조건 파라미터, 가감속 부가 전력 W)를 돌려준다.

    1) '평지 73km/h 정속, 보조전력 1kW, 질량 6t'에서 정속 주행거리 338km가 나오도록 Cd를 구한다.
    2) 복합 주행거리 303km(도심·고속 정속 근사를 55:45로 연비 가중 조화평균)가 재현되도록 시간당
       부가 전력을 구한다. 승용차(공인 도심·고속 연비 보정)와 같은 기준으로 정지·출발 소비를 넣기 위함이다.
    """
    cd_term, _, const = per_km_terms(COUNTY_CERT, COUNTY_STEADY_KMH)
    cd = (COUNTY_BATTERY_KWH * 1000.0 / COUNTY_STEADY_RANGE_KM - const) / cd_term
    if cd <= 0:
        raise ValueError("보정 실패: 구름저항만으로 공식 전비를 넘습니다. 질량 가정을 확인하세요.")
    params = replace(COUNTY_CERT, drag_coefficient=float(cd))
    target = COUNTY_BATTERY_KWH * 1000.0 / COUNTY_COMBINED_RANGE_KM
    extra_w = brentq(lambda x: combined_wh_per_km(params, x, CITY_KMH, HWY_KMH) - target, 0.0, 50000.0)
    return params, float(extra_w)
