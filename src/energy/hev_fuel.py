"""하이브리드(HEV) 승용차의 링크 단위 연료·CO2 근사식.

Ahn & Rakha(2019, "A Simple Hybrid Electric Vehicle Fuel Consumption Model for Transportation
Applications", IntechOpen, DOI 10.5772/intechopen.89055)의 VT-CPFM 계열 HEV 모델을 따른다.

    엔진 모드  FC [ml/s] = a + b·v + c·P + d·P²     (v ≥ 32km/h 또는 P ≥ 10kW, P > 0)
    전기 모드  v < 32km/h 이고 P < 10kW 이거나 P ≤ 0
    계수(2010 Toyota Prius): a=0.006, b=0.003998, c=0.077092, d=-9.155E-05 (v: km/h, P: kW)

원 논문은 전기 모드 연료를 '전기 모드 구간 평균 연료율'로 두지만 값을 제시하지 않는다. 일반
하이브리드는 외부 충전이 없으므로(충전 유지형) 전기 모드에서 쓴 배터리 에너지를 결국 엔진이
채운다. 그래서 전기 모드 연료를 엔진 모드의 한계 연료율(c)로 그 동력을 만들 때의 연료를
충방전 효율로 나눈 값(c·P/η_cs)으로 대체한다.

링크 내 정속(가속도 0) 근사에서는 가감속 연료가 빠지므로, 휘발유차(ice_fuel.py)와 같은 방식으로
공인 도심·고속 연비를 각 시험주기 평균속도의 정속 주행에서 재현하도록 두 계수를 역산한다.
    FC_보정 = SC × FC_모델 + β     (SC: 차량 규모 배율 — 원 논문의 공인연비 비율 배율에 해당,
                                    β: 가감속 등 정속 근사에서 빠지는 시간당 부가 연료)

기준 차량: 현대 쏘나타 디 엣지 하이브리드 2.0(18인치) 공차중량 1,585kg + 운전자 75kg,
공인 도심 16.8km/L·고속 17.4km/L(제조사 공개 제원). 경사는 ±12%로 제한한다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .link_energy import rolling_resistance_coefficient, validate_energy_inputs


@dataclass(frozen=True)
class HEVParameters:
    mass_kg: float = 1585.0 + 75.0
    drag_coefficient: float = 0.27
    frontal_area_m2: float = 1.860 * 1.445 * 0.85
    air_density_kg_m3: float = 1.2256
    gravity_m_s2: float = 9.8066
    driveline_efficiency: float = 0.92
    co2_kg_per_l: float = 2.29
    city_km_per_l: float = 16.8
    highway_km_per_l: float = 17.4
    city_cycle_kmh: float = 34.1
    highway_cycle_kmh: float = 77.7
    ev_speed_kmh: float = 32.0
    ev_power_kw: float = 10.0
    a: float = 0.006
    b: float = 0.003998
    c: float = 0.077092
    d: float = -9.155e-05
    charge_sustain_efficiency: float = 0.8
    slope_clip_pct: float = 12.0


DEFAULT_HEV = HEVParameters()


def power_kw(params: HEVParameters, speed_m_s: np.ndarray, theta_rad: np.ndarray) -> np.ndarray:
    crr = rolling_resistance_coefficient(speed_m_s)
    force = (
        params.mass_kg * params.gravity_m_s2 * crr * np.cos(theta_rad)
        + params.mass_kg * params.gravity_m_s2 * np.sin(theta_rad)
        + 0.5 * params.air_density_kg_m3 * params.drag_coefficient * params.frontal_area_m2 * np.square(speed_m_s)
    )
    return force * speed_m_s / params.driveline_efficiency / 1000.0


def model_fuel_ml_s(params: HEVParameters, speed_kmh: np.ndarray, p_kw: np.ndarray) -> np.ndarray:
    engine = params.a + params.b * speed_kmh + params.c * p_kw + params.d * np.square(p_kw)
    ev_mode = (p_kw <= 0) | ((speed_kmh < params.ev_speed_kmh) & (p_kw < params.ev_power_kw))
    ev_fuel = params.c * np.maximum(p_kw, 0.0) / params.charge_sustain_efficiency
    return np.where(ev_mode, ev_fuel, engine)


def calibrate(params: HEVParameters = DEFAULT_HEV) -> tuple[float, float]:
    """공인 도심·고속 연비를 정속 평지 주행에서 재현하는 (SC, β [ml/s])를 구한다."""
    rows, rhs = [], []
    for kmh, km_per_l in ((params.city_cycle_kmh, params.city_km_per_l), (params.highway_cycle_kmh, params.highway_km_per_l)):
        v = np.array([kmh])
        p = power_kw(params, v / 3.6, np.array([0.0]))
        seconds_per_km = 1000.0 / (kmh / 3.6)
        rows.append([float(model_fuel_ml_s(params, v, p)[0]) * seconds_per_km, seconds_per_km])
        rhs.append(1000.0 / km_per_l)  # ml/km
    scale, beta = np.linalg.solve(np.array(rows), np.array(rhs))
    if scale <= 0 or beta < 0:
        raise ValueError(f"보정 실패: SC={scale:.3g}, β={beta:.3g}")
    return float(scale), float(beta)


def calculate_link_fuel(frame: pd.DataFrame, params: HEVParameters = DEFAULT_HEV) -> pd.DataFrame:
    """링크별 속도·길이·경사에서 하이브리드 승용차 연료[L]와 CO2[g]를 벡터 계산한다."""
    validate_energy_inputs(frame)
    scale, beta = calibrate(params)
    length_m = pd.to_numeric(frame["LENGTH"], errors="coerce").to_numpy(dtype="float64")
    speed_kmh = pd.to_numeric(frame["speed_kmh"], errors="coerce").to_numpy(dtype="float64")
    slope = np.clip(pd.to_numeric(frame["slope_pct"], errors="coerce").to_numpy(dtype="float64"),
                    -params.slope_clip_pct, params.slope_clip_pct)
    travel_time_sec = length_m / (speed_kmh / 3.6)
    p = power_kw(params, speed_kmh / 3.6, np.arctan(slope / 100.0))
    fuel_l = (scale * model_fuel_ml_s(params, speed_kmh, p) + beta) * travel_time_sec / 1000.0
    result = frame.copy()
    result["travel_time_sec"] = travel_time_sec
    result["power_kw"] = p
    result["fuel_l"] = fuel_l
    result["co2_g"] = fuel_l * params.co2_kg_per_l * 1000.0
    result["co2_g_per_km"] = result["co2_g"] / (length_m / 1000.0)
    return result
