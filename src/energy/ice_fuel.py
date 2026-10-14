"""휘발유 승용차의 링크 단위 연료·CO2 근사식.

VT-CPEM(전기차)과 같은 연구 계열의 VT-CPFM(Rakha et al., 2011, Transportation Research
Part D 16(7)) 구조를 따른다. 바퀴에 필요한 힘(구름·경사·공기저항)은 전기차 식과 공유하고,
그 힘을 연료로 바꾸는 단계만 다르다.

    엔진 출력 P = max(바퀴 동력, 0) / 구동계 효율
    연료율 FC [L/s] = α0 + P / (η × 휘발유 발열량)

α0는 '공회전 + 가감속 등 정속 근사에서 빠지는 부가 소비'를 묶은 기본 연료율, η는 엔진의 유효
열효율이다. 링크 내 정속(가속도 0) 근사에 맞추기 위해 원 모델의 2차항은 두지 않고, 두 계수를
공인 도심·고속 연비가 각 시험주기 평균속도의 정속 주행에서 재현되도록 역산한다. 정속 근사에서
빠지는 가감속 연료가 α0에 흡수되므로 저속·정체 구간일수록 1km당 연료가 커진다.

가정
----
* 기준 차량: 현대 아반떼 1.6 가솔린(16인치) 공차중량 1,265kg + 운전자 75kg,
  공인 도심 13.3km/L·고속 17.0km/L(제조사 공개 제원).
* 공기저항계수 0.27, 전면적 2.20m²(전폭 1.825m × 전고 1.42m × 0.85).
* 시험주기 평균속도: 도심 FTP-75 34.1km/h, 고속 HWFET 77.7km/h.
* 휘발유 발열량 32.0MJ/L, 배기관 CO2 2.29kg/L(캐나다 천연자원부 자료).
* 경사는 전기차 식과 같이 ±12%로 제한한다. 내리막에서 바퀴 동력이 음수면 엔진 출력은 0이고
  기본 연료율 α0만 소비한다.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from .link_energy import rolling_resistance_coefficient, validate_energy_inputs


@dataclass(frozen=True)
class ICEParameters:
    mass_kg: float = 1265.0 + 75.0
    drag_coefficient: float = 0.27
    frontal_area_m2: float = 2.20
    air_density_kg_m3: float = 1.2256
    gravity_m_s2: float = 9.8066
    driveline_efficiency: float = 0.92
    fuel_energy_j_per_l: float = 32.0e6
    co2_kg_per_l: float = 2.29
    city_km_per_l: float = 13.3
    highway_km_per_l: float = 17.0
    city_cycle_kmh: float = 34.1
    highway_cycle_kmh: float = 77.7
    slope_clip_pct: float = 12.0


DEFAULT_ICE = ICEParameters()

# 서울 승용차 연료별 대표 차종(중형급, 제조사 공개 제원: 공차중량 + 운전자 75kg, 공인 도심·고속 연비).
# 전면적 = 전폭 × 전고 × 0.85. CO2: 휘발유 2.29·경유 2.66kg/L(캐나다 천연자원부),
# LPG 1.68kg/L(IPCC 기본 배출계수 63,100kgCO2/TJ × 부탄 순발열량 45.7MJ/kg × 0.584kg/L).
CAR_PRESETS = {
    "휘발유": replace(DEFAULT_ICE, mass_kg=1475.0 + 75.0, frontal_area_m2=1.860 * 1.445 * 0.85,
                    city_km_per_l=11.2, highway_km_per_l=14.8),                     # 쏘나타 디 엣지 2.0(17인치)
    "경유": replace(DEFAULT_ICE, mass_kg=1815.0 + 75.0, drag_coefficient=0.33, frontal_area_m2=1.900 * 1.695 * 0.85,
                  co2_kg_per_l=2.66, city_km_per_l=12.7, highway_km_per_l=17.0),     # 쏘렌토 2.2 디젤 2WD(18인치)
    "LPG": replace(DEFAULT_ICE, mass_kg=1475.0 + 75.0, frontal_area_m2=1.860 * 1.445 * 0.85,
                   co2_kg_per_l=1.68, city_km_per_l=8.3, highway_km_per_l=11.7),     # 쏘나타 디 엣지 LPi 2.0(17인치)
}


def engine_power_w(params: ICEParameters, speed_m_s: np.ndarray, theta_rad: np.ndarray) -> np.ndarray:
    crr = rolling_resistance_coefficient(speed_m_s)
    force = (
        params.mass_kg * params.gravity_m_s2 * crr * np.cos(theta_rad)
        + params.mass_kg * params.gravity_m_s2 * np.sin(theta_rad)
        + 0.5 * params.air_density_kg_m3 * params.drag_coefficient * params.frontal_area_m2 * np.square(speed_m_s)
    )
    return np.maximum(force * speed_m_s, 0.0) / params.driveline_efficiency


def calibrate(params: ICEParameters = DEFAULT_ICE) -> tuple[float, float]:
    """공인 도심·고속 연비를 정속 평지 주행에서 재현하는 (α0 [L/s], η)를 구한다.

    각 주기에서 1km 연료 = (α0 + P/(η·LHV)) × (1000/u) 이므로 두 식을 α0, 1/η에 대한
    연립 일차방정식으로 푼다.
    """
    rows, rhs = [], []
    for kmh, km_per_l in ((params.city_cycle_kmh, params.city_km_per_l), (params.highway_cycle_kmh, params.highway_km_per_l)):
        u = kmh / 3.6
        p = float(engine_power_w(params, np.array([u]), np.array([0.0]))[0])
        seconds_per_km = 1000.0 / u
        rows.append([seconds_per_km, p / params.fuel_energy_j_per_l * seconds_per_km])
        rhs.append(1.0 / km_per_l)
    alpha0, inv_eta = np.linalg.solve(np.array(rows), np.array(rhs))
    if alpha0 <= 0 or inv_eta <= 0:
        raise ValueError(f"보정 실패: α0={alpha0:.3g}, 1/η={inv_eta:.3g}")
    return float(alpha0), float(1.0 / inv_eta)


def calculate_link_fuel(frame: pd.DataFrame, params: ICEParameters = DEFAULT_ICE) -> pd.DataFrame:
    """링크별 속도·길이·경사에서 휘발유 승용차 연료[L]와 CO2[g]를 벡터 계산한다."""
    validate_energy_inputs(frame)
    alpha0, eta = calibrate(params)
    length_m = pd.to_numeric(frame["LENGTH"], errors="coerce").to_numpy(dtype="float64")
    speed_m_s = pd.to_numeric(frame["speed_kmh"], errors="coerce").to_numpy(dtype="float64") / 3.6
    slope = np.clip(pd.to_numeric(frame["slope_pct"], errors="coerce").to_numpy(dtype="float64"),
                    -params.slope_clip_pct, params.slope_clip_pct)
    travel_time_sec = length_m / speed_m_s
    power = engine_power_w(params, speed_m_s, np.arctan(slope / 100.0))
    fuel_l = (alpha0 + power / (eta * params.fuel_energy_j_per_l)) * travel_time_sec
    result = frame.copy()
    result["travel_time_sec"] = travel_time_sec
    result["engine_power_w"] = power
    result["fuel_l"] = fuel_l
    result["co2_g"] = fuel_l * params.co2_kg_per_l * 1000.0
    result["co2_g_per_km"] = result["co2_g"] / (length_m / 1000.0)
    return result
