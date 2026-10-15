"""전기차 에너지 모델 VT-CPEM(Fiori et al. 2016)의 링크 단위 근사식을 구현한다.

    E = [max(F_구름 + F_경사 + F_공기, 0) · u / η + P_보조] · Δt
    F_구름 = m·g·Crr(u)·cos θ,  F_경사 = m·g·sin θ,  F_공기 = ½·ρ·Cd·A·u²,  Δt = 링크 길이 / u

가정
----
* 링크를 지나는 동안 평균속도는 일정하다(가속도 0). 정지·출발 소비는 이 함수에 포함되지 않으며,
  v2 버스 시나리오는 upgrade.energy의 명시적 정지·출발 항을 별도로 더한다.
* 회생제동은 적용하지 않는다. 내리막에서 바퀴 동력이 음수여도 추진 동력은 0으로
  제한하고, 보조장치 전력은 계속 소비한다.
* DEM 기반 원시 경사도는 짧은 링크에서 과대해질 수 있어 [-12%, +12%]로 제한한다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class VTCPEMParameters:
    """VT-CPEM 원 논문의 기준 차량(닛산 리프) 상수. 버스·승용차는 ev_calibration에서 차종별로 바꿔 쓴다."""

    mass_kg: float = 1521.0
    drag_coefficient: float = 0.28
    frontal_area_m2: float = 2.3316
    air_density_kg_m3: float = 1.2256
    gravity_m_s2: float = 9.8066
    drivetrain_efficiency: float = 0.92
    motor_efficiency: float = 0.91
    battery_efficiency: float = 0.90
    auxiliary_power_w: float = 700.0
    slope_clip_pct: float = 12.0

    @property
    def total_efficiency(self) -> float:
        return self.drivetrain_efficiency * self.motor_efficiency * self.battery_efficiency


DEFAULT_PARAMETERS = VTCPEMParameters()
REQUIRED_INPUT_COLUMNS = ("LINK_ID", "LENGTH", "speed_kmh", "slope_pct")


def rolling_resistance_coefficient(speed_m_s: np.ndarray) -> np.ndarray:
    """Crr(u) = 1.75e-3 * (0.0328u + 4.575)."""
    return 1.75e-3 * (0.0328 * speed_m_s + 4.575)


def validate_energy_inputs(frame: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_INPUT_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"필수 입력 컬럼이 없습니다: {missing}")

    numeric = frame[["LENGTH", "speed_kmh", "slope_pct"]].apply(pd.to_numeric, errors="coerce")
    invalid = {
        "length": int((numeric["LENGTH"].isna() | (numeric["LENGTH"] <= 0)).sum()),
        "speed": int((numeric["speed_kmh"].isna() | (numeric["speed_kmh"] <= 0)).sum()),
        "slope": int(numeric["slope_pct"].isna().sum()),
    }
    if any(invalid.values()):
        raise ValueError(f"에너지 계산 불가 입력 행: {invalid}")


def calculate_link_energy(
    frame: pd.DataFrame,
    parameters: VTCPEMParameters = DEFAULT_PARAMETERS,
) -> pd.DataFrame:
    """링크별 속도·길이·경사에서 VT-CPEM 에너지(energy_ev_wh 등)와 중간 항을 벡터 계산한다."""
    validate_energy_inputs(frame)
    result = frame.copy()

    length_m = pd.to_numeric(result["LENGTH"], errors="coerce").to_numpy(dtype="float64")
    speed_kmh = pd.to_numeric(result["speed_kmh"], errors="coerce").to_numpy(dtype="float64")
    slope_raw = pd.to_numeric(result["slope_pct"], errors="coerce").to_numpy(dtype="float64")

    speed_m_s = speed_kmh / 3.6
    slope_used = np.clip(slope_raw, -parameters.slope_clip_pct, parameters.slope_clip_pct)
    theta_rad = np.arctan(slope_used / 100.0)
    travel_time_sec = length_m / speed_m_s

    crr = rolling_resistance_coefficient(speed_m_s)
    rolling_force_n = parameters.mass_kg * parameters.gravity_m_s2 * crr * np.cos(theta_rad)
    grade_force_n = parameters.mass_kg * parameters.gravity_m_s2 * np.sin(theta_rad)
    aero_force_n = (
        0.5
        * parameters.air_density_kg_m3
        * parameters.drag_coefficient
        * parameters.frontal_area_m2
        * np.square(speed_m_s)
    )
    wheel_power_w = (rolling_force_n + grade_force_n + aero_force_n) * speed_m_s
    traction_battery_power_w = np.maximum(wheel_power_w, 0.0) / parameters.total_efficiency
    battery_power_w = traction_battery_power_w + parameters.auxiliary_power_w
    energy_ev_j = battery_power_w * travel_time_sec
    energy_ev_wh = energy_ev_j / 3600.0

    result["slope_pct_raw"] = slope_raw
    result["slope_pct_used"] = slope_used
    result["slope_outlier_flag"] = (np.abs(slope_raw) > parameters.slope_clip_pct).astype("int8")
    result["speed_m_s"] = speed_m_s
    result["travel_time_sec"] = travel_time_sec
    result["crr"] = crr
    result["rolling_force_n"] = rolling_force_n
    result["grade_force_n"] = grade_force_n
    result["aero_force_n"] = aero_force_n
    result["wheel_power_w"] = wheel_power_w
    result["traction_battery_power_w"] = traction_battery_power_w
    result["battery_power_w"] = battery_power_w
    result["energy_ev_j"] = energy_ev_j
    result["energy_ev_wh"] = energy_ev_wh
    result["energy_ev_kwh"] = energy_ev_j / 3_600_000.0
    result["energy_wh_per_km"] = energy_ev_wh / (length_m / 1000.0)
    return result
