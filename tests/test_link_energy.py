"""차량 에너지·배출 모델 단위 테스트: 손계산 일치, 경사·정체 방향성, 공인 연비·전비 재현, 서울 실측 비율 재현."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.energy import bus_fuel, hev_fuel, ice_fuel
from src.energy.ev_calibration import calibrate_two_point
from src.energy.ice_fuel import DEFAULT_ICE, calculate_link_fuel
from src.energy.link_energy import DEFAULT_PARAMETERS, calculate_link_energy, rolling_resistance_coefficient


class LinkEnergyTests(unittest.TestCase):
    def test_flat_link_matches_hand_calculation(self):
        # 1 km, 36 km/h(=10 m/s), 평지의 손계산 가능한 기준 사례.
        frame = pd.DataFrame({"LINK_ID": ["flat"], "LENGTH": [1000.0], "speed_kmh": [36.0], "slope_pct": [0.0]})
        row = calculate_link_energy(frame).iloc[0]
        p = DEFAULT_PARAMETERS
        u = 10.0
        crr = rolling_resistance_coefficient(np.array([u]))[0]
        force = p.mass_kg * p.gravity_m_s2 * crr + 0.5 * p.air_density_kg_m3 * p.drag_coefficient * p.frontal_area_m2 * u**2
        expected_j = (force * u / p.total_efficiency + p.auxiliary_power_w) * 100.0
        self.assertAlmostEqual(row["travel_time_sec"], 100.0, places=9)
        self.assertAlmostEqual(row["energy_ev_j"], expected_j, places=6)

    def test_uphill_cost_exceeds_flat_and_downhill_keeps_auxiliary_cost(self):
        frame = pd.DataFrame({
            "LINK_ID": ["down", "flat", "up"],
            "LENGTH": [1000.0, 1000.0, 1000.0],
            "speed_kmh": [36.0, 36.0, 36.0],
            "slope_pct": [-5.0, 0.0, 5.0],
        })
        result = calculate_link_energy(frame).set_index("LINK_ID")
        self.assertGreater(result.loc["up", "energy_ev_j"], result.loc["flat", "energy_ev_j"])
        self.assertGreaterEqual(result.loc["down", "energy_ev_j"], 700.0 * result.loc["down", "travel_time_sec"])
        self.assertGreater(result.loc["down", "energy_ev_j"], 0.0)

    def test_extreme_slope_is_clipped(self):
        frame = pd.DataFrame({"LINK_ID": ["extreme"], "LENGTH": [100.0], "speed_kmh": [20.0], "slope_pct": [310.0]})
        row = calculate_link_energy(frame).iloc[0]
        self.assertEqual(row["slope_pct_used"], 12.0)
        self.assertEqual(row["slope_outlier_flag"], 1)


class IceFuelTests(unittest.TestCase):
    def test_calibration_reproduces_certified_fuel_economy(self):
        p = DEFAULT_ICE
        frame = pd.DataFrame({
            "LINK_ID": ["city", "highway"],
            "LENGTH": [1000.0, 1000.0],
            "speed_kmh": [p.city_cycle_kmh, p.highway_cycle_kmh],
            "slope_pct": [0.0, 0.0],
        })
        km_per_l = 1.0 / calculate_link_fuel(frame)["fuel_l"].to_numpy()
        np.testing.assert_allclose(km_per_l, [p.city_km_per_l, p.highway_km_per_l], rtol=1e-9)

    def test_congestion_and_uphill_raise_fuel_per_km(self):
        frame = pd.DataFrame({
            "LINK_ID": ["jam", "free", "up"],
            "LENGTH": [1000.0, 1000.0, 1000.0],
            "speed_kmh": [10.0, 40.0, 40.0],
            "slope_pct": [0.0, 0.0, 5.0],
        })
        result = calculate_link_fuel(frame).set_index("LINK_ID")
        self.assertGreater(result.loc["jam", "fuel_l"], result.loc["free", "fuel_l"])
        self.assertGreater(result.loc["up", "fuel_l"], result.loc["free", "fuel_l"])

    def test_car_presets_reproduce_their_certified_fuel_economy(self):
        for preset in ice_fuel.CAR_PRESETS.values():
            frame = pd.DataFrame({
                "LINK_ID": ["city", "highway"], "LENGTH": [1000.0, 1000.0],
                "speed_kmh": [preset.city_cycle_kmh, preset.highway_cycle_kmh], "slope_pct": [0.0, 0.0],
            })
            km_per_l = 1.0 / calculate_link_fuel(frame, preset)["fuel_l"].to_numpy()
            np.testing.assert_allclose(km_per_l, [preset.city_km_per_l, preset.highway_km_per_l], rtol=1e-9)


class HevAndBusTests(unittest.TestCase):
    def test_hev_calibration_reproduces_certified_fuel_economy(self):
        p = hev_fuel.DEFAULT_HEV
        frame = pd.DataFrame({
            "LINK_ID": ["city", "highway"], "LENGTH": [1000.0, 1000.0],
            "speed_kmh": [p.city_cycle_kmh, p.highway_cycle_kmh], "slope_pct": [0.0, 0.0],
        })
        km_per_l = 1.0 / hev_fuel.calculate_link_fuel(frame)["fuel_l"].to_numpy()
        np.testing.assert_allclose(km_per_l, [p.city_km_per_l, p.highway_km_per_l], rtol=1e-9)

    def test_hev_uses_less_fuel_than_conventional_car_in_slow_traffic(self):
        frame = pd.DataFrame({"LINK_ID": ["slow"], "LENGTH": [1000.0], "speed_kmh": [15.0], "slope_pct": [0.0]})
        hev = hev_fuel.calculate_link_fuel(frame)["co2_g"].iloc[0]
        ice = calculate_link_fuel(frame, ice_fuel.CAR_PRESETS["휘발유"])["co2_g"].iloc[0]
        self.assertLess(hev, ice)

    def test_bus_fuel_ratio_reproduces_seoul_measurement(self):
        self.assertAlmostEqual(bus_fuel.diesel_g_per_km(bus_fuel.SEOUL_EBUS_KWH_PER_KM), 723.0, places=9)
        self.assertAlmostEqual(bus_fuel.cng_g_per_km(bus_fuel.SEOUL_EBUS_KWH_PER_KM), 721.0, places=9)

    def test_ev_two_point_calibration_reproduces_inputs(self):
        points = [(34.1, 172.4), (77.7, 217.4)]
        params, extra_w = calibrate_two_point(DEFAULT_PARAMETERS, points)
        from dataclasses import replace
        params = replace(params, auxiliary_power_w=params.auxiliary_power_w + extra_w)
        frame = pd.DataFrame({"LINK_ID": ["a", "b"], "LENGTH": [1000.0, 1000.0],
                              "speed_kmh": [34.1, 77.7], "slope_pct": [0.0, 0.0]})
        np.testing.assert_allclose(calculate_link_energy(frame, params)["energy_ev_wh"].to_numpy(), [172.4, 217.4], rtol=1e-9)


if __name__ == "__main__":
    unittest.main()
