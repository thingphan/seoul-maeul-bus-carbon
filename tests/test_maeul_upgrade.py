import unittest
import numpy as np
import pandas as pd
from src.transit.upgrade.demand import reconstruct
from src.transit.upgrade.model import calibration_offset,features
from src.transit.upgrade.policy import solve_portfolio
from src.transit.upgrade.energy import stop_energy_kwh
from src.energy.ev_calibration import calibrate_county_electric,per_km_terms
from src.transit.upgrade.access import summarize_access

class UpgradeTests(unittest.TestCase):
    def test_newly_disconnected_population_is_not_dropped(self):
        result=summarize_access([2,4,np.inf],[np.inf,7,np.inf],[10,20,30],'net')
        self.assertEqual(result['net_elderly_affected'],30)
        self.assertEqual(result['net_elderly_disconnected_new'],10)
        self.assertEqual(result['net_elderly_over_10min_new'],10)
        self.assertEqual(result['net_added_min_mean'],3)
        self.assertEqual(result['net_elderly_unreachable_baseline'],30)

    def test_access_threshold_is_total_after_time_not_extra_ten_minutes(self):
        result=summarize_access([9,2],[11,4],[10,20],'simple')
        self.assertEqual(result['simple_elderly_over_10min_new'],10)
        self.assertEqual(result['simple_added_min_mean'],2)
    def test_missing_alight_and_rounding_interval(self):
        frame=pd.DataFrame({'ROUTE_ID':['r']*3,'VHCL_ID':['v']*3,'TRIP_DEPR_DTM':['20260706070000']*3,'RIDE_DTM':['20260706070100','20260706070200','20260706070300'],'ALGH_DTM':['20260706070200','20260706070400',None],'PSNG_NUM':[2,3,1],'USER_DVS_CD':['01','06','07']})
        t,_=reconstruct(frame)
        self.assertEqual(t.peak_lower.iloc[0],3)
        self.assertEqual(t.peak_upper.iloc[0],6)
        self.assertEqual(t.elder_boardings.iloc[0],3)
        self.assertFalse(t.usable.iloc[0])

    def test_invalid_departure_assignment_is_flagged(self):
        frame=pd.DataFrame({'ROUTE_ID':['r'],'VHCL_ID':['v'],'TRIP_DEPR_DTM':['20260706071000'],'RIDE_DTM':['20260706070100'],'ALGH_DTM':['20260706070200'],'PSNG_NUM':[1],'USER_DVS_CD':['01']})
        t,_=reconstruct(frame)
        self.assertEqual(t.bad_board_passengers.iloc[0],1)
        self.assertFalse(t.usable.iloc[0])

    def test_cross_midnight_label_not_available_as_previous_day_outcome(self):
        frame=pd.DataFrame({'ROUTE_ID':['r'],'VHCL_ID':['v'],'TRIP_DEPR_DTM':['20260706233000'],'RIDE_DTM':['20260706233100'],'ALGH_DTM':['20260707001000'],'PSNG_NUM':[1],'USER_DVS_CD':['01']})
        t,_=reconstruct(frame)
        self.assertEqual(t.peak_upper.iloc[0],1)
        self.assertFalse(t.usable.iloc[0])

    def test_upper_calibration_order_statistic(self):
        self.assertEqual(calibration_offset([0,1,2,3],.25),3)
        self.assertEqual(calibration_offset([-2,-1],.01),0)
        with self.assertRaises(ValueError):calibration_offset([])

    def test_milp_inventory_and_optimality(self):
        d=pd.DataFrame({'ROUTE_ID':['a','a','b','b'],'service_date':['d1','d2','d1','d2'],'saved_kg':[10.,10.,15.,1.]})
        allocation,chosen,info=solve_portfolio(d,1)
        self.assertTrue(info['success'])
        self.assertEqual(allocation.sum(),1)
        self.assertEqual(allocation.get('a'),1)
        self.assertAlmostEqual(d.loc[chosen,'saved_kg'].sum(),20)

    def test_steady_certification_and_positive_stop_cycle(self):
        params,extra=calibrate_county_electric()
        a,b,c=per_km_terms(params,73)
        self.assertAlmostEqual(params.drag_coefficient*a+extra*b+c,128000/338,places=8)
        self.assertEqual(extra,0)
        self.assertGreater(stop_energy_kwh(5000,20,10,params.total_efficiency),0)

    def test_lag_features_never_use_same_or_future_date(self):
        f=pd.DataFrame({'ROUTE_ID':['r']*3,'service_date':['20260706','20260707','20260708'],'hour':[8]*3,'departure':pd.to_datetime(['2026-07-06 08:00','2026-07-07 08:00','2026-07-08 08:00']),'usable':[True]*3,'peak_upper':[2.,4.,1000.],'weekday':[0,1,2],'length_km':[5.]*3,'n_stops':[10]*3})
        priority=pd.DataFrame({'ROUTE_ID':['r'],'catchment_elderly_share':[.3],'no_alt_boarding_share':[.2],'dependence_score':[.5]})
        result,_=features(f,priority)
        self.assertTrue(np.isnan(result.lag7_mean.iloc[0]))
        self.assertEqual(result.lag7_mean.iloc[1],2)
        self.assertEqual(result.lag7_mean.iloc[2],3)

if __name__=='__main__':unittest.main()
