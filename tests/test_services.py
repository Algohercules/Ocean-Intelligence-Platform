"""
tests/test_services.py
======================
Unit tests for data, inference, argo, and heatwave backend services.
"""

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import unittest
from backend.services.data_service import DataService
from backend.services.inference_service import InferenceService
from backend.services.argo_service import ArgoService
from backend.services.heatwave_service import HeatwaveService


class TestServices(unittest.TestCase):

    def test_data_service_landmask(self):
        mask = DataService.get_land_mask()
        self.assertEqual(mask.shape, (180, 360))
        self.assertTrue(mask.dtype == bool)

    def test_data_service_point_details(self):
        res = DataService.get_point_details(lat=15.0, lon=65.0, depth=75.0)
        self.assertIn("temperature_c", res)
        self.assertIn("salinity_psu", res)
        self.assertGreater(res["temperature_c"], 0)

    def test_data_service_vertical_profile(self):
        df = DataService.get_vertical_profile(lat=12.0, lon=80.0)
        self.assertGreater(len(df), 5)
        self.assertIn("Temperature (°C)", df.columns)

    def test_inference_service_reconstruction(self):
        res = InferenceService.reconstruct_subsurface_grid(depth=50.0, model_type="conv_lstm")
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["grid_shape"], [180, 360])
        self.assertGreater(res["spearman_corr"], 0.8)

    def test_argo_service_floats(self):
        floats = ArgoService.get_all_floats(current_depth=75.0)
        self.assertGreater(len(floats), 0)
        self.assertIn("wmo_id", floats[0])

    def test_heatwave_service(self):
        hw = HeatwaveService.get_active_events()
        self.assertGreater(hw["total_active_events"], 0)


if __name__ == "__main__":
    unittest.main()
