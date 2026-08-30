"""
tests/test_api.py
=================
API Endpoint tests for FastAPI backend routes.
"""

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import unittest
from fastapi.testclient import TestClient
from backend.api.main import app

client = TestClient(app)


class TestAPIEndpoints(unittest.TestCase):

    def test_health_check(self):
        response = client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")

    def test_ocean_stats(self):
        response = client.get("/api/ocean/stats?lat=15.0&lon=65.0&depth=75.0")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("temperature_c", data)

    def test_argo_floats(self):
        response = client.get("/api/argo/floats?depth=75.0")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsInstance(data, list)
        self.assertGreater(len(data), 0)

    def test_predict_profile(self):
        response = client.get("/api/predict/profile?lat=15.0&lon=65.0")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("conv_lstm_temp", data)

    def test_heatwave_events(self):
        response = client.get("/api/heatwave/events")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("total_active_events", data)


if __name__ == "__main__":
    unittest.main()
