"""
tests/test_geo.py
─────────────────
Unit tests for geo utilities (distance, speed, filter, convex hull)
and for the prediction endpoint response shape.

Run with:  pytest tests/ -v
"""
import math
import pytest
from datetime import datetime, timezone, timedelta


# ── Import the module under test ──────────────────────────────────────────────
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from app.services.geo import (
    haversine_m,
    speed_ms,
    median_filter_positions,
    compute_convex_hull,
)


# ═══════════════════════════════════════════════════════════════════════════════
# haversine_m
# ═══════════════════════════════════════════════════════════════════════════════

class TestHaversine:
    def test_same_point_is_zero(self):
        assert haversine_m(48.8566, 2.3522, 48.8566, 2.3522) == pytest.approx(0.0, abs=1e-6)

    def test_known_distance_paris_london(self):
        # Paris  (48.8566, 2.3522)  →  London (51.5074, -0.1278)
        # Haversine on a sphere gives ≈ 343-344 km (ignores Earth's oblateness)
        d = haversine_m(48.8566, 2.3522, 51.5074, -0.1278)
        assert 340_000 < d < 350_000, f"Got {d:.0f} m"

    def test_symmetry(self):
        d1 = haversine_m(48.85, 2.35, 48.86, 2.36)
        d2 = haversine_m(48.86, 2.36, 48.85, 2.35)
        assert d1 == pytest.approx(d2, rel=1e-9)

    def test_short_distance_approx_100m(self):
        # Moving ~0.001° in latitude ≈ 111 m
        d = haversine_m(48.8566, 2.3522, 48.8576, 2.3522)
        assert 100 < d < 120, f"Got {d:.1f} m"

    def test_returns_positive(self):
        d = haversine_m(0.0, 0.0, -1.0, -1.0)
        assert d > 0

    def test_equator_one_degree_longitude(self):
        # At equator 1° longitude ≈ 111.319 km
        d = haversine_m(0.0, 0.0, 0.0, 1.0)
        assert 111_000 < d < 112_000, f"Got {d:.0f} m"


# ═══════════════════════════════════════════════════════════════════════════════
# speed_ms
# ═══════════════════════════════════════════════════════════════════════════════

class TestSpeedMs:
    T0 = datetime(2025, 5, 1, 12, 0, 0, tzinfo=timezone.utc)
    T1 = datetime(2025, 5, 1, 12, 10, 0, tzinfo=timezone.utc)  # +10 min

    def test_zero_distance_zero_speed(self):
        v = speed_ms(48.85, 2.35, self.T0, 48.85, 2.35, self.T1)
        assert v == pytest.approx(0.0, abs=1e-6)

    def test_zero_time_returns_zero(self):
        v = speed_ms(48.85, 2.35, self.T0, 48.86, 2.36, self.T0)
        assert v == 0.0

    def test_known_speed(self):
        # distance ≈ 111 m in 10 min (600 s) → ~0.185 m/s
        t1 = self.T0 + timedelta(seconds=600)
        v  = speed_ms(48.8566, 2.3522, self.T0, 48.8576, 2.3522, t1)
        assert 0.15 < v < 0.25, f"Got {v:.4f} m/s"

    def test_speed_is_non_negative(self):
        v = speed_ms(48.85, 2.35, self.T0, 48.86, 2.36, self.T1)
        assert v >= 0


# ═══════════════════════════════════════════════════════════════════════════════
# median_filter_positions
# ═══════════════════════════════════════════════════════════════════════════════

class TestMedianFilter:
    def test_output_length_matches_input(self):
        lats = [48.85, 48.851, 48.852, 48.853, 48.854]
        lons = [2.35,  2.351,  2.352,  2.353,  2.354]
        fl, fo = median_filter_positions(lats, lons, window=3)
        assert len(fl) == len(lats)
        assert len(fo) == len(lons)

    def test_spike_is_attenuated(self):
        """A large spike in the middle should be smoothed out."""
        lats = [48.85] * 5
        lons = [2.35]  * 5
        lats[2] = 49.0   # big spike
        lons[2] = 3.0
        fl, fo = median_filter_positions(lats, lons, window=5)
        # Middle value should be much closer to 48.85 than 49.0
        assert fl[2] < 48.9, f"Spike not attenuated: {fl[2]}"

    def test_flat_signal_unchanged(self):
        lats = [48.85] * 10
        lons = [2.35]  * 10
        fl, fo = median_filter_positions(lats, lons)
        for v in fl:
            assert v == pytest.approx(48.85)
        for v in fo:
            assert v == pytest.approx(2.35)

    def test_single_point(self):
        fl, fo = median_filter_positions([48.85], [2.35])
        assert fl == [pytest.approx(48.85)]
        assert fo == [pytest.approx(2.35)]


# ═══════════════════════════════════════════════════════════════════════════════
# compute_convex_hull
# ═══════════════════════════════════════════════════════════════════════════════

class TestConvexHull:
    # A simple square of ≈ 1 km side near Paris
    LATS = [48.850, 48.850, 48.860, 48.860]
    LONS = [2.350,  2.362,  2.350,  2.362]

    def test_returns_geojson_polygon(self):
        geojson, area, centroid = compute_convex_hull(self.LATS, self.LONS)
        assert geojson["type"] in ("Polygon", "MultiPolygon")

    def test_area_positive(self):
        _, area, _ = compute_convex_hull(self.LATS, self.LONS)
        assert area > 0

    def test_area_roughly_correct(self):
        # ~1.1 km × 1.1 km square → area ≈ 0.7-1.5 km²
        _, area, _ = compute_convex_hull(self.LATS, self.LONS)
        assert 0.5 < area < 2.5, f"Area = {area:.4f} km²"

    def test_centroid_inside_bbox(self):
        _, _, (clat, clon) = compute_convex_hull(self.LATS, self.LONS)
        assert min(self.LATS) <= clat <= max(self.LATS)
        assert min(self.LONS) <= clon <= max(self.LONS)

    def test_raises_on_too_few_points(self):
        with pytest.raises(ValueError):
            compute_convex_hull([48.85, 48.86], [2.35, 2.36])

    def test_large_dataset(self):
        import random, math
        n = 500
        lats = [48.856 + random.gauss(0, 0.005) for _ in range(n)]
        lons = [2.352  + random.gauss(0, 0.005) for _ in range(n)]
        geojson, area, centroid = compute_convex_hull(lats, lons)
        assert area > 0
        assert geojson["type"] in ("Polygon", "MultiPolygon")


# ═══════════════════════════════════════════════════════════════════════════════
# Prediction schema validation (no DB required – mock the endpoint response)
# ═══════════════════════════════════════════════════════════════════════════════

class TestPredictionSchema:
    def test_prediction_out_model(self):
        from app.models.schemas import PredictionOut
        pred = PredictionOut(
            chat_id=1,
            predicted_latitude=48.857,
            predicted_longitude=2.353,
        )
        assert pred.chat_id == 1
        assert pred.predicted_latitude == pytest.approx(48.857)
        assert pred.model_version == "lstm_v1"

    def test_home_range_out_model(self):
        from app.models.schemas import HomeRangeOut
        hr = HomeRangeOut(
            chat_id=1,
            area_km2=1.234,
            polygon_geojson={"type": "Polygon", "coordinates": []},
            centroid={"lat": 48.856, "lon": 2.352},
            n_points=100,
        )
        assert hr.area_km2 == pytest.approx(1.234)
        assert hr.n_points == 100

    def test_upload_result_model(self):
        from app.models.schemas import UploadResult
        r = UploadResult(chat_id=1, inserted=500, skipped=3, model_retrained=False)
        assert r.inserted == 500
        assert r.skipped == 3
