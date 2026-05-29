"""
Geo utility functions – distance, speed, median filter, home range.
All pure Python / numpy – no DB dependency so they are easily unit-testable.
"""
import math
import numpy as np
from shapely.geometry import MultiPoint, mapping
from typing import List, Tuple


EARTH_RADIUS_M = 6_371_000.0


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Haversine distance between two WGS-84 coordinates, in metres.

    Parameters
    ----------
    lat1, lon1 : float  – first point (decimal degrees)
    lat2, lon2 : float  – second point (decimal degrees)

    Returns
    -------
    float  – distance in metres (≥ 0)
    """
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi       = math.radians(lat2 - lat1)
    dlambda    = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


def speed_ms(lat1: float, lon1: float, ts1, lat2: float, lon2: float, ts2) -> float:
    """Speed in m/s between two GPS fixes. Returns 0.0 if Δt ≤ 0."""
    dt_s = (ts2 - ts1).total_seconds()
    if dt_s <= 0:
        return 0.0
    dist = haversine_m(lat1, lon1, lat2, lon2)
    return dist / dt_s


def median_filter_positions(
    lats: List[float],
    lons: List[float],
    window: int = 5,
) -> Tuple[List[float], List[float]]:
    """
    Apply a 1-D median filter independently on latitudes and longitudes
    to remove GPS noise spikes.

    Parameters
    ----------
    lats, lons : lists of floats
    window     : filter half-window (odd integer recommended)

    Returns
    -------
    Tuple[List[float], List[float]]  – filtered lats, filtered lons
    """
    arr_lat = np.array(lats, dtype=float)
    arr_lon = np.array(lons, dtype=float)
    half = window // 2

    filtered_lat, filtered_lon = [], []
    for i in range(len(arr_lat)):
        lo, hi = max(0, i - half), min(len(arr_lat), i + half + 1)
        filtered_lat.append(float(np.median(arr_lat[lo:hi])))
        filtered_lon.append(float(np.median(arr_lon[lo:hi])))

    return filtered_lat, filtered_lon


def compute_convex_hull(
    lats: List[float], lons: List[float]
) -> Tuple[dict, float, Tuple[float, float]]:
    """
    Compute the convex hull (home range) of GPS points.

    Returns
    -------
    geojson_polygon : dict       – GeoJSON-serialisable Polygon
    area_km2        : float      – approximate area in km²
    centroid        : (lat, lon)
    """
    if len(lats) < 3:
        raise ValueError("At least 3 points are required to compute a convex hull.")

    points   = MultiPoint(list(zip(lons, lats)))   # shapely uses (x=lon, y=lat)
    hull     = points.convex_hull
    geojson  = mapping(hull)

    # Approximate area via equirectangular projection
    lat_mid  = sum(lats) / len(lats)
    cos_lat  = math.cos(math.radians(lat_mid))
    # Convert degree² to km²
    area_deg2 = hull.area
    km_per_deg_lat = 111.32
    km_per_deg_lon = 111.32 * cos_lat
    area_km2 = area_deg2 * km_per_deg_lat * km_per_deg_lon

    centroid_geom = hull.centroid
    centroid      = (centroid_geom.y, centroid_geom.x)  # (lat, lon)

    return dict(geojson), abs(area_km2), centroid
