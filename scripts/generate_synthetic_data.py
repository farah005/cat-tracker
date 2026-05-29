#!/usr/bin/env python3
"""
generate_synthetic_data.py
──────────────────────────
Generates a realistic synthetic GPS dataset for a domestic cat.

Behaviour model
───────────────
* The cat spends most of its time near home (within 200 m).
* It makes 1-3 excursions per day (up to 800 m away).
* It is mostly active at dawn (05-07 h) and dusk (18-21 h) – crepuscular.
* GPS noise: ±0.0001° (≈ 10 m) Gaussian.
* Occasional outlier spikes are added then naturally handled by the median filter.

Usage
─────
    python scripts/generate_synthetic_data.py \
        --days 30 \
        --lat-home 48.8566 \
        --lon-home 2.3522 \
        --output data/synthetic_cat.csv
"""
import argparse
import math
import random
import csv
from datetime import datetime, timedelta, timezone
from pathlib import Path


# ── Helpers ───────────────────────────────────────────────────────────────────

def _activity_level(hour: float) -> float:
    """Return a 0-1 activity weight based on hour of day (crepuscular pattern)."""
    dawn = math.exp(-((hour - 6) ** 2) / 4)
    dusk = math.exp(-((hour - 19) ** 2) / 4)
    return max(0.05, dawn + dusk)


def _random_bearing() -> float:
    return random.uniform(0, 2 * math.pi)


def _move(lat: float, lon: float, bearing: float, distance_m: float):
    """Move a point by distance_m metres in the given bearing."""
    R = 6_371_000.0
    d = distance_m / R
    lat1 = math.radians(lat)
    lon1 = math.radians(lon)
    lat2 = math.asin(math.sin(lat1) * math.cos(d) + math.cos(lat1) * math.sin(d) * math.cos(bearing))
    lon2 = lon1 + math.atan2(
        math.sin(bearing) * math.sin(d) * math.cos(lat1),
        math.cos(d) - math.sin(lat1) * math.sin(lat2),
    )
    return math.degrees(lat2), math.degrees(lon2)


def _add_noise(lat: float, lon: float, sigma: float = 0.00008):
    """Add Gaussian GPS noise."""
    import random as _r
    return lat + _r.gauss(0, sigma), lon + _r.gauss(0, sigma)


# ── Generator ─────────────────────────────────────────────────────────────────

def generate(
    days: int,
    lat_home: float,
    lon_home: float,
    interval_min: int = 10,
    outlier_prob: float = 0.02,
) -> list[dict]:
    """
    Returns a list of dicts {timestamp, latitude, longitude}.
    """
    records = []
    start = datetime(2025, 5, 1, 0, 0, 0, tzinfo=timezone.utc)
    total_steps = days * 24 * 60 // interval_min

    lat, lon = lat_home, lon_home
    bearing  = _random_bearing()
    excursion_remaining = 0          # steps still in excursion mode
    excursion_target_lat = lat_home
    excursion_target_lon = lon_home

    for step in range(total_steps):
        ts   = start + timedelta(minutes=step * interval_min)
        hour = ts.hour + ts.minute / 60
        act  = _activity_level(hour)

        # ── Decide movement ───────────────────────────────────────────────────
        if excursion_remaining > 0:
            # Move towards excursion target
            dy = excursion_target_lat - lat
            dx = excursion_target_lon - lon
            bearing = math.atan2(dx * math.cos(math.radians(lat)), dy)
            dist = random.uniform(30, 80) * act
            lat, lon = _move(lat, lon, bearing, dist)
            excursion_remaining -= 1
        else:
            # Random walk biased towards home
            home_dy = lat_home - lat
            home_dx = lon_home - lon
            home_dist = math.sqrt(home_dy ** 2 + home_dx ** 2) * 111_000
            home_pull = min(1.0, home_dist / 300)

            if random.random() < home_pull * 0.4:
                # Pull towards home
                bearing = math.atan2(home_dx * math.cos(math.radians(lat)), home_dy)
                dist = random.uniform(10, 40)
            else:
                # Random wander
                bearing += random.gauss(0, 0.5)
                dist = random.uniform(0, 25) * act

            lat, lon = _move(lat, lon, bearing, dist)

            # Occasionally start an excursion
            if act > 0.4 and random.random() < 0.015:
                angle = _random_bearing()
                excursion_dist = random.uniform(200, 800)
                excursion_target_lat, excursion_target_lon = _move(lat_home, lon_home, angle, excursion_dist)
                excursion_remaining = random.randint(10, 30)

        # ── GPS noise ─────────────────────────────────────────────────────────
        lat_n, lon_n = _add_noise(lat, lon)

        # Occasional outlier spike (simulates GPS glitch)
        if random.random() < outlier_prob:
            lat_n += random.gauss(0, 0.003)
            lon_n += random.gauss(0, 0.003)

        records.append({
            "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "latitude":  round(lat_n, 7),
            "longitude": round(lon_n, 7),
        })

    return records


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic cat GPS data")
    parser.add_argument("--days",      type=int,   default=30,       help="Number of days to simulate")
    parser.add_argument("--lat-home",  type=float, default=48.8566,  help="Home latitude")
    parser.add_argument("--lon-home",  type=float, default=2.3522,   help="Home longitude")
    parser.add_argument("--interval",  type=int,   default=10,       help="GPS interval in minutes")
    parser.add_argument("--output",    type=str,   default="data/synthetic_cat.csv")
    args = parser.parse_args()

    print(f"Generating {args.days} days of GPS data "
          f"(Δt={args.interval} min) around ({args.lat_home}, {args.lon_home}) …")

    records = generate(
        days=args.days,
        lat_home=args.lat_home,
        lon_home=args.lon_home,
        interval_min=args.interval,
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "latitude", "longitude"])
        writer.writeheader()
        writer.writerows(records)

    print(f"✅  {len(records)} records written to {out}")


if __name__ == "__main__":
    main()
