#!/usr/bin/env python3
"""
Generate weather feature CSV for Sen1Floods11 images.

Two modes:
  1. Synthetic (default) — realistic per-event weather values with small
     per-chip random perturbations. No internet required.
  2. NASA POWER API     — fetches actual historical daily meteorological
     data from https://power.larc.nasa.gov/ using each event's lat/lon/date.
     Requires internet access; free and no API key needed.

Usage:
    # Synthetic (fast, no internet):
    python scripts/generate_weather.py --mode synthetic \
        --image_dir data/raw/S1Hand --output data/processed/weather.csv

    # Real NASA POWER data (slow, requires internet):
    python scripts/generate_weather.py --mode nasa \
        --image_dir data/raw/S1Hand --output data/processed/weather.csv
"""

import os
import re
import time
import argparse
import numpy as np
import pandas as pd

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Per-event metadata: approximate centroid lat/lon and acquisition date
# Source: Sen1Floods11 supplementary material
# ---------------------------------------------------------------------------
FLOOD_EVENT_INFO = {
    'Bolivia':  {'lat': -14.5, 'lon': -67.5, 'date': '20180215',
                 'precip': 12.5, 'temp': 28.0, 'humid': 88.0, 'wind': 2.1, 'pres': 1010.0},
    'Cambodia': {'lat':  12.5, 'lon': 104.0,  'date': '20201010',
                 'precip': 18.3, 'temp': 29.0, 'humid': 90.0, 'wind': 3.5, 'pres': 1008.0},
    'Canada':   {'lat':  52.5, 'lon':-100.0,  'date': '20200620',
                 'precip':  5.2, 'temp': 14.5, 'humid': 72.0, 'wind': 4.8, 'pres': 1015.0},
    'Ghana':    {'lat':   7.5, 'lon':  -1.5,  'date': '20190715',
                 'precip':  9.8, 'temp': 26.5, 'humid': 80.0, 'wind': 3.2, 'pres': 1011.0},
    'India':    {'lat':  25.0, 'lon':  85.0,  'date': '20190725',
                 'precip': 22.4, 'temp': 30.5, 'humid': 91.0, 'wind': 5.1, 'pres': 1006.0},
    'Mekong':   {'lat':  15.5, 'lon': 105.0,  'date': '20200820',
                 'precip': 20.1, 'temp': 30.0, 'humid': 88.0, 'wind': 3.8, 'pres': 1007.0},
    'Nigeria':  {'lat':   7.0, 'lon':   6.5,  'date': '20190905',
                 'precip': 15.6, 'temp': 27.5, 'humid': 83.0, 'wind': 2.8, 'pres': 1009.0},
    'Pakistan': {'lat':  29.0, 'lon':  70.0,  'date': '20190910',
                 'precip':  8.3, 'temp': 33.5, 'humid': 60.0, 'wind': 4.2, 'pres': 1005.0},
    'Paraguay': {'lat': -21.0, 'lon': -58.0,  'date': '20210115',
                 'precip': 11.2, 'temp': 31.0, 'humid': 78.0, 'wind': 3.0, 'pres': 1009.5},
    'Somalia':  {'lat':   2.5, 'lon':  42.5,  'date': '20190420',
                 'precip':  6.7, 'temp': 35.0, 'humid': 55.0, 'wind': 5.5, 'pres': 1008.0},
    'Spain':    {'lat':  40.0, 'lon':  -3.5,  'date': '20190915',
                 'precip':  7.4, 'temp': 22.0, 'humid': 70.0, 'wind': 6.2, 'pres': 1016.0},
}
COLUMNS = ['filename', 'precipitation', 'temperature', 'humidity', 'wind_speed', 'pressure']


def _detect_event(filename: str) -> str:
    """Extract flood event name from Sen1Floods11 filename (e.g. 'Bolivia_103757_S1Hand.tif')."""
    for event in FLOOD_EVENT_INFO:
        if event in filename:
            return event
    return None


# ---------------------------------------------------------------------------
# Mode 1: Synthetic
# ---------------------------------------------------------------------------

def generate_synthetic(image_dir: str, output_path: str, noise_std: float = 0.08, seed: int = 42):
    """Generate plausible synthetic weather per image with per-chip Gaussian noise."""
    np.random.seed(seed)
    files = sorted(f for f in os.listdir(image_dir) if f.endswith('.tif'))
    if not files:
        raise FileNotFoundError(f"No .tif files found in {image_dir}")

    rows = []
    default = {'precip': 10.0, 'temp': 25.0, 'humid': 75.0, 'wind': 4.0, 'pres': 1010.0}
    for fname in files:
        ev   = _detect_event(fname)
        info = FLOOD_EVENT_INFO.get(ev, default) if ev else default

        def jitter(val):
            return float(max(0.0, val + np.random.normal(0, val * noise_std)))

        rows.append({
            'filename':      fname,
            'precipitation': jitter(info['precip']),
            'temperature':   jitter(info['temp']),
            'humidity':      min(100.0, jitter(info['humid'])),
            'wind_speed':    jitter(info['wind']),
            'pressure':      jitter(info['pres']),
        })

    df = pd.DataFrame(rows, columns=COLUMNS)
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✓ Synthetic weather CSV saved → {output_path}  ({len(df)} rows)")
    return df


# ---------------------------------------------------------------------------
# Mode 2: NASA POWER API
# ---------------------------------------------------------------------------

NASA_POWER_URL = (
    "https://power.larc.nasa.gov/api/temporal/daily/point"
    "?parameters=PRECTOTCORR,T2M,RH2M,WS2M,PS"
    "&community=RE&format=JSON"
    "&start={start}&end={end}&latitude={lat}&longitude={lon}"
)


def _fetch_nasa_power(lat: float, lon: float, date: str) -> dict:
    """Query NASA POWER API for daily weather on a given date."""
    if not REQUESTS_AVAILABLE:
        raise ImportError("pip install requests  to use NASA POWER mode")
    url = NASA_POWER_URL.format(start=date, end=date, lat=lat, lon=lon)
    r   = requests.get(url, timeout=30)
    r.raise_for_status()
    data = r.json()['properties']['parameter']
    return {
        'precipitation': float(list(data['PRECTOTCORR'].values())[0]),
        'temperature':   float(list(data['T2M'].values())[0]),
        'humidity':      float(list(data['RH2M'].values())[0]),
        'wind_speed':    float(list(data['WS2M'].values())[0]),
        'pressure':      float(list(data['PS'].values())[0]) * 10,  # kPa → hPa
    }


def generate_nasa(image_dir: str, output_path: str, delay: float = 0.5):
    """Fetch real historical weather from NASA POWER for each image's flood event."""
    files = sorted(f for f in os.listdir(image_dir) if f.endswith('.tif'))
    if not files:
        raise FileNotFoundError(f"No .tif files found in {image_dir}")

    # Cache one API call per event
    event_cache = {}
    rows = []
    for i, fname in enumerate(files):
        ev = _detect_event(fname)
        if ev and ev not in event_cache:
            info = FLOOD_EVENT_INFO[ev]
            print(f"  Fetching NASA POWER for {ev} (lat={info['lat']}, lon={info['lon']}, date={info['date']})…")
            try:
                event_cache[ev] = _fetch_nasa_power(info['lat'], info['lon'], info['date'])
            except Exception as e:
                print(f"    WARNING: API call failed for {ev}: {e}. Using synthetic fallback.")
                event_cache[ev] = {
                    'precipitation': info['precip'], 'temperature': info['temp'],
                    'humidity':      info['humid'],  'wind_speed':  info['wind'],
                    'pressure':      info['pres'],
                }
            time.sleep(delay)

        weather = event_cache.get(ev, {
            'precipitation': 10.0, 'temperature': 25.0,
            'humidity': 75.0, 'wind_speed': 4.0, 'pressure': 1010.0,
        })
        rows.append({'filename': fname, **weather})
        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(files)} files…")

    df = pd.DataFrame(rows, columns=COLUMNS)
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n✓ NASA POWER weather CSV saved → {output_path}  ({len(df)} rows)")
    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Generate weather features for Sen1Floods11')
    parser.add_argument('--mode', choices=['synthetic', 'nasa'], default='synthetic',
                        help='Weather data source (default: synthetic)')
    parser.add_argument('--image_dir', default='data/raw/S1Hand',
                        help='Directory containing Sen1Floods11 S1Hand .tif files')
    parser.add_argument('--output', default='data/processed/weather.csv',
                        help='Output CSV path')
    parser.add_argument('--noise_std', type=float, default=0.08,
                        help='(synthetic mode) relative Gaussian noise std (default 0.08)')
    parser.add_argument('--delay', type=float, default=0.5,
                        help='(nasa mode) seconds between API calls (default 0.5)')
    args = parser.parse_args()

    print("=" * 60)
    print(f"  Weather Generator — mode: {args.mode.upper()}")
    print("=" * 60)

    if args.mode == 'synthetic':
        generate_synthetic(args.image_dir, args.output, noise_std=args.noise_std)
    else:
        generate_nasa(args.image_dir, args.output, delay=args.delay)


if __name__ == '__main__':
    main()
