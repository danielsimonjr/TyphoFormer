#!/usr/bin/env python3
"""
era5_steering_poc.py — proof-of-concept for roadmap Item 8 (FINDINGS §25).

Does the environmental STEERING FLOW predict tropical-cyclone motion? A TC is
largely advected by the deep-layer mean wind, but TyphoFormer sees only position,
velocity, intensity and text — never the atmosphere that moves the storm. This
script collocates the 500 hPa wind from ERA5 with HURDAT track points and
correlates it against each storm's observed next-6h motion.

It establishes two things that the *full* steering feature (not built yet — see
§25) will need:
  1. The data is reachable with NO Copernicus CDS credential: Google's public
     ARCO-ERA5 mirror opens anonymously and carries u/v on all 37 levels.
  2. The premise holds: measured corr(u500, Δlon) ~ +0.63, corr(v500, Δlat) ~ +0.91.

Requires xarray + zarr + gcsfs (install on a Python that has pip — the stripped
WSL system Python does not; the machine's Windows Python does):
    python -m pip install xarray zarr gcsfs numpy
Run:
    python tools/era5_steering_poc.py ../HURDAT_full.csv [N_POINTS]

NOTE: naive point extraction is ~27 s/point (ARCO's 1-hourly chunks are ~150 MB).
The production feature must batch reads by timestamp; this POC deliberately samples
only a handful of points.
"""
import sys, csv, time
import numpy as np
import xarray as xr

ARCO = "gs://gcp-public-data-arco-era5/ar/1959-2022-full_37-1h-0p25deg-chunk-1.zarr-v2"


def parse_ll(s):
    s = s.strip(); h = s[-1]
    x = float(s[:-1]) if h in "NSEW" else float(s)
    return -x if h in "SW" else x


def sample_transitions(path, lo_year, hi_year, n):
    """A few 6-hourly same-storm transitions in [lo_year, hi_year] (ARCO ends 2021)."""
    pts, prev = [], None
    for row in csv.reader(open(path)):
        if len(row) < 8 or row[0].strip() == "typhoon_id":
            continue
        try:
            lat, lon = parse_ll(row[5]), parse_ll(row[6])
        except ValueError:
            prev = None; continue
        d, t = row[1].strip(), row[2].strip()
        if not (lo_year <= int(d[:4]) <= hi_year) or t not in ("0000", "0600", "1200", "1800"):
            prev = None; continue
        cur = (row[0].strip(), d, t, lat, lon)
        if prev and prev[0] == cur[0]:
            dlat, dlon = lat - prev[3], lon - prev[4]
            if abs(dlat) < 4 and abs(dlon) < 4:      # drop day-boundary gaps
                pts.append((prev[3], prev[4], prev[1], prev[2], dlat, dlon))
        prev = cur
    step = max(1, len(pts) // n)
    return pts[::step][:n]


def main(path, n):
    pts = sample_transitions(path, 2015, 2020, n)
    print(f"{len(pts)} track transitions sampled; opening ARCO-ERA5 (anon)...", flush=True)
    t0 = time.time()
    ds = xr.open_zarr(ARCO, storage_options={"token": "anon"}, chunks=None)
    u = ds["u_component_of_wind"].sel(level=500)
    v = ds["v_component_of_wind"].sel(level=500)
    print(f"  opened in {time.time()-t0:.0f}s", flush=True)
    su, sv, mlon, mlat = [], [], [], []
    for i, (lat, lon, d, t, dlat, dlon) in enumerate(pts):
        tt = np.datetime64(f"{d[:4]}-{d[4:6]}-{d[6:8]}T{t[:2]}:00")
        t0 = time.time()
        uu = float(u.sel(time=tt, method="nearest").sel(latitude=lat, longitude=lon % 360, method="nearest").values)
        vv = float(v.sel(time=tt, method="nearest").sel(latitude=lat, longitude=lon % 360, method="nearest").values)
        su.append(uu); sv.append(vv); mlon.append(dlon); mlat.append(dlat)
        print(f"  pt{i} ({lat:.1f},{lon:.1f}) u500={uu:+.1f} v500={vv:+.1f} | "
              f"move Δlon={dlon:+.2f} Δlat={dlat:+.2f}  [{time.time()-t0:.0f}s]", flush=True)
    su, sv, mlon, mlat = map(np.array, (su, sv, mlon, mlat))
    if len(su) >= 4:
        print(f"\ncorr(u500, Δlon) = {np.corrcoef(su, mlon)[0,1]:+.3f}"
              f"   corr(v500, Δlat) = {np.corrcoef(sv, mlat)[0,1]:+.3f}", flush=True)
        print("positive => the 500 hPa steering flow predicts storm motion (the premise).", flush=True)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 6)
