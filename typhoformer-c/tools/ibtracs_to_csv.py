#!/usr/bin/env python3
"""
ibtracs_to_csv.py — convert an IBTrACS best-track CSV into the CSV schema this
repository's C loader reads (the format of HURDAT_2new_3000.csv / HURDAT_full.csv).

IBTrACS (https://www.ncei.noaa.gov/products/international-best-track-archive) is
NOAA's *global* best-track archive — every basin, not just the Atlantic + East
Pacific that HURDAT2 covers. This matters because TyphoFormer is a *typhoon*
model and the West Pacific (BASIN=WP) is the actual typhoon basin; the bundled
HURDAT data is Atlantic-heavy. FINDINGS §3 measured data as the accuracy ceiling,
so this is the highest-leverage way to grow the training set.

Usage:
    # download a per-basin file (WP = West Pacific typhoons) from
    #   https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/
    python tools/ibtracs_to_csv.py ibtracs.WP.list.v04r01.csv ../IBTRACS_WP.csv [--since=1980] [--track=main]
    # then, exactly like the HURDAT path (text-free — the language branch does not help, §2/§6):
    #   ./typhoformer train 30 --motion --physics --direct --huber=0.1 --csv=../IBTRACS_WP.csv --emb=none

Two IBTrACS specifics the C pipeline needs handled here:
  1. IBTrACS mixes 3- and 6-hourly (and off-synoptic) records; the repo's sliding
     window assumes a UNIFORM 6-hourly cadence. We keep only the synoptic times
     00/06/12/18 UTC on the minute, so every retained step is 6 h apart.
  2. IBTrACS carries "spur" and provisional sub-tracks (TRACK_TYPE); --track=main
     (default) keeps only the primary track so a storm is one clean trajectory.

Column mapping (IBTrACS -> repo schema):
  typhoon_id        = SID (+ _NAME)
  date/time         = ISO_TIME (YYYY-MM-DD HH:MM:SS -> YYYYMMDD / HHMM)
  latitude/longitude= LAT / LON, signed decimals re-encoded as HURDAT hemisphere
                      strings ("29.1N", "133.4E") so parse_latlon() reads them
  max_wind          = USA_WIND (fallback WMO_WIND), missing -> 0
  min_pressure      = USA_PRES (fallback WMO_PRES), missing -> 0
  12 wind radii     = USA_R34_NE .. USA_R64_NW, missing -> 0
  final_marker      = -999 (constant, matches the bundled CSV)
"""
import csv
import sys

HEADER = ("typhoon_id, date, time, record_identifier, system_status, latitude,"
          " longitude, max_wind, min_pressure, wind_radii_34_NE, wind_radii_34_SE,"
          " wind_radii_34_SW, wind_radii_34_NW, wind_radii_50_NE, wind_radii_50_SE,"
          " wind_radii_50_SW, wind_radii_50_NW, wind_radii_64_NE, wind_radii_64_SE,"
          " wind_radii_64_SW, wind_radii_64_NW, final_marker")

RADII = ["USA_R34_NE", "USA_R34_SE", "USA_R34_SW", "USA_R34_NW",
         "USA_R50_NE", "USA_R50_SE", "USA_R50_SW", "USA_R50_NW",
         "USA_R64_NE", "USA_R64_SE", "USA_R64_SW", "USA_R64_NW"]


def num(v, fallback=""):
    """IBTrACS missing values are blank; some numerics use ' ' or negative fills.
    Clamp to a non-negative integer string like the HURDAT converter (repo uses 0)."""
    for cand in (v, fallback):
        cand = (cand or "").strip()
        if cand:
            try:
                return str(max(0, int(round(float(cand)))))
            except ValueError:
                pass
    return "0"


def latlon(v, is_lat):
    """Signed decimal -> HURDAT hemisphere string. lat: +N/-S, lon: +E/-W."""
    x = float(v)
    if is_lat:
        return f"{abs(x):.4f}{'N' if x >= 0 else 'S'}"
    # normalize lon to (-180,180] so the E/W letter is meaningful
    while x > 180.0:  x -= 360.0
    while x <= -180.0: x += 360.0
    return f"{abs(x):.4f}{'E' if x >= 0 else 'W'}"


def main(inp, outp, since=1980, track="main"):
    storms, kept_rows = set(), 0
    with open(inp, newline="") as f, open(outp, "w") as out:
        out.write(HEADER + "\n")
        r = csv.reader(f)
        cols = next(r)              # column-name header
        idx = {name: i for i, name in enumerate(cols)}
        next(r, None)               # units row (IBTrACS second header line)
        def g(row, name):
            j = idx.get(name, -1)
            return row[j].strip() if 0 <= j < len(row) else ""
        for row in r:
            if not row:
                continue
            if track and g(row, "TRACK_TYPE") and g(row, "TRACK_TYPE") != "main":
                continue
            iso = g(row, "ISO_TIME")            # "YYYY-MM-DD HH:MM:SS"
            if len(iso) < 16:
                continue
            yyyy, mm, dd = iso[0:4], iso[5:7], iso[8:10]
            hh, mi = iso[11:13], iso[14:16]
            if mi != "00" or hh not in ("00", "06", "12", "18"):
                continue            # keep only synoptic 6-hourly steps
            if int(yyyy) < since:
                continue
            lat, lon = g(row, "LAT"), g(row, "LON")
            if not lat or not lon or lat == " " or lon == " ":
                continue
            try:
                lat_s, lon_s = latlon(lat, True), latlon(lon, False)
            except ValueError:
                continue
            sid = g(row, "SID") or g(row, "NUMBER")
            name = g(row, "NAME") or "UNNAMED"
            storm_id = f"{sid}_{name}"
            wind = num(g(row, "USA_WIND"), g(row, "WMO_WIND"))
            pres = num(g(row, "USA_PRES"), g(row, "WMO_PRES"))
            radii = [num(g(row, c)) for c in RADII]
            status = g(row, "USA_STATUS") or g(row, "NATURE") or "XX"
            out.write(f"{storm_id},{yyyy}{mm}{dd},{hh}{mi},,{status},{lat_s},{lon_s},"
                      f"{wind},{pres},{','.join(radii)},-999\n")
            storms.add(storm_id)
            kept_rows += 1
    print(f"{inp}: wrote {kept_rows} synoptic records across {len(storms)} storms "
          f"(since {since}, track={track}) -> {outp}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    since, track = 1980, "main"
    for a in sys.argv[3:]:
        if a.startswith("--since="): since = int(a[8:])
        elif a.startswith("--track="): track = a[8:]
    main(sys.argv[1], sys.argv[2], since, track)
