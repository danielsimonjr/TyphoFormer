#!/usr/bin/env python3
"""
hurdat2_to_csv.py — convert a raw NOAA HURDAT2 database file into the CSV
schema this repository's C loader reads (the format of HURDAT_2new_3000.csv).

This is the dataset-scaling path: the bundled CSV covers 2020-2024 (~98
storms), while NOAA publishes HURDAT2 back to 1851. Because the language
branch measurably does not help on this task (FINDINGS §2/§6), the converted
CSV can be used WITHOUT text embeddings — train with `--emb=none`, which
feeds the same zero vectors as the `--no_text` ablation. No GPT-4o/MiniLM
regeneration required.

Usage:
    python tools/hurdat2_to_csv.py hurdat2-1851-2024.txt ../HURDAT_full.csv [--since=2004]
    # download the input from https://www.nhc.noaa.gov/data/#hurdat
    # then: ./typhoformer train 30 --motion --physics --cv \
    #           --csv=../HURDAT_full.csv --emb=none

--since filters to storms whose first record is in that year or later.
The default 2004 is where the 34/50/64-kt wind radii start being recorded
(before that they are -999 = missing); pass --since=1851 for everything,
at the cost of zero-filled radii columns.

HURDAT2 input format (comma-separated, fixed field order):
    header  AL092021,             IDA,     40,
            ^basin+number+year    ^name    ^number of data rows
    data    20210829, 1655, L, HU, 29.1N, 90.2W, 130, 931, 130, 110, ...
            date, time, record id, status, lat, lon, max wind, min pressure,
            twelve wind radii (34NE..64NW) [, radius of max wind (2022+ files)]

Output schema (one row per record, header included):
    typhoon_id, date, time, record_identifier, system_status, latitude,
    longitude, max_wind, min_pressure, wind_radii_34_NE ... wind_radii_64_NW,
    final_marker
typhoon_id is "<cycloneid>_<NAME>" (e.g. AL092021_IDA); missing numeric
values (HURDAT2's -99/-999) are written as 0 to match the bundled CSV;
final_marker is the constant -999.
"""
import sys

HEADER = ("typhoon_id, date, time, record_identifier, system_status, latitude,"
          " longitude, max_wind, min_pressure, wind_radii_34_NE, wind_radii_34_SE,"
          " wind_radii_34_SW, wind_radii_34_NW, wind_radii_50_NE, wind_radii_50_SE,"
          " wind_radii_50_SW, wind_radii_50_NW, wind_radii_64_NE, wind_radii_64_SE,"
          " wind_radii_64_SW, wind_radii_64_NW, final_marker")


def clamp(v):
    """HURDAT2 marks missing numerics as -99/-999; the repo CSV uses 0."""
    try:
        return str(max(0, int(v)))
    except ValueError:
        return "0"


def main(inp, outp, since=2004):
    storms = kept = rows = 0
    with open(inp) as f, open(outp, "w") as out:
        out.write(HEADER + "\n")
        storm_id, keep = None, False
        for line in f:
            fields = [x.strip() for x in line.strip().split(",")]
            if len(fields) >= 3 and fields[0][:2].isalpha() and len(fields[0]) == 8:
                # header line: AL092021, IDA, 40
                cyclone, name = fields[0], fields[1] or "UNNAMED"
                year = int(cyclone[4:8])
                storm_id = f"{cyclone}_{name}"
                keep = year >= since
                storms += 1
                kept += keep
                continue
            if storm_id is None or not keep or len(fields) < 20:
                continue
            # data line: date, time, recid, status, lat, lon, wind, pressure, 12 radii
            date, time, recid, status, lat, lon = fields[0:6]
            wind, pressure = clamp(fields[6]), clamp(fields[7])
            radii = [clamp(x) for x in fields[8:20]]
            out.write(f"{storm_id},{date},{time},{recid},{status},{lat},{lon},"
                      f"{wind},{pressure},{','.join(radii)},-999\n")
            rows += 1
    print(f"{inp}: {storms} storms total, kept {kept} (since {since}), wrote {rows} records -> {outp}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    since = 2004
    for a in sys.argv[3:]:
        if a.startswith("--since="):
            since = int(a[8:])
    main(sys.argv[1], sys.argv[2], since)
