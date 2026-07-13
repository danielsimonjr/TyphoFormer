#!/bin/bash
# fetch_hurdat2.sh — download the latest NOAA HURDAT2 databases (Atlantic +
# NE/Central Pacific), convert them with tools/hurdat2_to_csv.py, and merge
# into one training CSV. Run from typhoformer-c/:
#
#     bash tools/fetch_hurdat2.sh [SINCE] [OUTDIR]
#     # defaults: SINCE=2004 (wind radii exist from 2004), OUTDIR=..
#
# then train text-free with the best-known recipe:
#
#     ./typhoformer train 30 --patience=8 --motion --physics --cv --huber=0.1 \
#         --csv=../HURDAT_full.csv --emb=none --threads=4
#
# Needs outbound HTTPS to www.nhc.noaa.gov (data page:
# https://www.nhc.noaa.gov/data/#hurdat). The newest file of each basin is
# discovered from the directory listing, so date-suffixed filenames
# (e.g. hurdat2-1851-2024-040425.txt) resolve automatically.
set -euo pipefail
SINCE="${1:-2004}"
OUT="${2:-..}"
BASE="https://www.nhc.noaa.gov/data/hurdat"
TOOLS="$(dirname "$0")"

listing=$(curl -fsSL "$BASE/")
atl=$(echo "$listing"  | grep -oE 'hurdat2-1[0-9]{3}-[0-9]{4}-[0-9]+\.txt'       | sort | tail -1)
epac=$(echo "$listing" | grep -oE 'hurdat2-nepac-1[0-9]{3}-[0-9]{4}-[0-9]+\.txt' | sort | tail -1)
[ -n "$atl" ]  || { echo "could not find the Atlantic hurdat2-*.txt in $BASE/"; exit 1; }
echo "Atlantic:   $atl"
echo "NE Pacific: ${epac:-<not found — continuing with Atlantic only>}"

curl -fsSL "$BASE/$atl" -o "$OUT/hurdat2_atl.txt"
python3 "$TOOLS/hurdat2_to_csv.py" "$OUT/hurdat2_atl.txt" "$OUT/HURDAT_atl.csv" --since="$SINCE"
cp "$OUT/HURDAT_atl.csv" "$OUT/HURDAT_full.csv"

if [ -n "$epac" ]; then
    curl -fsSL "$BASE/$epac" -o "$OUT/hurdat2_epac.txt"
    python3 "$TOOLS/hurdat2_to_csv.py" "$OUT/hurdat2_epac.txt" "$OUT/HURDAT_epac.csv" --since="$SINCE"
    tail -n +2 "$OUT/HURDAT_epac.csv" >> "$OUT/HURDAT_full.csv"   # append sans header
fi

echo "wrote $OUT/HURDAT_full.csv ($(($(wc -l < "$OUT/HURDAT_full.csv") - 1)) records)"
echo "train: ./typhoformer train 30 --patience=8 --motion --physics --cv --huber=0.1 --csv=$OUT/HURDAT_full.csv --emb=none --threads=4"
