#!/bin/bash
# fetch_ibtracs.sh — download an IBTrACS best-track basin file from NOAA NCEI,
# convert it with tools/ibtracs_to_csv.py, and write a training CSV. Run from
# typhoformer-c/:
#
#     bash tools/fetch_ibtracs.sh [BASIN] [SINCE] [OUTDIR]
#     # defaults: BASIN=WP (West Pacific — the typhoon basin), SINCE=1980, OUTDIR=..
#     # basins: WP EP NA NI SI SP SA  (or ALL for the global file, ~400 MB)
#
# then train text-free with the best-known recipe:
#
#     ./typhoformer train 30 --patience=8 --motion --physics --direct --huber=0.1 \
#         --csv=../IBTRACS_WP.csv --emb=none --threads=4
#
# Needs outbound HTTPS to www.ncei.noaa.gov. IBTrACS is the WMO-blessed *global*
# best-track archive; unlike HURDAT2 (Atlantic + E/C Pacific only) it covers the
# West Pacific, the actual typhoon basin. Data is the accuracy ceiling (FINDINGS
# §3), and this is how the training set grows past HURDAT's 826 storms.
set -euo pipefail

BASIN="${1:-WP}"
SINCE="${2:-1980}"
OUTDIR="${3:-..}"
VER="v04r01"
BASE="https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/${VER}/access/csv"
URL="${BASE}/ibtracs.${BASIN}.list.${VER}.csv"
RAW="$(mktemp -t ibtracs.XXXXXX.csv)"
OUT="${OUTDIR}/IBTRACS_${BASIN}.csv"
HERE="$(cd "$(dirname "$0")" && pwd)"

echo "downloading IBTrACS ${BASIN} (${VER}) ..."
curl -fSL --retry 3 --max-time 600 -o "$RAW" "$URL"
echo "  $(wc -l < "$RAW") raw lines, $(du -h "$RAW" | cut -f1)"

python3 "$HERE/ibtracs_to_csv.py" "$RAW" "$OUT" "--since=$SINCE" --track=main
rm -f "$RAW"
echo "wrote $OUT"
