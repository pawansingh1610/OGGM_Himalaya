#!/usr/bin/env bash
# Download external datasets for the population-exposure analysis into data_external/.
# Logs every URL with its UTC download date and file size to data_external/download_log.tsv
# (SOURCES.md is written from that log). Re-running skips files that already exist.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/data_external"; mkdir -p "$OUT/raw"
LOG="$OUT/download_log.tsv"
[ -f "$LOG" ] || printf 'file\turl\tdownloaded_utc\tbytes\n' > "$LOG"

get() {  # url
  local url="$1" f; f="$OUT/raw/$(basename "$url")"
  if [ -s "$f" ]; then echo "exists: $(basename "$f")"; return; fi
  curl -fL --retry 3 --max-time 3600 -o "$f.part" "$url"
  mv "$f.part" "$f"
  printf '%s\t%s\t%s\t%s\n' "$(basename "$f")" "$url" "$(date -u +%Y-%m-%dT%H:%MZ)" "$(stat -c %s "$f")" >> "$LOG"
  echo "ok: $(basename "$f")"
}

# HydroBASINS v1c (Lehner & Grill 2013), standard (no lakes), Asia + Europe/Middle East
# (Central Asian basins - Amu Darya, Syr Darya, Balkhash - may fall in either tile).
for reg in as eu; do
  for lev in 03 04 05 06 07 08; do
    get "https://data.hydrosheds.org/file/HydroBASINS/standard/hybas_${reg}_lev${lev}_v1c.zip"
  done
done

# GHS-POP R2023A (JRC), 30 arc-second, WGS84. 2025 is a projection in R2023A (supplementary only).
for ep in 1975 2000 2020 2025; do
  get "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_POP_GLOBE_R2023A/GHS_POP_E${ep}_GLOBE_R2023A_4326_30ss/V1-0/GHS_POP_E${ep}_GLOBE_R2023A_4326_30ss_V1_0.zip"
done

cd "$OUT/raw"
for z in *.zip; do
  d="${z%.zip}"; [ -d "$d" ] || unzip -q -o "$z" -d "$d"
done
echo "done"
