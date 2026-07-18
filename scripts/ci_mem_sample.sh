#!/usr/bin/env bash
# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0
#
# Lightweight system-memory sampler for CI headroom analysis. The instrumented
# Linux coverage job is the only place we run the full suite under load, so we
# sample memory there to (a) confirm whether the box ever swaps and (b) size how
# many pytest-xdist workers would fit before hitting the runner's RAM ceiling.
#
# Two modes:
#
#   ci_mem_sample.sh sample [INTERVAL_S] [OUT_TSV]
#       Loop until killed, appending one row every INTERVAL_S seconds (default 2):
#       elapsed_s, mem_used_mb, mem_avail_mb, swap_used_mb, max_proc_rss_mb.
#       Run it in the background and kill the PID to stop it.
#
#   ci_mem_sample.sh summary [OUT_TSV]
#       Print the peak memory used, minimum memory available, peak swap used, and
#       the peak single-process RSS from a TSV produced by `sample`. The peak
#       single-process RSS approximates one xdist worker's footprint: with N
#       workers, expect roughly that much resident per worker on top of the
#       shared baseline.
#
# All values are MB. Best-effort — it must never fail the job it wraps.

set -uo pipefail

mode="${1:-sample}"

sample_row() {
  # elapsed \t mem_used \t mem_avail \t swap_used  (MB), from /proc/meminfo (kB).
  local elapsed="$1"
  awk -v elapsed="$elapsed" '
    /^MemTotal:/     { mt = $2 }
    /^MemAvailable:/ { ma = $2 }
    /^SwapTotal:/    { st = $2 }
    /^SwapFree:/     { sf = $2 }
    END {
      printf "%s\t%d\t%d\t%d", elapsed, int((mt - ma) / 1024), int(ma / 1024), int((st - sf) / 1024)
    }' /proc/meminfo
  # Largest single-process RSS right now (ps reports kB -> MB).
  local rss
  rss=$(ps -eo rss= 2>/dev/null | awk 'BEGIN { m = 0 } { if ($1 > m) m = $1 } END { print int(m / 1024) }')
  printf "\t%s\n" "${rss:-0}"
}

case "$mode" in
  sample)
    interval="${2:-2}"
    out="${3:-mem-samples.tsv}"
    printf 'elapsed_s\tmem_used_mb\tmem_avail_mb\tswap_used_mb\tmax_proc_rss_mb\n' > "$out"
    {
      printf '# MemTotal_mb=%s\n' "$(awk '/^MemTotal:/ { print int($2 / 1024) }' /proc/meminfo)"
      printf '# SwapTotal_mb=%s\n' "$(awk '/^SwapTotal:/ { print int($2 / 1024) }' /proc/meminfo)"
      printf '# nproc=%s\n' "$(nproc 2>/dev/null || echo '?')"
    } >> "$out"
    start=$SECONDS
    while :; do
      sample_row "$((SECONDS - start))" >> "$out"
      sleep "$interval"
    done
    ;;
  summary)
    out="${2:-mem-samples.tsv}"
    [ -f "$out" ] || { echo "no samples file: $out" >&2; exit 0; }
    grep -v '^#' "$out" | awk -F'\t' '
      NR == 1 { next }  # header
      {
        if ($2 > peak_used) peak_used = $2
        if (min_avail == "" || $3 < min_avail) min_avail = $3
        if ($4 > peak_swap) peak_swap = $4
        if ($5 > peak_rss) peak_rss = $5
      }
      END {
        printf "peak_mem_used=%d MB  min_mem_avail=%d MB  peak_swap_used=%d MB  peak_single_proc_rss=%d MB\n", \
          peak_used, min_avail, peak_swap, peak_rss
      }'
    ;;
  *)
    echo "usage: $0 {sample [interval_s] [out.tsv] | summary [out.tsv]}" >&2
    exit 2
    ;;
esac
