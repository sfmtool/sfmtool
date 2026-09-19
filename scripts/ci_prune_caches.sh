#!/usr/bin/env bash
# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0
#
# Delete superseded generations of the CI cargo caches (see ci.yml).
#
# Usage: ci_prune_caches.sh KEEP_HASH [PREFIX ...]
#
# Every cargo cache key is `<prefix>-<Cargo.lock hash>`. KEEP_HASH is the
# current generation's hash. With PREFIX arguments (e.g.
# `Windows-cargo-test-target`) only those prefixes are considered -- that is how
# each job prunes its own caches right after saving them; with none, every
# `*-cargo-*` prefix is, which is the end-of-run sweep in the prune-caches job.
#
# Two guards keep this from eating anything load-bearing:
#
#   1. An old entry goes only once its prefix has an entry at KEEP_HASH. A save
#      that failed or was skipped therefore leaves the old generation alone.
#   2. An entry newer than that replacement is never touched, so a run cannot
#      delete caches written after the one it is keeping.
#
# Needs GH_TOKEN with `actions: write`, and REPO (owner/name).
set -euo pipefail

keep=${1-}
shift || true
# Empty when there is no lockfile, or when a job stopped before its restore step
# computed the key. Either way there is no current generation to measure against.
if [ -z "$keep" ]; then
  echo "no current-generation hash; nothing pruned"
  exit 0
fi

tsv=$(mktemp)
plan=$(mktemp)
trap 'rm -f "$tsv" "$plan"' EXIT
gh cache list --repo "$REPO" --limit 100 \
  --json id,key,createdAt,sizeInBytes \
  --jq '.[] | [.id, .key, .createdAt, .sizeInBytes] | @tsv' > "$tsv"

# Decide in awk rather than with bash associative arrays: macOS runners can
# resolve `bash` to the stock 3.2, which has none. Two passes over the listing:
# the first records each prefix's current-generation creation time, the second
# emits `keep <key> <reason>` or `delete <id> <key> <size>` for every other
# entry. Timestamps are fixed-width UTC ISO-8601, so string order is time order.
awk -F'\t' -v keep="$keep" -v wanted="$*" '
  function prefix(k) { sub(/-[^-]*$/, "", k); return k }
  function hash(k) { sub(/.*-/, "", k); return k }
  function considered(k,   n, i, w) {
    if (k !~ /-cargo-/) return 0
    if (wanted == "") return 1
    n = split(wanted, w, " ")
    for (i = 1; i <= n; i++) if (w[i] == prefix(k)) return 1
    return 0
  }
  NR == FNR { if (considered($2) && hash($2) == keep) live[prefix($2)] = $3; next }
  !considered($2) || hash($2) == keep { next }
  !(prefix($2) in live) { print "keep\t" $2 "\tno current-generation entry for this prefix"; next }
  !($3 < live[prefix($2)]) { print "keep\t" $2 "\tnewer than its replacement"; next }
  { print "delete\t" $1 "\t" $2 "\t" $4 }
' "$tsv" "$tsv" > "$plan"

freed=0
while IFS=$'\t' read -r action a b c; do
  if [ "$action" = keep ]; then
    echo "keeping  $a ($b)"
    continue
  fi
  echo "deleting $b ($((c / 1048576)) MB)"
  if gh cache delete "$a" --repo "$REPO"; then
    freed=$((freed + c))
  else
    echo "         already gone"
  fi
done < "$plan"

echo "freed $((freed / 1048576)) MB"
gh cache list --repo "$REPO" --limit 100 --json sizeInBytes --jq \
  '"remaining: \([.[].sizeInBytes] | add // 0 | . / 1073741824 * 100 | round / 100) GB"'
