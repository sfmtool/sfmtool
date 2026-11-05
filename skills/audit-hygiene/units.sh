#!/usr/bin/env bash
# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0
#
# List the audit units of the hygiene audit's deep read, with their line
# counts. Run from the repository root:
#
#   bash skills/audit-hygiene/units.sh          # "<lines>\t<unit>", one per unit
#   bash skills/audit-hygiene/units.sh --files  # "<unit>\t<lines>\t<file>", one per file
#
# The units are derived from the tracked files, so the inventory follows the
# tree. A start node is each crate under crates/, src/sfmtool, tests and
# scripts. A module is a file without its extension together with the
# directory of the same name (`foo.rs` with `foo/`, `foo.py` with `foo/`). A
# module of at most LIMIT lines is one unit. A larger one is split:
#
# - its own file, when that alone is over LIMIT, is the unit
#   "<module> (own file)";
# - its child modules of at least MIN lines become units, largest first (each
#   split again the same way when it is over LIMIT), until what is left totals
#   at most LIMIT lines;
# - what is left is the unit "<module> (rest)", or, when it is still over
#   LIMIT, the units "<module> (rest 1)", "<module> (rest 2)", ..., as few
#   as LIMIT allows and of about equal size, cut in name order.
#
# LIMIT defaults to 5000 and MIN to 1000; set them in the environment to
# change them.
set -euo pipefail

mode=units
[ "${1:-}" = "--files" ] && mode=files
limit=${LIMIT:-5000}
min=${MIN:-1000}

git ls-files -z -- 'crates/*.rs' 'crates/*.wgsl' 'src/sfmtool/*.py' \
    'tests/*.py' 'scripts/*.py' 'scripts/*.sh' 'scripts/*.ps1' |
  xargs -0 wc -l |
  awk -v LIMIT="$limit" -v MIN="$min" -v MODE="$mode" '
    # Sort a[1..n] by name, or by size (largest first, ties by name).
    function sort(a, n, bysize,   i, j, t) {
      for (i = 2; i <= n; i++)
        for (j = i; j > 1; j--) {
          if (bysize && size[a[j]] != size[a[j-1]]) {
            if (size[a[j]] < size[a[j-1]]) break
          } else if (a[j] > a[j-1]) break
          t = a[j]; a[j] = a[j-1]; a[j-1] = t
        }
    }
    function walk(node,   k, nk, i, m, ownsize, rest, left, part, parts, acc) {
      if (size[node] <= LIMIT || kids[node] == "") { unit[node] = node; return }
      nk = split(substr(kids[node], 2), k, SUBSEP)
      ownsize = size[node]
      for (i = 1; i <= nk; i++) ownsize -= size[k[i]]
      rest = size[node]
      if (ownsize > LIMIT) { own[node] = node " (own file)"; rest -= ownsize }
      sort(k, nk, 1)
      for (i = 1; i <= nk && rest > LIMIT && size[k[i]] >= MIN; i++) {
        walk(k[i]); rest -= size[k[i]]
      }
      if (rest <= LIMIT) {
        for (; i <= nk; i++) unit[k[i]] = node " (rest)"
        if (!(node in own) && ownsize > 0) own[node] = node " (rest)"
        return
      }
      m = 0
      for (; i <= nk; i++) left[++m] = k[i]
      sort(left, m, 0)
      # Each module goes to the part its middle line falls in, for parts of
      # equal size.
      parts = int((rest + LIMIT - 1) / LIMIT); acc = 0
      if (!(node in own) && ownsize > 0) { own[node] = node " (rest 1)"; acc = ownsize }
      for (i = 1; i <= m; i++) {
        part = 1 + int((acc + size[left[i]] / 2) * parts / rest)
        if (part > parts) part = parts
        unit[left[i]] = node " (rest " part ")"; acc += size[left[i]]
      }
    }
    $2 == "total" && NF == 2 { next }
    {
      n = $1; f = $0; sub(/^ *[0-9]+ /, "", f)
      np = split(f, p, "/")
      if (p[1] == "crates" || p[1] == "src") { node = p[1] "/" p[2]; i = 3 }
      else { node = p[1]; i = 2 }
      if (!(node in size)) roots[++nroots] = node
      size[node] += n
      for (; i <= np; i++) {
        c = p[i]
        if (i == np) sub(/\.[^.]*$/, "", c)
        child = node "/" c
        if (!(child in size)) kids[node] = kids[node] SUBSEP child
        size[child] += n
        node = child
      }
      nfiles++; file[nfiles] = f; fl[nfiles] = n; leaf[nfiles] = node
    }
    END {
      for (r = 1; r <= nroots; r++) walk(roots[r])
      for (x = 1; x <= nfiles; x++) {
        node = leaf[x]
        if (node in own) u = own[node]
        else {
          while (!(node in unit)) sub(/\/[^\/]*$/, "", node)
          u = unit[node]
        }
        total[u] += fl[x]
        if (MODE == "files") print u "\t" fl[x] "\t" file[x]
      }
      if (MODE == "units") for (u in total) print total[u] "\t" u
    }' |
  if [ "$mode" = files ]; then LC_ALL=C sort; else LC_ALL=C sort -t "$(printf '\t')" -k 2; fi
