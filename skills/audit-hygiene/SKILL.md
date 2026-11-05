---
name: audit-hygiene
description: Survey the codebase for organizational drift — oversized files, files that should be split or merged, misleading names, inconsistent naming conventions, duplicated doc comments, directory structures that hurt navigation — and for code that breaks the quality standards in specs/STANDARDS.md (points at infinity, Progress on long-running work, no panics on caller input, no Python wrappers over bindings, medians and quantiles from sfmtool_core::numeric, the smallest numeric precision the use justifies, one I/O shape per file format, viewer item visibility, Python names and lazy loading, Rust module files and *Options settings structs, license headers, rustdoc, MSRV and mcp-less builds, no documented exceptions, tests that can fail, tests that cost only what their assertions need). Use when the user asks to clean up, reorganize, or review codebase structure.
---

# Codebase hygiene audit

As a project grows, files get bigger, coherent designs stretch, conventions fork
at module boundaries, and names and comments drift away from contents. This skill
surveys the codebase (Python and Rust) for those smells and produces a
prioritized list of structural fixes.

## Scope

- Python under `src/sfmtool/` and `tests/`
- Rust crates under `crates/`
- Top-level layout (`scripts/`, `specs/`)

The code is about 700,000 lines (2026-11-04). Reading all of it in one run
takes too long, so the audit runs at two depths, as `audit-specs` does:

- **The whole tree, mechanically.** The mechanical checks below run over all
  of it on every run. They cost little, they are the baselines one snapshot
  compares with the next, and they acquit code that no one read in full.
- **A sample, in full.** A set of audit units, chosen under "Choosing the
  units to read" below and limited by a budget in lines, is read in full by
  subagents. Repeated runs cover the whole tree, and each report's coverage
  table records when each unit was last read.

## Choosing the units to read

### Audit units

An audit unit is a part of the tree small enough for one subagent to read in
full. The inventory is derived from the tracked files by
[`units.sh`](units.sh), so every run gets the same units from the same tree
and the units follow the tree as it changes:

```bash
bash skills/audit-hygiene/units.sh          # <lines>\t<unit>, one line per unit
bash skills/audit-hygiene/units.sh --files  # <unit>\t<lines>\t<file>, one line per file
```

It counts the tracked `.rs` and `.wgsl` files under `crates/`, the `.py`
files under `src/sfmtool/` and `tests/`, and the `.py`, `.sh` and `.ps1` files
under `scripts/`. `specs/` has no units: this audit checks it only
mechanically, and `audit-specs` reads it. The script starts from each crate,
`src/sfmtool`, `tests` and `scripts`. A module is a file and the directory of the same name (`foo.rs`
with `foo/`, `foo.py` with `foo/`). A module of at most 5,000 lines is one
unit. A larger one is split: its own file, when that alone is over 5,000
lines, is the unit `<module> (own file)`; its child modules of at least 1,000
lines become units, largest first and each split the same way when it is
over 5,000 lines, until what is left totals at most 5,000 lines; and what is
left is the unit `<module> (rest)`, or `<module> (rest 1)`, `(rest 2)`, …
of equal size in name order when it is still over 5,000 lines. A single file
over 5,000 lines is a unit by itself. The script's header comment has the
same rules, and `LIMIT` and `MIN` in the environment change the two sizes.
Keep them at their defaults, so that unit names stay the same from run to
run.

A unit's name is the module path without an extension. The parts of a
`(rest N)` unit are cut in name order, so a file added or removed before a
part moves the boundaries of that part and the parts after it; their coverage
rows are then a little less exact, and the churn rule below does not see the
files that moved in.

On 2026-11-04 (`8edb7c02`) the script listed 247 units with 702,520 lines,
from 850 to 8,878 lines each and about 2,800 on average. 242 units are
between 1,000 and 5,000 lines. Four single files are over 5,000 lines:
`crates/sfmtool-core/src/geometry/bundle_adjust/tests (own file)` (8,878),
`crates/sfmtool-core/src/bench/tests (own file)` (8,291),
`crates/sfm-explorer/src/track_view/body/tests (own file)` (6,604) and
`crates/sfmtool-core/src/geometry/bundle_adjust (own file)` (5,554). By
start node: `sfmtool-core` 98 units, `sfm-explorer` 73, `sfmtool-py` 13, the
format crates and the other small crates 16, `src/sfmtool` 17, `tests` 22 and
`scripts` 8. At the default budget of 15,000 lines, of which slow-test dives
take up to 3,000, reading every unit once takes from 47 to about 59 runs, and
more when pools 1 and 2 read a unit again before every other unit has been
read.

### Budget

The budget is in lines read, 15,000 by default. Use another figure when the
user asks for a deeper or a shorter pass, and state it in the report. Fill it
unit by unit in the order below: take each unit that still fits in what is
left of the budget and skip one that does not. The first unit is always
taken, even when it alone is over the budget.

### Selection order

The budget is filled in this order. Pools 1 and 2 and the dives are
evidence, not a sample.

1. **Units with mechanical hits from this run.** A kept hit is a hit that a
   mechanical check below kept after reading it: a kept points-at-infinity
   hit (check 9), a public function without a `Progress` or a
   `Progress::none()` inside a function with one in scope (check 10), a
   progress variant (check 11), a kept caller-input panic (check 12), a
   Python wrapper that repeats a binding (check 13), a bare `pub` field in the
   viewer (check 16), a kept median or quantile (check 18), a kept precision
   hit (check 19), a kept test that cannot fail (check 20), or a file of a
   duplicate pair from checks 3 and 4 that shares 20 or more lines. A hit is
   *new* when no finding of the previous report lists it. A unit is pulled in
   when it holds a new kept hit, or holds any kept hit and its last deep read
   is more than 30 days old or `never`. A unit whose kept hits are all known
   and whose last deep read is within 30 days is left to the other pools, so
   a known backlog does not take the budget run after run. Map a hit's file
   to its unit with the `--files` output. Order these units by the number of
   new kept hits, most first, then by the total number of kept hits.
   Check 21's slow tests do not pull in their unit; they go to the dives.
2. **Units with churn since their last deep read.** Lines added plus lines
   deleted in a unit's current files since the commit its coverage row
   records, at least half of its current line count. Order by that ratio,
   highest first. A unit with no row, or a row that says `never`, is left to
   pool 3.
3. **Slow-test dives**, up to 3,000 lines, under "Slow-test dives" below.
4. **A seeded random draw from the units read least recently.** Units never
   read first, then by the date of their last deep read, oldest first, in a
   random order within each date. Record the seed.

A unit the user names is read whatever the pools say, and goes first.

When pools 1 and 2 hold more than the budget, the order above decides which
are read. List the ones left out in the Sample block as *pulled in, not read:
over budget*, so the reader sees them and the next run pulls them in again
while their hits or churn remain. Do not exceed the budget to fit them; a
larger budget is the user's choice.

The commands, in Git Bash on Windows or bash on Linux and macOS (macOS has no
`shuf`: install GNU coreutils with `brew install coreutils` and write `gshuf`
for `shuf`). Read
the coverage table of the newest hygiene report that has one, or, when the
last one was retired with no successor, the one in git history:

```bash
w=$(mktemp -d); echo "$w"
bash skills/audit-hygiene/units.sh > "$w/units.tsv"          # <lines> <unit>
bash skills/audit-hygiene/units.sh --files > "$w/files.tsv"  # <unit> <lines> <file>
prev=
for f in $(ls reports/*-hygiene-audit.md 2>/dev/null | sort -r); do
  grep -q '^## Coverage$' "$f" && { prev=$(cat "$f"); break; }
done
if [ -z "$prev" ]; then  # the last report was retired with no successor
  c=$(git log -1 --format=%H --diff-filter=D -- 'reports/*-hygiene-audit.md')
  [ -n "$c" ] && f=$(git show --format= --name-only --diff-filter=D "$c" -- 'reports/*-hygiene-audit.md' | sort -r | head -1) &&
    prev=$(git show "$c^:$f")
fi
printf '%s\n' "$prev" | awk -F'|' '/^## Coverage$/ {f = 1; next} /^## / {f = 0}
  f && /^\| `/ {u = $2; d = $4; c = $5; gsub(/^ *`|` *$/, "", u); gsub(/ /, "", d); gsub(/ /, "", c)
                print u "\t" d "\t" c}' > "$w/coverage.tsv"
# Every current unit with its last deep read; a new unit is "never", a unit
# that is gone is dropped.
awk -F'\t' 'FILENAME == ARGV[1] {last[$1] = $2 "\t" $3; next}
  {print $2 "\t" $1 "\t" ($2 in last ? last[$2] : "never\t-")}' \
  "$w/coverage.tsv" "$w/units.tsv" > "$w/current.tsv"          # <unit> <lines> <date> <commit>
# The slow tests examined, from the same report.
printf '%s\n' "$prev" | awk -F'|' '/^## Slow tests examined$/ {f = 1; next} /^## / {f = 0}
  f && /^\| `/ {for (i = 2; i <= 7; i++) gsub(/^ +| +$/, "", $i); gsub(/^`|`$/, "", $2)
                print $2 "\t" $3 "\t" $4 "\t" $5 "\t" $6 "\t" $7}' > "$w/examined.tsv"
                # <test id> <seconds> <measured by> <date> <commit> <outcome>
cutoff=$(date -d '30 days ago' +%F 2>/dev/null || date -v-30d +%F)  # GNU date, else BSD date
```

An empty `coverage.tsv` means this is a first run; see "First run" below.
Pool 1. Write the kept hits of this run's checks, except check 21's, to
`$w/hits.tsv`, one per line as `<file>\t<new|known>`; a hit in a duplicate
pair is one line for each of its two files:

```bash
awk -F'\t' -v cut="$cutoff" 'FILENAME == ARGV[1] {unit[$3] = $1; next}
  FILENAME == ARGV[2] {last[$1] = $3; next}
  {u = unit[$1]; if (u == "") next; total[u]++; if ($2 == "new") fresh[u]++}
  END {for (u in total) if (fresh[u] > 0 || last[u] == "never" || last[u] < cut)
         print fresh[u] + 0 "\t" total[u] "\t" u}' "$w/files.tsv" "$w/current.tsv" "$w/hits.tsv" |
  sort -t "$(printf '\t')" -k1,1nr -k2,2nr -k3,3 > "$w/pool1.tsv"  # <new hits> <hits> <unit>
cut -f3 "$w/pool1.tsv"                                           # pool 1, in order
```

Churn since each unit's last deep read (pool 2). When the recorded commit is
no longer in the repository, as happens to a branch commit after a squash
merge, the last commit on `HEAD` on or before that date stands in for it:

```bash
awk -F'\t' '$3 != "never" {print $3 "\t" $4}' "$w/current.tsv" | sort -u |
while IFS=$'\t' read -r d c; do
  base=$c
  git cat-file -e "$c^{commit}" 2>/dev/null || base=$(git rev-list -1 --before="$d 23:59:59" HEAD)
  git diff --numstat --no-renames "$base" HEAD | awk -F'\t' -v d="$d" -v c="$c" '{print d "\t" c "\t" $1 + $2 "\t" $3}'
done > "$w/changes.tsv"
awk -F'\t' 'FILENAME == ARGV[1] {size[$1] = $2; last[$1] = $3 "\t" $4; next}
  FILENAME == ARGV[2] {unit[$3] = $1; next}
  {u = unit[$4]; if (u != "" && last[u] == $1 "\t" $2) changed[u] += $3}
  END {for (u in changed) if (2 * changed[u] >= size[u])
         printf "%.2f\t%d\t%d\t%s\n", changed[u] / size[u], changed[u], size[u], u}' \
  "$w/current.tsv" "$w/files.tsv" "$w/changes.tsv" |
  sort -t "$(printf '\t')" -k1,1nr -k4,4 > "$w/churn.tsv"         # <ratio> <changed> <lines> <unit>
cut -f4 "$w/churn.tsv"                                           # pool 2, in order
```

The draw (pool 4), then the fill. Write the units the user named and those of
pools 1 and 2 to `$w/pulled.txt`, one per line in that order (an empty file
when there are none). Set `DIVE_LINES` to the lines the dives read (0 before
they are done; at most 3,000, and at most what the pulled-in units leave of
the budget):

```bash
SEED=$RANDOM; echo "seed $SEED"                                # record SEED in the report
awk -F'\t' '{print ($3 == "never" ? "0000-00-00" : $3) "\t" $2 "\t" $1}' "$w/current.tsv" |
  shuf --random-source=<(yes "$SEED") | LC_ALL=C sort -s -t "$(printf '\t')" -k1,1 > "$w/draw.tsv"
BUDGET=15000; DIVE_LINES=0
awk -F'\t' -v B="$BUDGET" -v D="$DIVE_LINES" 'FILENAME == ARGV[1] {size[$2] = $1; next}
  FILENAME == ARGV[3] && !dived {dived = 1; used += D; print D "\tslow-test dives"}
  FILENAME == ARGV[2] {u = $0; why = "pulled in"} FILENAME == ARGV[3] {u = $3; why = "drawn"}
  u in seen {next} {seen[u] = 1}
  used == 0 || used + size[u] <= B {used += size[u]; print size[u] "\t" why "\t" u; next}
  why == "pulled in" {print size[u] "\tover budget\t" u}
  END {print used "\tlines read"}' "$w/units.tsv" "$w/pulled.txt" "$w/draw.tsv"
```

Run the fill once with `DIVE_LINES=0` to see what the pulled-in units leave,
do the dives, then run it again with the dives' lines. The same seed gives the
same draw from the same coverage table. Remove `$w` when the run is done.

### Coverage

Every report carries a `## Coverage` section with one row per unit in the
current inventory, in the inventory's order:

```
## Coverage

| Unit | Lines | Last deep read | Commit |
|---|---|---|---|
| `crates/sfmtool-core/src/geometry/bundle_adjust (own file)` | 5554 | 2026-11-04 | 8edb7c02 |
| `scripts (rest 2)` | 4547 | never | - |
```

Keep this format exactly: the commands above find the section by its
heading, and read a row as a backticked unit name, the line count, the date in
`YYYY-MM-DD` form or `never`, and the short commit or `-`. Put no other
backticked table rows in the section. Copy every row forward from the
previous report, write today's date and the commit read (`git rev-parse
--short=8 HEAD`) on the rows of the units read in full this run, update every
row's line count, add a unit that is new to the inventory as `never`, and drop
a unit that is no longer in it. A run whose deep read does not finish a unit
records it as not read. `current.tsv` already holds the copied rows, the
current line counts and the new and dropped units, so the table is written
from it and a list of the units read in full this run:

```bash
# $w/read.txt: the units read in full this run, one per line
awk -F'\t' -v t="$(date +%F)" -v h="$(git rev-parse --short=8 HEAD)" \
  'FILENAME == ARGV[1] {read[$0] = 1; next}
   {d = $3; c = $4; if ($1 in read) {d = t; c = h}
    print "| `" $1 "` | " $2 " | " d " | " c " |"}' "$w/read.txt" "$w/current.tsv"
```

### Slow-test dives

Check 21 lists the Python test phases over 5 seconds and the Rust tests over
60 seconds. Its slow tests do not pull their unit into pool 1, since some of
them sit in test files of 6,000 to 9,000 lines. Instead, once pools 1 and 2
are placed, each run examines the 3 slowest of them in a dive:

- **Ranking.** A Python test's time is its longest setup or call phase in
  check 21's `pytest --durations` output. A Rust test's time is 60 seconds,
  the least its "has been running for over 60 seconds" line in the CI log
  says; that log gives no per-test time. Rank by that time, slowest first, a
  Python test before a Rust test of the same time.
- **Recently examined.** Skip a test the slow-tests table records as examined
  in the last 30 days, unless its time has grown by more than half since that
  examination. Name the skipped tests in the Sample block.
- **The dive.** Read the test, its fixtures and the main code path it
  exercises, not the whole unit. Count the lines actually read against the
  budget, at most 3,000 for all of a run's dives together; when the dives
  reach that, stop and leave the rest for the next run. Time the test alone
  (`pixi run test -- <node id>`, or `pixi run cargo test -p <crate> <test
  name>`). Report whether it meets STANDARDS.md § "Tests cost what their
  assertions need", and if it does not, which of the standard's four ways
  (the smallest input that exercises the behaviour, one run for several
  tests, the call rather than the pipeline, a measurement outside the suite)
  would cut its time with every case and assertion kept. A dive never
  proposes fewer cases or weaker assertions.

The candidates and the dives, with `$out` the pytest output of check 21 and
`$w/rust_slow.txt` its `grep` of the CI log:

```bash
{ awk '/slowest durations/ {f = 1} f && ($2 == "setup" || $2 == "call") && $1 + 0 > 5 {
         if ($1 + 0 > t[$3]) t[$3] = $1 + 0}
       END {for (id in t) print t[id] "\tlocal pytest\t" id}' "$out"
  sed -E 's/.*test (.+) has been running for over 60 seconds.*/60\tCI test-linux-rust, over 60 s\t\1/' "$w/rust_slow.txt"
} > "$w/slow.tsv"                                                # <seconds> <measured by> <test id>
: > "$w/skipped.tsv"
awk -F'\t' -v cut="$cutoff" -v w="$w" 'FILENAME == ARGV[1] {when[$1] = $4; was[$1] = $2; next}
  ($3 in when) && when[$3] >= cut && $1 <= 1.5 * was[$3] {print > (w "/skipped.tsv"); next}
  {print}' "$w/examined.tsv" "$w/slow.tsv" |
  LC_ALL=C sort -t "$(printf '\t')" -k1,1gr -k2,2r -k3,3 | head -3 > "$w/dives.tsv"
cat "$w/dives.tsv"; echo "skipped:"; cat "$w/skipped.tsv"
```

### Slow tests examined

Every report carries a `## Slow tests examined` section after the coverage
table, one row per test examined in a dive that check 21 still lists:

```
## Slow tests examined

| Test | Seconds | Measured by | Examined | Commit | Outcome |
|---|---|---|---|---|---|
| `tests/test_solve.py::test_solve_global` | 41.2 | local pytest | 2026-11-04 | 8edb7c02 | finding "Share one solve across the solve tests" |
| `geometry::bundle_adjust::tests::large_problem` | 60 | CI test-linux-rust, over 60 s | 2026-11-04 | 8edb7c02 | meets the standard |
```

Keep this format exactly, as for the coverage table. Seconds is the time the
test was ranked by, so the next run compares like with like; Outcome is `meets
the standard` or `finding "<title>"`. Copy every row forward, replace the rows
of the tests dived this run, and drop the rows of tests check 21 no longer
lists. Write it from `$w/dived.tsv`, one line per dive as `<seconds>\t<measured
by>\t<test id>\t<outcome>`:

```bash
awk -F'\t' -v t="$(date +%F)" -v h="$(git rev-parse --short=8 HEAD)" \
  'FILENAME == ARGV[1] {slow[$3] = 1; next}
   FILENAME == ARGV[2] {row[$1] = $0; next}
   {row[$3] = $3 "\t" $1 "\t" $2 "\t" t "\t" h "\t" $4}
   END {for (id in row) if (id in slow) {split(row[id], r, "\t")
          print "| `" r[1] "` | " r[2] " | " r[3] " | " r[4] " | " r[5] " | " r[6] " |"}}' \
  "$w/slow.tsv" "$w/examined.tsv" "$w/dived.tsv" | LC_ALL=C sort
```

### First run

A run that finds no hygiene report with a coverage table, in the tree or in
git history, treats every unit as `never` read and no slow test as examined,
and selects by the same order and budget: pools 1 and 2 and the dives as
usual, then a draw from all the units. `current.tsv` already lists every
unit as `never` in that case.

## What to look for

### A. Structure

1. **Oversized files** — modules or source files that have grown past what their
   purpose justifies. Look at line count, but more importantly at whether the file
   holds multiple distinct concerns.
2. **Files that should be combined** — small files that fragment a single concern,
   or near-duplicates.
3. **Misleading names** — files whose name no longer describes what's inside (e.g.,
   grew to cover a second topic, or the original concept was renamed but the file
   wasn't).
4. **Directory-level smells** — directories that are flat when they should be
   grouped, grouped when they should be flat, or where the grouping no longer
   matches how the code is actually used.
5. **Dead or near-dead code** — modules referenced only by tests, commented-out
   blocks, `_foo_old.py`-style leftovers.

### B. Duplication

6. **Duplication against an abstraction that already exists.** The highest-value
   finding class in this repo's history, and it does not correlate with file size.
   Two shapes:
   - A shared helper exists and callers hand-inline it anyway (`numeric::median`
     re-implemented in `resect_images.rs`; `parse_patch_window` inlined in five of
     eight bindings).
   - **Sibling files that are 50% byte-identical**, each too small to trip a
     size sweep. Compare parallel families explicitly: same-named files across
     sibling module directories, per-command modules, per-backend modules.
   Do not rely on the size overview to surface these — run a dedicated pass
   (mechanical checks 3 and 4 below).
7. **Invariants maintained only by a comment.** A doc comment saying *mirrors X* /
   *must match Y* / *kept in sync with Z* / *local copy of W* is the code admitting
   a duplication with no enforcement behind it. Each one is either a finding or a
   deliberate, justified copy — read it and decide, don't assume either way. The
   fix for a finding is usually one shared constant plus a test, not a rewrite.

   A comment names only the copies its author knew about. Before stating how
   many copies there are, search for the duplicated **content** itself (the
   variant list, the literal, the formula) across the workspace, the bindings
   included, and cite the count that search found. A function that already
   holds the same content under some name is the shared home to propose, and
   its name is the one to use. A 2026-10-08 finding that proposed
   `has_bare_focal` counted two copies of a camera-model list by following a
   *mirrors* comment; there were four, one of them already a public function
   named `focal_is_releasable`.

   **Check a proposed name against everything it would classify.** A name for
   a new shared predicate, constant or set is a claim about which items belong.
   Enumerate the whole domain (every variant of the enum, every registered
   model) and confirm the name is true of exactly the items in the list and
   false of every other. If a natural-sounding property also holds for items
   outside the list, it is the wrong name: name the reason the list exists,
   which is usually stated in the comment at the copy you are reading.
   `has_bare_focal` failed this check: `SIMPLE_RADIAL`, `RADIAL` and
   `RADIAL_FISHEYE` also have a single focal, but bundle adjustment does not
   release it.

   **Put the shared item in the module whose reasoning it encodes, not on the
   type it lists.** A set of enum variants is often a fact about an
   algorithm (which models a solver can handle, which formats a reader
   supports), not a fact about the enum. Read why the list exists. If the
   reason is in an algorithm's code (a derivative, a kernel's slots, a
   format's fields), the shared home is in that algorithm's module, and the
   enum's own module stays free of it. A method on the enum is right only when
   the property holds for the type whatever code uses it. The release list
   above is a property of the bundle adjustment kernel's analytic focal
   column, so it belongs in `geometry::bundle_adjust`, not on `CameraModel`.
   Code on the type that copies the algorithm's list (a setter that refuses
   the models the algorithm does not release) is a second finding: fix that
   code to state only the type's own fact.

### C. Naming and convention consistency

**Read [`specs/GLOSSARY.md`](../../specs/GLOSSARY.md) before running any check in
this section, and treat it as authoritative.** Where it names a preferred word,
that is the answer no matter which spelling the tree currently holds more of,
and a file still on the old word is a finding rather than evidence. The tallies
below are for splits the glossary has **not** ruled on.

This matters because a majority is wrong often enough to be dangerous, in two
ways the count itself cannot distinguish:

- **A word can win by being written first.** In the bench, `surfel` outnumbered
  `patch` 132 to a handful, and `patch` was the right answer -- it is the
  PatchMatch term and it already carries the geometry, which is an argument a
  tally cannot see. Converging on the majority would have deleted the correct
  name.
- **A migration reads exactly like drift.** The em-dash is still the majority
  across `specs/` by roughly seven to one, while the files recent work has
  rewritten carry none at all. A tally recommends putting them back. Note also
  that what replaces an em-dash is a varied sentence rather than one substitute
  character, so this split cannot be scored as a two-way spelling contest in the
  first place: some conventions are not countable, and a tally that forces them
  into a count will report something false with a number attached.

So when a tally and the glossary disagree, the glossary wins; when a tally finds
a split the glossary is silent on, the finding is *"decide this and record it"*,
and the recommendation names the candidates and the argument for each rather
than just the bigger pile. Add the ruling to the glossary as part of the fix.

8. **Conventions that fork at a module boundary.** A convention held uniformly on
   both sides of a line but differing across it: `*Options` in one module vs
   `*Params` in its siblings; underscore-private file names in one subpackage and
   plain names in the next; SPDX headers on some test files and not others; error
   messages capitalized in one crate and lowercase in another. Each half looks
   self-consistent from inside, which is why these survive review. Check the
   glossary first; failing a ruling there, state the house majority, say which
   side is newer, and recommend on the argument rather than the count alone.
   A boundary the glossary declares **deliberate** -- `patch` in the bench and
   `surfel` in the renderer -- is not a finding, and eroding it from either side
   is. Where `specs/STANDARDS.md` already states the convention, the minority
   is a breach of that standard, not a split to decide: settings-struct
   suffixes (§ "A settings struct is named *Options"), `mod.rs` against
   `foo.rs` (§ "A Rust module with submodules is a file beside its
   directory"), `_`-prefixed Python names (§ "Python names mark what is
   public") and license headers (§ "Every source file carries the license
   header").
9. **Names in a parallel family that don't parallel.** Sibling commands, sibling
   transforms, sibling bindings methods: check that the family's entry point,
   suffix, and argument names follow one rule, and name the exceptions.
10. **Implementation detail leaking into a public name.** Language-boundary
    suffixes (`_py`, `_rs`), internal type names, or a private module's vocabulary
    surfacing in a user-facing API — a Python keyword argument, a CLI flag, an
    on-disk key. These are the most expensive to fix later, so flag them early
    and small.
11. **The same concept under two names in one API.** Two spellings for one thing
    (`indexes`/`indices`, `view_indices`/`member_views`, `load_`/`read_`) matter
    most where they meet the user; inside one function body they usually don't.
    Weight findings by how public the name is.

### D. Comment and doc economy

12. **Doc bloat concentrated in one layer.** Compare doc-lines-to-code-lines per
    crate, and per item for wrapper layers. A binding or wrapper whose docs
    outweigh a parameter reference is usually carrying the layer below it a second
    time.
13. **A doc comment that restates its callee's doc comment.** A `sfmtool-py`
    method re-deriving the argument its `sfmtool-core` function already makes; one
    `impl` block repeating the module doc above it. Same finding class as check 6,
    in prose. Whether a doc block restates a **spec** is `audit-specs`' finding,
    not this one — it means reading the spec, so leave it there.

Do not flag ordinary API reference prose. A long `Args:` block on a
Python-visible binding is doing a job — a REPL user cannot click into `specs/`.
The finding is rationale where reference belongs, not length by itself.

### E. Quality standards

Check every standard in [`specs/STANDARDS.md`](../../specs/STANDARDS.md).
That file holds each rule, its reason, what to look for, how to confirm a
finding before reporting it and what is out of scope; this section adds only
how this audit runs them.

- **Check per function, not per convention split.** A breach is a finding
  even where the whole module is consistent with itself. Run the
  mechanical checks 9–21 below, and the standards' tallies in check 5,
  first, then read each hit against the standard it feeds.
- **Cite the standard.** Each finding names the standard by its heading
  and links it (`specs/STANDARDS.md#caller-input-never-panics`).
- **Propose the code fix.** Per STANDARDS.md § "Direction of a fix", the
  proposed fix for a breach changes the code to meet the standard. It is
  never a spec or doc edit that describes the breach. When the code fix is a
  large design change, say so under Effort and Risk and leave the choice to
  the maintainer, rather than proposing the doc edit instead.

## How to work

1. Get a size overview: list the audit units with `units.sh`, and file line
   counts per directory, the longest files and the longest functions over the
   whole tree.
2. Run the mechanical checks below and keep the raw numbers — they are the
   report's evidence and the next snapshot's baseline.
3. Read the previous report's coverage and slow-tests tables and choose what
   to read, per "Choosing the units to read" above: pool 1 from step 2's
   kept hits, pool 2 from the churn command, up to three slow-test dives from
   check 21, then the seeded draw, filled up to the budget. Write the
   selection down before reading anything; the report's Sample block names
   it.
4. Dispatch `Agent` subagents in parallel, one per selected unit or one per
   few small units, each given the unit's file list from `units.sh --files`,
   and one per slow-test dive, given the test id and "Slow-test dives" above.
   Each unit's subagent reads its files in full and assesses them against
   "What to look for": for a long file, how many distinct top-level concerns
   it holds; for a family of siblings, whether they parallel; and every
   standard. Give each one `specs/GLOSSARY.md` and the convention majorities
   from step 2, in that order of authority, so their naming findings are
   comparable and none of them recommends converging onto a word the glossary has already retired.
   Give each one `specs/STANDARDS.md` as well, so every unit is checked
   against the same standards, and the previous report's findings whose
   Location lies in its unit, so it re-measures them.
5. Carry the previous report's findings forward, per "Carried findings"
   under Output.
6. Consolidate findings, removing duplicates and ranking.

### Mechanical checks

Cheap, repeatable, and comparable across snapshots. Run them, cite the numbers,
and note the ones that came back clean — an acquittal with a number behind it is
worth more than a paragraph.

1. **Doc density** — doc lines vs total lines per crate, and the longest
   contiguous comment blocks in the tree. Feeds checks 12–13.
2. **Doc-to-code ratio per item** — for a wrapper layer, the doc line count
   against the function body line count. Anything over ~0.5x deserves a read.
3. **Cross-file duplicate prose and code** — normalize long lines (lowercase,
   strip markup) and bucket them; report file pairs sharing many. Catches
   sibling-file duplication and callee-restating docs in one pass. Code only;
   `audit-specs` runs the same scan across `specs/`.
4. **Parallel-family diff** — for each set of same-named files across sibling
   directories (`*/prof.rs`, `_commands/*.py`, `patches/*.rs`), count identical
   lines pairwise.
5. **Convention tallies** — count both spellings of each candidate convention and
   report the split with locations of the minority: option-bag type suffixes,
   `indexes`/`indices`, verb prefixes on public functions, private-module naming,
   license headers, error message capitalization, CLI flag shapes. Report the
   minority's **age** beside its size -- if the newer files hold it, the split is
   a migration and the majority is the thing to fix. Check each split against
   `specs/GLOSSARY.md` before recommending either way. Always report three
   tallies with the commands STANDARDS.md gives for them: `mod.rs` files
   against modules in the `foo.rs`-beside-`foo/` form, feeding § "A Rust
   module with submodules is a file beside its directory"; public settings
   structs in `sfmtool-core` still ending in `Params` or `Config`, feeding
   § "A settings struct is named *Options"; and plain-named modules in the
   re-exporting subpackages and in `_commands/` that define no command,
   feeding § "Python names mark what is public".
6. **Public-name leak scan** — every name reachable from Python or the CLI,
   checked for implementation-detail suffixes and for a spelling that disagrees
   with its siblings.
7. **Reference integrity** — every `specs/…` path cited from code resolves to a
   file, and every area's `specs/README.md` index lists the specs actually
   present. Link rot only, no reading; whether the linked spec is *right* is
   `audit-specs`' call.
8. **Glossary conformance** — for every entry in `specs/GLOSSARY.md`, count uses
   of the preferred word and of what it replaced, **inside that entry's stated
   scope only**. A word outside its scope is not a violation: the glossary's
   boundaries are entries too, so `surfel` in the renderer is a pass and
   `surfel` in the bench is a finding. Report the residue per file, and report
   the entries that came back clean -- a retired word with a zero behind it is
   how the glossary earns its keep. An entry whose residue is large and *newer*
   than the ruling is the one case where the glossary itself is the suspect:
   say so rather than filing a hundred findings.
9. **Points-at-infinity scan** — grep library code for finite-only point
   interfaces and unguarded homogeneous divisions: `positions_xyz\b`,
   `\[:, *:3\]` on an xyzw array, `/ *w\b`, and public signatures taking
   `[f64; 3]` / `Vec3` slices of reconstruction points. Report the hit count
   per pattern, then read each hit and keep only those with no `w = 0`
   handling (`is_at_infinity`, an explicit branch, a documented finite-only
   contract with a visible skip count). Feeds STANDARDS.md § "Points at
   infinity are supported".
10. **Progress coverage** — list public functions that loop over a whole
    reconstruction, run a solver or read or write a whole file, and report how
    many take a `&Progress`. Separately count `Progress::none()` in non-test
    library code and read each one inside a function that itself has a
    `Progress` in scope. Feeds STANDARDS.md § "Long-running functions
    take a Progress".
11. **Progress variants** — grep for `_with_progress`, `_without_progress`,
    `_no_progress`, `_silent`, `_quiet`, `Option<&Progress`, and
    `verbose: bool` / `show_progress` parameters in library code. Each hit is a
    candidate for STANDARDS.md § "One interface, no progress variants";
    report the count even when it is zero.
12. **Caller-input panics** — apply the scope in STANDARDS.md § "Caller
    input never panics". Count `assert!`, `.expect(`, `.unwrap()`, `panic!`
    and `from_row_slice` per crate, leaving out test code (`tests/`
    directories, `tests.rs` files, inline `#[cfg(test)]` modules) and
    `debug_assert!`. In `sfm-explorer` count only `cli/`, `mcp/` and the
    code that parses what it reads from a file: find that code by grepping
    non-test files for `File::open`, `fs::read`, `serde_json::from` and the
    `read_*` format readers, and follow the parsed data from each hit.
    Count `# Panics` doc sections the same way. Then read the hits in
    bindings and in public functions of the library crates that take caller
    data, and keep those a caller's argument can reach. Also grep specs for
    "panics". Report the kept hits as one ranked finding that lists the
    functions, those Python can reach through a binding first, then the rest
    of the public Rust API, each with the argument that reaches the panic.
    Report the raw counts beside it so the next snapshot can measure the
    backlog shrinking.
13. **Python over bindings** — list the Python functions and classes in
    `src/sfmtool/` that call a `_sfmtool` name, and the root `_LAZY_NAMES`
    entries, and read each for re-checked arguments, re-read files or a body
    that only forwards. Feeds STANDARDS.md § "Logic lives in Rust, not in
    Python wrappers".
14. **Format I/O table** — one row per on-disk format, one column per
    function that STANDARDS.md § "Every file format has the same I/O shape"
    lists, and the module each lives in. Report the table even when every
    row is full.
15. **Documented exceptions** — grep `AGENTS.md`, `specs/` and doc comments
    for "the exception", "exception to", "except for", "unlike the others"
    and "the one … that does not" (``the one [a-z_` ]* that (does|is) not``;
    a bare "the one" matches over a thousand lines).
    Feeds STANDARDS.md § "A convention has no documented exceptions".
16. **Viewer visibility** — the `unreachable_pub` lint checks every bare
    `pub` item in `sfm-explorer` except struct fields and code compiled only
    on Windows. Grep `crates/sfm-explorer/src/` for bare `pub` fields
    (`^\s*pub [a-z_][a-z0-9_]*\s*:` and tuple fields `(pub `), leaving out
    `CatalogTool`'s fields, which the crate's public interface exports, and
    grep the Windows-only code (`platform/windows.rs` and
    `#[cfg(target_os = "windows")]` blocks) for any bare `pub`. Report the
    count even when it is zero. Feeds STANDARDS.md § "Viewer items are no
    more visible than their users need".
17. **Build and import gates** — tests and CI check four standards, so this
    check confirms the gates are still in place rather than re-deriving
    them: every `crates/*/Cargo.toml` has `[lints] workspace = true`
    (§ "Rustdoc builds without warnings"); `pixi run cargo check -p
    sfm-explorer --no-default-features` passes with no warning (§ "Code
    builds at the MSRV and without the viewer's MCP feature");
    `tests/test_lazy_loading.py` and `tests/test_module_layout.py` pass
    (§ "Importing sfmtool and sfm loads only what is used", § "Python names
    mark what is public"); and `pixi run cargo test -p sfmtool-core --test
    license_headers` passes (§ "Every source file carries the license
    header"). Then grep the examples in `specs/` for imports of
    `_`-prefixed `sfmtool` names, which no test scans, and the command
    modules for module-level work beyond imports and defining the command.
    Report each gate as passed or failed.
18. **Medians and quantiles** — `pixi run cargo test -p sfmtool-core --lib
    numeric` runs the name scan, so report it as passed or failed. Then grep
    `crates/` for what the scan cannot see, leaving out `numeric.rs` and test
    code (`tests/` directories, `tests.rs` files, inline `#[cfg(test)]`
    modules): an inline median, `select_nth_unstable` and
    `\[\s*([a-z_.]+\.len\(\)|n|len)\s*/\s*2\s*\]`; and a hand-written
    percentile, an index from a fraction times a length
    (`(\.len\(\)|\bn)\s*(-\s*1\s*)?\)?\s*as f64[^;]*as usize`) and an
    interpolation position (`\* *\(?(\w+\.len\(\)|n) *- *1\)? *as f64`).
    Report the hit count per pattern, then read each hit and keep those that
    compute an order statistic of a population. Feeds STANDARDS.md
    § "Medians and quantiles come from one module".
19. **Numeric precision** — grep library code under `crates/` for `f64` in
    pixel and sample loops: `as f64 / 255`, `f64::from(` of a `u8` sample,
    and `vec![0f64` / `vec![0.0f64` / `Vec<f64>` buffers in image, patch,
    tile, flow and feature code, and `= 0.0f64` accumulators in those loops;
    for whole-image sums in one running `f32` accumulator (`.sum::<f32>()`
    over every pixel); and for float reductions whose order rayon chooses
    (`.sum()` or `.reduce(` on a `par_iter`) where the result is stored or
    compared. Report the hit count per
    pattern, then read each hit and keep those whose precision has no stated
    reason or whose stated reason the values' range, the digits the use needs
    and the accumulation error do not support. Feeds STANDARDS.md § "Numeric
    precision is the smallest the use justifies".
20. **Tests that cannot fail** — grep the test code with `rg -nP` for the
    shapes STANDARDS.md § "Tests assert something that can fail" lists: an
    `is not None` (`assert .* is not None` in `tests/`) or its Rust form
    (`assert!\(.*\.is_some\(\)\)` in `crates/`, keeping the hits in test
    code); a length or a value checked against itself
    (`assert len\(([\w.\[\]]+)\) == len\(\1\)` and
    `assert ([\w.\[\]]+) == \1\s*$` in `tests/`,
    `assert_eq!\(([\w.]+), \1\)` in `crates/`); and work held, paused or
    ordered in a test (`Barrier|\bsleep\(|threading\.Event|\.wait\(` in
    `tests/`, and `Barrier|thread::sleep|Condvar|\.wait\(` in `crates/`,
    keeping the hits in test code). In `crates/`, test code is `tests.rs`
    files, `tests/` directories and the inline `mod tests { … }` blocks of
    other files, so a path glob alone misses some of it. Report the hit
    count per pattern, then read each hit. Keep an `is not None` or `is_some()` only where the
    value cannot be `None`, and a hold or ordering only where it also decides
    the result the test's key assertion checks. The other two shapes, a value
    compared with the result of the same call and a fixture that already
    satisfies the assertion before the code under test runs, have no
    pattern; look for them while reading these hits and the tests that
    check 21 picks out. Before reporting a test, name the change to the
    guarded code under which it still passes. The proposed fix makes the
    assertion able to fail, or covers each ordering with its own
    deterministic test; it never deletes a case. Feeds STANDARDS.md § "Tests
    assert something that can fail".
21. **Test time** — run the Python suite with every duration listed, then
    list and sum the setup and call phases over 5 seconds:

    ```bash
    out=$(mktemp)
    pixi run test -- --durations=0 > "$out" 2>&1
    awk '/slowest durations/ {f=1} f && ($2 == "setup" || $2 == "call") && $1 + 0 > 5 {n++; s += $1; print} END {printf "%d phases, %.1f s\n", n, s}' "$out"
    ```

    The `test` task in `pixi.toml` runs `pytest -n auto --dist worksteal`,
    so these are the times of a normal `pixi run test`, the run the
    standard's 5-second limit is stated for. For Rust, read the "has been
    running for over 60 seconds" lines of the `test-linux-rust` job in the
    latest successful CI run on `main`:

    ```bash
    run=$(gh run list --repo sfmtool/sfmtool --workflow ci.yml --branch main --status success --limit 1 --json databaseId --jq '.[0].databaseId')
    job=$(gh run view "$run" --repo sfmtool/sfmtool --json jobs --jq '.jobs[] | select(.name == "test-linux-rust") | .databaseId')
    gh run view --repo sfmtool/sfmtool --job "$job" --log | grep "has been running for over 60 seconds"
    ```

    Report the number of Python phases over 5 seconds with their summed
    seconds, and the number of Rust tests over 60 seconds, even when they
    are zero, so the next snapshot can compare. Whole-suite times can differ
    by up to a factor of two between runs, so compare per-test times with
    the last snapshot, not the suite's wall time. The limits only pick tests
    for a closer look; a test over them is not a finding by itself. Each run
    reads three of them, chosen under "Slow-test dives" above, and reports
    the rest by their numbers alone. For each one read: group the tests that run the same expensive call on the same
    input, and note a test that lowers a size for the steps it checks but
    leaves an expensive step at its default. Keep only a test that one of
    the standard's ways (the smallest input that exercises the behaviour,
    one run for several tests, the call rather than the pipeline, a
    measurement outside the suite) would make faster with every case and
    assertion kept. Before reporting it, time it alone (`pixi run test --
    <node id>`, or `pixi run cargo test -p <crate> <test name>`), and name
    the way and how each of its cases and assertions is kept. A proposed fix
    never drops a case, weakens or removes an assertion, or narrows what the
    test covers. A test over the limits that no way would make faster
    without doing one of those meets the standard and is not a finding.
    Feeds STANDARDS.md § "Tests cost what their assertions need".

## Output

Open with a **Sample** block, so a reader knows what this run read in full
and what it did not:

```
**Sample:** N of M units read in full, L of T lines (units and dives),
against a budget of B lines. Seed `<seed>`; the draw's pool was <never read |
last read on <date>> (P units). Pulled in by mechanical hits: <unit — new and
total kept hits, the checks they came from, per unit>. Pulled in by churn:
<unit — lines changed / lines, per unit>. Named by the user: <units>.
Slow-test dives: <test id — seconds, measured by, lines read, per test>;
skipped as examined in the last 30 days: <test ids, or none>. Drawn: <units>.
Pulled in, not read, over budget: <units, or none>. The mechanical checks
below cover all M units.
```

Then the mechanical numbers, then one section per recommendation:

```
**<short title>**
- Location: <file or dir path>
- Problem: <specific smell — "this 1200-line file mixes matching, filtering, and I/O">
- Proposed fix: <split into X and Y | merge with Z | rename to W | regroup under foo/>
- Effort: <low | medium | high>
- Risk: <low | medium | high> — <what could break>
```

For a breach of a standard in `specs/STANDARDS.md` or another rule in
`AGENTS.md`, the proposed fix changes the code and names the standard; it is
never a spec or doc edit that records the breach as how the code works.

Then a **Top 3** section: the fixes with the best effort-to-value ratio. End
the report with the `## Coverage` section and then the
`## Slow tests examined` section, in the formats under "Coverage" and "Slow
tests examined" above.

### Carried findings

Every still-open finding of the previous report (one with no status line, or
marked `Partially done`, `Not done` or `Needs decision`) appears in the new
one, with its status lines. A finding is placed in the units that hold the
files of its Location, found with the `--files` output.

- **In a unit read this run**, the subagent reading it re-measures the
  finding as the code is now, and the report states it in its current form
  with no marker.
- **In a unit not read this run**, it is carried forward with the date of the
  last deep read of its unit in its title line:
  `**<short title>** (carried, last read 2026-10-08)`. Its numbers are still
  re-measured: re-run the mechanical check, grep or line count it cites at
  the current `HEAD`, and update them. When its location spans several
  units, it is re-measured when any of them is read, and carries the oldest
  date otherwise.
- **A finding the mechanical numbers show fixed** (the hit is gone, the count
  is zero, the file was split) is closed without a deep read. List its title
  under a **Closed since the last snapshot** heading with a status line
  giving the number that closed it, `> _Status (<today>): **Done** — <the
  number>._`, so `fix-report-findings` does not pick it up.

Never write that a unit is clean because it was not read. A unit not read this
run is acquitted only by the mechanical numbers, and the report says which
ones.

### The "Explicitly not flagged" list

Carry one, but treat it as a measurement, not a verdict. Every entry states the
figure and the date it was taken, and the section opens by saying so: this repo
has repeatedly grown a cleared file 30–90% within a month of clearing it. Prefer
acquitting with a mechanical number (longest function, duplicate-line count) that
the next snapshot can re-run, over a prose judgement it would have to re-derive.
A prose judgement comes only from a unit read in full this run, and names it.

## Saving the report

Write the full report to `reports/<date>-hygiene-audit.md`, where `<date>` is
today's date in `YYYY-MM-DD` form (get it with `date +%F`). Create the
`reports/` directory if it doesn't exist. Each run is a dated snapshot — do not
overwrite a prior day's report. Carry forward every still-open finding from the
report being superseded, per "Carried findings" above, and its coverage and
slow-tests tables, per "Coverage" and "Slow tests examined". The new report
then holds everything the old one tracked, so it supersedes it: retire the
old snapshot in the same commit, per the rules in
`AGENTS.md`. When a hygiene report is retired with no successor, the next run
reads its tables back from git history. After saving, tell the user
the path and give a short summary of the Top 3 in the conversation; don't paste
the whole report back.

## Guidelines

- Be specific. "X is too big" is useless; "X is 900 lines covering matching AND
  geometric filtering AND serialization — split serialization into a sibling
  file" is useful.
- Cite line counts or symbol counts when flagging size; cite both sides of the
  split when flagging a convention.
- Don't flag a file just because it's long — long is fine if the file has a single
  coherent purpose.
- Don't flag a difference just because it's a difference. Check whether the
  minority spelling is carrying meaning before calling it drift: a batched binding
  legitimately pluralizes its core counterpart's name, `estimate_` and `compute_`
  legitimately distinguish fitting from evaluation, and a suffix may be domain
  notation rather than a language tag. Read the item before reporting it.
- When a convention is real but unwritten, say so, and propose where it should be
  written down (`AGENTS.md`, the area's `specs/README.md`). Undocumented
  conventions are how the next module forks one.
- Prefer fixes that end in enforcement — a shared constant, a compile-time assert,
  a clippy `disallowed_methods` entry, a grep-based test — over fixes that end in
  a doc comment. This repo has a demonstrated pattern of re-breaking contracts
  that live only in prose.
- Never imply the sample was the tree. Say "N of M units read in full"
  wherever it could be misread; the mechanical checks acquit a unit, the
  sample's silence does not.
- Do not modify any code during this skill — it is read-only analysis.
