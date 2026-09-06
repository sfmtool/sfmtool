# Why `test-linux` is the slow job, and what actually fixes it

**Question asked:** the Linux coverage job is now the critical path (1920 s on
[#384](https://github.com/sfmtool/sfmtool/pull/384)). The hypothesis on the table was
that `rayon` × the `llvm-cov`-instrumented build is what makes it slow, and that the fix
is to stop handing rayon tiny chunks:

```rust
my_vector.par_iter().with_min_len(5000).map(expensive).collect::<Vec<_>>()
```

**Answer:** the interaction is real and large — but `with_min_len` does not touch it. The
cost is contention on *shared coverage counters* between threads running the same
instrumented code, and it scales with thread count, not with chunk size. The lever that
works is turning rayon **down** for the instrumented run: measured −29 % on the pytest
phase and −21 % on the Rust phase, coverage-neutral. Separately, #384's 1920 s was a cache
miss rather than a regression; warm, the job is ~1030 s.

Everything below is measured. Local numbers are an i9-14900HX with the process pinned to
logical CPUs 0–3 to approximate the 4-core runner; CI numbers are read out of the workflow
logs.

---

## 1. Where the time goes

Warm-cache `test-linux`, from the step log of run 34003866655 (1027 s), confirmed by
34010579983 (1028 s):

| phase | s | share |
| --- | ---: | ---: |
| checkout + cache restore + `setup-pixi` | 55 | 5 % |
| `maturin develop --release` (incremental) | 94 | 9 % |
| `cargo test` build | 169 | 16 % |
| Rust test execution | 103 | 10 % |
| **`pytest -n 4 --cov`** | **594** | **58 %** |
| `llvm-cov report` | 5 | — |

Two thirds of any fix has to come out of the pytest phase.

### #384's 1920 s was a cold compile, not a slow test run

```
Cache not found for input keys: Linux-cargo-coverage-target-777c…, Linux-cargo-coverage-target-
```

The exact-key entry was written by the concurrent `main` run **7 s later** (03:39:01
against the 03:38:54 restore), and the `restore-keys` prefix fallback matched nothing
because the previous entry had already been LRU-evicted. The repo currently holds
**9.56 GB of the 10 GB** per-repo cache budget across 14 entries, so every save evicts
something and this recurs. Cold, the two builds cost 475 s + 721 s instead of 94 s + 169 s
— the whole 900 s difference between 1920 s and 1030 s.

That is worth more than any parallelism tuning on the runs where it bites, and it is
independent of everything else here.

---

## 2. What instrumentation costs

Same runner class (4-core x64), same suite, from CI:

| job | pytest phase | build |
| --- | ---: | --- |
| `test-os (windows-latest)` | 276 s | uninstrumented, no `--cov` |
| `test-linux` | 594 s | `-C instrument-coverage` + `--cov` |

**2.15× on comparable hardware.** (`test-os (macos-latest)` runs it in 188 s, but on a
3-core arm64 runner, so it is not like-for-like.)

Locally, isolating the extension build with everything else held fixed
(`pytest -n 4 --cov`, pinned to 4 logical CPUs):

| extension | `RAYON_NUM_THREADS` | wall |
| --- | ---: | ---: |
| uninstrumented | 4 | 302.7 s |
| instrumented | 4 | 1205.6 s |

**4.0×.** Collecting Rust coverage through the Python suite is the single most expensive
thing this job does.

---

## 3. The mechanism, and why chunk size cannot reach it

`-C instrument-coverage` emits one counter per coverage region into a process-wide
`__llvm_prf_cnts` section, incremented by plain non-atomic loads and stores. Two
consequences follow:

1. **The number of increments is fixed by the data**, not by how the work is divided.
   `par_chunks_mut(2)` and `par_chunks_mut(5000)` execute the closure body the same number
   of times, so they write the counter the same number of times. Chunk size changes *which
   thread* runs an item, never how much counter traffic exists.
2. **Every thread in a data-parallel loop increments the same address.** A hot loop that
   was thread-private without coverage becomes a single cache line bouncing between cores
   under it. This is the part that scales with thread count.

`with_min_len` reduces rayon's split/join count. That is real work, but it is the *wrong*
work: cargo-llvm-cov 0.9 instruments only this workspace's crates —
`__CARGO_LLVM_COV_RUSTC_WRAPPER_CRATE_NAMES` names exactly the nine crates plus their build
scripts — so rayon is compiled **without** `-C instrument-coverage`. Its splitting
machinery is not instrumented at all. Shrinking it shrinks the cheap half.

The prediction that separates the two theories: if it is counter contention, then holding
chunk size fixed and lowering the thread count should recover the time. It does.

### Rust lib tests, pinned to 4 logical CPUs

`sfmtool-core --lib`, identical source, only the build and the two thread knobs varying:

| build | `RAYON_NUM_THREADS` | `--test-threads` | wall |
| --- | ---: | ---: | ---: |
| uninstrumented | 1 | 4 | 14.8 s |
| uninstrumented | 4 | 4 | **9.6 s** |
| instrumented | 4 | 4 | 94.6 s ← what CI runs |
| instrumented | 2 | 4 | **74.6 s** |
| instrumented | 1 | 4 | 76.7 s |
| instrumented | 4 | 1 | 100.8 s |
| instrumented | 1 | 1 | 84.5 s |

Uninstrumented, rayon is worth 1.5×. Instrumented, **rayon is a 23 % loss** — the parallel
speedup is more than consumed by counter traffic. Unpinned across all 32 logical CPUs the
inversion is starker: the instrumentation tax rises from **2.0×** single-threaded to
**14.7×** at `--test-threads=4 RAYON_NUM_THREADS=32` (6.3 s → 92.5 s), because instrumented
code barely parallelizes at all — 85.6 s → 71.5 s across the whole sweep, against 41.9 s →
6.3 s uninstrumented.

### Python suite, pinned to 4 logical CPUs

`pytest -n 4 --cov=sfmtool`:

| extension | `RAYON_NUM_THREADS` | wall | vs. that build's baseline |
| --- | ---: | ---: | ---: |
| uninstrumented | 4 | 302.7 s | — |
| uninstrumented | 1 | 388.4 s | **+28 %** |
| instrumented | 4 | 1205.6 s | — |
| instrumented | 2 | 987.5 s | −18 % |
| instrumented | 1 | **851.5 s** | **−29 %** |

The sign flips with the build, and on this phase the instrumented side is monotone: one
rayon thread is best, and every thread added costs. Uninstrumented, rayon inside an xdist worker is worth 28 %
even at one worker per core, because the workers spend much of their time in serial Python
and I/O and rayon fills the gaps. Instrumented, that same parallelism costs 29 %.

---

## 4. What to do

**1. Turn rayon down for the instrumented run only** (`scripts/coverage.sh`).
`RAYON_NUM_THREADS=1` for the pytest phase, 1–2 for `cargo test`. Projected on the CI
numbers: pytest 594 s → ~420 s, Rust tests 103 s → ~82 s, **`test-linux` ~1030 s →
~840 s**.

> _Status (2026-09-05): Done — one `export RAYON_NUM_THREADS=1` in `scripts/coverage.sh`,
> covering both test phases. `pixi run test-rust` deliberately left alone: the measurements
> are for the 4-core CI shape and a many-core dev box was not measured._

This is coverage-neutral. Thread count does not change which regions execute; the one place
this workspace branches on rayon state is `warp_map::parallelize_rows`, which tests
`rayon::current_thread_index().is_none()` — a property of *where* the caller runs, not of
pool size — so both of its branches stay reachable. Genuine wide-parallel execution is
still exercised on every PR by `test-os` on Windows and macOS, which run the same two
suites uninstrumented at full width.

**2. Get the cache under its budget.** 9.56/10 GB across 14 entries means continuous
eviction, and an evicted target cache costs ~900 s on the job that misses it. That is the
larger number on a bad run.

[#382](https://github.com/sfmtool/sfmtool/pull/382) did land and did help — the Linux and
macOS pixi env caches are gone — but the budget is not under control, because the pressure
was never mostly pixi. At one lockfile hash the steady state is:

| what | GB |
| --- | ---: |
| seven Rust target caches (`{Linux,Windows,macOS}` × `{coverage,test,ui,lint}`) | 6.98 |
| `pixi-win-64` — the one env cache #382 deliberately kept | 1.41 |
| three `cargo-home` caches + apt | 0.26 |
| **steady state** | **8.65** |

That leaves **1.35 GB of headroom, less than a single target cache.** The seven are keyed
on `Cargo.lock` alone, so a lockfile bump needs both generations resident at once — ~14 GB
against a 10 GB cap — and LRU evicts roughly half of them. That is precisely what #384 hit:
its restore found neither the new key nor, via `restore-keys`, the old one. The current
list still carries 1.14 GB of the superseded `1a22…` generation alongside the live `777c…`
one.

So this is structural rather than incidental, and it recurs on every dependency bump. Two
levers, both untested here: shrink what the target caches hold (the four `ui` and `lint`
entries total 3.0 GB and overlap heavily with the `test`/`coverage` ones), or accept the
transient double-generation and stop caching the cheapest of them. The 1.41 GB
`pixi-win-64` entry is also worth re-deriving — it buys ~38 s on one job for 14 % of the
budget, a trade struck before the target caches were this large.

> _Status (2026-09-05): Done, by a third route —
> [#389](https://github.com/sfmtool/sfmtool/pull/389) adds a `prune-caches` job that deletes
> the superseded generation once the current one exists, which is what actually bounds the
> total, and drops the `pixi-win-64` cache as extra margin (28 s/GB against a target
> cache's ~165 s/GB). Steady state 9.56 GB → ~7.0 GB. Shrinking what the target caches hold
> was not needed and is not done._

**3. Decide, explicitly, whether Rust-coverage-from-Python is worth its price.** It is
~318 s per run today (594 s instrumented against 276 s for the same suite uninstrumented on
the same runner class), and ~144 s after fix 1. Dropping it would mean running pytest
against an uninstrumented extension and taking Rust coverage from `cargo test` alone,
losing whatever Rust lines are reachable only through the PyO3 surface. That is a real loss
— `tests/rust_bindings/` exists precisely to cover it — so this is a judgement call rather
than a cleanup. Recorded here so the call can be made against a number.

> _Status (2026-09-05): Deferred by the maintainer — coverage-from-Python stays as it is
> for now. The number above is the standing price._

---

## 5. On `with_min_len` as a general performance measure

Largely already done here, by a better idiom. Where granularity mattered in this codebase
it was fixed with an explicit size threshold *plus a nesting guard*:

- [`warp_map.rs`](../crates/sfmtool-core/src/camera/warp_map.rs) —
  `parallelize_rows(pixels) = pixels > PAR_MIN_PIXELS && rayon::current_thread_index().is_none()`,
  worth 1.45× on a saturated batch (see
  [the normal-refinement report](2026-06-13-perf-patch-normal-refinement.md) § 1).
- [`scale_space.rs`](../crates/sfmtool-core/src/features/sift/scale_space.rs) —
  `par_chunks_mut(4096)` for the DoG difference, `BLUR_STRIPE_ROWS = 32` for the fused blur.

That idiom strictly dominates `with_min_len` for this codebase, because the expensive case
here is *nested* rayon — a per-tile row loop inside an already-saturated per-patch
`par_iter` — and `with_min_len` cannot see that it is running on a worker thread.

The genuinely thin remaining sites, for anyone who wants to chase them, are
[`analysis/cluster_radii.rs:92`](../crates/sfmtool-core/src/analysis/cluster_radii.rs)
(`par_chunks_exact(4)`, ~10 ns of body),
[`geometry/reprojection.rs:63`](../crates/sfmtool-core/src/geometry/reprojection.rs)
(`par_chunks_mut(2)`),
[`spherical/sphere_points.rs:130`](../crates/sfmtool-core/src/spherical/sphere_points.rs)
(`par_chunks_mut(3)`) and
[`spatial.rs:189`](../crates/sfmtool-core/src/spatial.rs) (`par_chunks_mut(k)`). None is on
the measured critical path of anything, and uninstrumented, rayon is a straight win at
every site timed above — so this is opportunistic, not a backlog item.

---

## Method

- CI numbers: `gh run view --log` on runs 34003866655, 34009506011 and 34010579983; cache
  state from `gh cache list`.
- Local: `sfmtool-core --lib` built twice into separate target dirs (plain, and via
  `cargo llvm-cov show-env` for `-C instrument-coverage`), then run under `Start-Process`
  with `ProcessorAffinity = 0xF`. The pytest arms use the same pinning with the extension
  rebuilt each way through `maturin develop --release`.
- The pinned mask is two physical P-cores plus their SMT siblings, not four independent
  cores. SMT siblings share L1/L2, which makes counter sharing *cheaper* than it is across
  distinct cores — so the contention effect measured here is a lower bound on what the
  GitHub runner sees.
