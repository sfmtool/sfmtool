# Defect: `sfm solve` aborts on a degenerate 0-point model, discarding good models — 2026-06-10

Defect report from the cluster-matcher `d` sweep (see
`specs/core/track-cluster-matching.md`, "Default `d` lowered 28 → 10"). The
sweep's seattle_backyard run at `--cluster-d 7` hit this: the incremental
mapper emitted a degenerate model 0 (2 images, **0** 3-D points) alongside a
perfect model 1 (26/26 images, 5,436 points), and the CLI aborted with
`RuntimeError: No 3D points found in reconstruction.` — **the good model was
never written to disk**. The numbers for that sweep point had to be recovered
by converting model 1 manually.

This is a solver-output-handling defect in `sfm solve`, unrelated to the
cluster matcher; any matching method can trigger it whenever COLMAP's mapper
splits off a junk model.

---

## 1. Primary defect: one empty model aborts the whole solve

**Where.** Both solvers share the same save loop:

- `src/sfmtool/_incremental_sfm.py:121-199` — iterates
  `sorted(reconstructions)` in model-index order; for each model calls
  `colmap_binary_to_rust_sfmr` (non-rig) or `pycolmap_to_rust_sfmr` (rig).
- `src/sfmtool/_global_sfm.py:121-` — identical structure.
- `src/sfmtool/colmap/io.py:287` (`colmap_binary_to_rust_sfmr`) and
  `src/sfmtool/colmap/io.py:346` (`pycolmap_to_rust_sfmr`) — both raise
  `RuntimeError("No 3D points found in reconstruction.")` when the model has
  zero points.

**Mechanism.** The loop converts and saves models strictly in index order with
no per-model error handling. If *any* model — typically a 2-image fragment the
mapper abandoned — has zero 3-D points, the converter's `RuntimeError`
propagates out of the loop and the command dies. Models with a higher index
than the degenerate one are silently lost, even when one of them is the real
reconstruction containing every registered image. The user-visible failure is
a hard abort that looks like the solve found nothing, when in fact it
succeeded.

**Reproduction.** Nondeterministic (depends on mapper variance / seed), but
observed concretely on `test-data/images/seattle_backyard` with cluster
matches at `--cluster-d 7` feeding `sfm solve -i`: model 0 = 2 images /
0 points, model 1 = 26 images / 5,436 points; the CLI aborted before writing
model 1. Any run where COLMAP's mapper produces a 0-point fragment with a
lower index than the main model reproduces it.

**Suggested fix.** Skip (with a warning) models below a viability threshold —
at minimum `points3D == 0`, arguably also 2-image fragments — instead of
converting them, and only raise if *no* model survives the filter. The
zero-point check in `colmap/io.py` can stay as a guard; the loop just
shouldn't feed it junk. A regression test can build a reconstruction directory
with an empty model 0 plus a real model 1 and assert the solve writes the real
one.

## 2. Secondary defect: the junk model claims the primary output slot

**Where.** `src/sfmtool/_incremental_sfm.py:148-154` + `_resolve_output_path`
(`_incremental_sfm.py:202-`), and the `return saved_paths[0]` at
`_incremental_sfm.py:199` (same in `_global_sfm.py`).

**Mechanism.** Output naming and the return value key off loop *order*, not
model quality: with an explicit `--output`, the **first** model gets that
exact path, and `saved_paths[0]` is returned as "the" reconstruction. When a
junk 2-image fragment happens to be model 0 (and has ≥ 1 point, so §1 doesn't
abort first), it claims the user's requested output name and becomes the
returned primary, while the real reconstruction lands under an auto-generated
name. Once §1 is fixed by skipping 0-point models, this becomes the next
sharp edge — a 2-image / 3-point fragment would still win the slot.

**Suggested fix.** Order the surviving models by image count (or registered
observations) before assigning output paths, so the largest model is the
primary / `--output` target. The per-model auto-naming already reflects each
model's actual images, so the secondary fragments stay distinguishable.

## 3. Observation: model splits correlate with degraded main-model quality

Not a code defect, recorded for context. In the same sweep, seattle_backyard
at `--cluster-d 10` produced a main model with mean reprojection error
0.978 px where sibling sweep points sat at ~0.33–0.61 px; the run had split a
junk fragment off first, suggesting the fragment captured a strong init pair
and the main model grew from a weaker seed. This is mapper variance, not
something the save loop can fix, but it means model splits are worth surfacing
loudly (the §1 warning) rather than silently skipping — a split is a hint the
run may be worth re-seeding.

---

## Status

Open. No code changes yet; this report is the record of the finding. When
fixed, annotate the sections above in place per the `reports/` convention
(`> _Status (YYYY-MM-DD): Done — <what changed>, commit <sha>._`).
