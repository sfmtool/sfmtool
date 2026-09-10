# Previewing the re-triangulation while a camera is moved (amendment)

**Status:** Draft

Amends [`../gui/edits/move-camera.md`](../gui/edits/move-camera.md), which
specifies the shipped lock and points back here.

While the Move Camera lock is held, every point stands where the value has it:
the photograph and the residual readout move, and re-triangulation is what the
commit does, once. That is the right default, because the points this image
observes were solved with its old ray in the mix and re-solving them live would
have them chase the camera — the reviewer would watch the points follow the
photograph instead of the photograph reaching the points. But a reviewer who has
already lined the photograph up sometimes wants the other question answered:
where will the observed points *land*. This draft proposes an option that shows
them there.

## What it would draw

A **Preview re-triangulation** toggle in the lock banner, off by default and
remembered for the session. With it on, the points this image observes are drawn
at the position the commit would give them, and only they: the fixed points
never move under any option, because nothing about the move changes them.

The preview writes over the same rows the base's point buffer already holds, and
restores them from the value when the lock ends — the rows-not-buffers discipline
the additions buffer follows in
[`../gui/document-model.md`](../gui/document-model.md). Patches in the preview
translate with their points and keep their frames; the frame's resize is a
commit-time rule, and a surfel drawn at a slightly wrong size for the duration of
a drag is not a lie about anything the reviewer is deciding.

## Why it needs a cost rule

The preview is per-track work on every frame the pose changes. Re-triangulating
`K` tracks is `O(K)` ray assemblies plus `K` fixed 3×3 eigensolves
([`../core/reconstruction/batch-triangulation-api.md`](../core/reconstruction/batch-triangulation-api.md)),
so it is a few microseconds a track, and pushing `K` new positions to the GPU is
one `write_buffer` over the affected rows. For a typical image that is a fraction
of a frame. It is not bounded, though: an image in a dense embed can observe tens
of thousands of tracks.

The rule should be measured, not guessed:

> The first preview update after the lock is entered is **timed**. If it took
> under 2 ms, the preview runs **every frame** the pose changes. If not, it runs
> **on rest**: when no navigation input has arrived for 150 ms, and once more at
> commit. The banner says which mode it is in (`preview: live` or
> `preview: on rest`).

"On rest" is the adaptive form: the photograph still moves every frame, so the
hand never lags, and the observed points catch up whenever the hand pauses, which
is when the reviewer is looking at them anyway. There is no middle setting —
every Nth frame, a time-sliced subset — because a preview that shows half the
points moved is a lie about the other half, and one that updates at 20 Hz behind
a 60 Hz hand feels broken in a way one that updates on rest does not.

## What it would take

- A per-frame path from the pending pose to a set of previewed positions, which
  is `sfmtool_core::move_camera`'s per-track settle without the value rebuild —
  either a second entry point on the core function or a preview-only helper
  beside it.
- A way for the renderer to write point positions for a set of rows and restore
  them, which the deleted mask's incremental write is the model for but not the
  mechanism: the mask writes one `u32` per point, and this writes three `f32`s.
- The timing decision, its two modes, and the banner text that says which.

## Open questions

- Whether the preview should also apply the patch-frame rescale, so a surfel's
  size is honest during the drag. It is one multiply per patch and the argument
  above says it does not matter; the argument is worth re-testing against a
  capture whose depths move a lot.
- Whether the timed decision should be re-taken when the observed set changes —
  it cannot, within one lock, since the lock is on one image and the value does
  not move while it is held — or simply on every entry, which is what the rule
  above says.
