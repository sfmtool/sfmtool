# Flat-shaded patches without bitmaps (amendment)

**Status:** Draft

Amends [`../gui/patch-rendering.md`](../gui/patch-rendering.md), which specifies
the shipped textured-surfel renderer and points back here.

A reconstruction can carry per-point patch *frames* — the centre, the normal and
the in-plane half-extent vectors `u`, `v` — without carrying the rendered patch
*bitmaps* that fill them. `sfm xform --to-embedded-patches` produces exactly
that: frames on every finite point, bitmaps only if `--refine-normals
bitmaps=true` ran afterwards. The viewer draws nothing for such a
reconstruction: `upload_patches` returns early when the bitmaps are absent, and
the Patches section of the HUD stays hidden. This draft proposes drawing those
patches flat.

## What it would draw

The same oriented quad the textured path builds — same corner expansion from
`center`/`u`/`v`, same `w = 0` branch for points at infinity, same alpha cutoff,
same MRT outputs and `PICK_TAG_POINT` pick id — with the atlas sample replaced by
a solid fill. Two candidate fills:

- **Point colour**, so the surfel reads as an oriented disc of the cloud it came
  from. Simplest, and consistent with how the point splats are coloured.
- **Point colour, Lambert-shaded by the patch normal against a fixed key light.**
  Costs one dot product and makes orientation legible at a glance, which is the
  whole reason to draw a frame-only patch rather than a splat.

The second is the more useful of the two and is the one to build; the first is
what it degenerates to with the light disabled.

## Why it is not simply the textured shader with a white texture

The atlas is what gates the whole upload path. `upload_patches` currently treats
"no bitmaps" as "no patch data", so enabling this means the upload has to build
instances from the frame arrays alone, with no atlas texture, no page grid, and
no per-instance atlas coordinates — which in turn means either a second pipeline
without the atlas bind group, or a shader branch and a 1×1 dummy atlas. The
second is less code and one more dynamic branch per fragment; the first keeps the
textured path exactly as it is. This is the decision the draft exists to make.

## What the UI does

The Patches section is currently hidden unless frames **and** bitmaps exist, and
the Show-patches toggle is greyed in Layers. With a flat path the gate becomes
frames alone, and the section needs one more control — the flat/textured choice
is not a user choice, it is forced by the data, so what it needs is a *readout*
rather than a switch, or nothing at all if the appearance is self-explanatory.

## Open questions

- Whether patches with bitmaps and patches without should be drawable in the same
  frame (a partially refined cloud) or whether the renderer may assume the whole
  reconstruction is one or the other. The atlas page-grid packing assumes the
  latter today.
- Whether the key light should follow the camera (always-lit, no dark patches) or
  sit in world space (consistent shading, some patches unlit). Camera-following
  is friendlier for inspection; world-space reads more like a surface.
