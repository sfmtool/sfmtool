# `sfm render-patches` Command

## Overview

Renders a reconstruction's per-point oriented patches on top of its source
images. Each finite 3D point can carry an oriented planar patch — the quad
spanned by its in-plane half-extent vectors `u`, `v`, centred on the point, with
the point's surface normal. The command projects those quads into every
registered image and composites them onto the source frame.

This is a qualitative aid for visually inspecting a reconstruction's geometry
and patches.

> _**Precondition — shipped (2026-06-25):** `render-patches` now **requires** a
> `feature_source == "embedded_patches"` reconstruction (which carries real
> per-point patch frames and reference bitmaps) and **rejects** `sift_files` with
> a `UsageError` pointing at `sfm xform --to-embedded-patches` (the gate runs in
> the command right after load, before any rendering)._

## Command Syntax

```bash
sfm render-patches <RECONSTRUCTION.sfmr> --output <DIR> [OPTIONS...]
```

The reconstruction must be `embedded_patches` (it carries the per-point patch
frames). Produce one with:

```bash
sfm xform in.sfmr out.sfmr --to-embedded-patches
# for --mode texture, build the bitmaps too:
sfm xform in.sfmr out.sfmr --to-embedded-patches --refine-normals bitmaps=true
```

## Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--output / -o` | path | required | Output directory for the overlay images |
| `--mode` | `texture` \| `normal` \| `flat` \| `wire` | `texture` | Patch fill (see below) |
| `--border / --no-border` | flag | `--no-border` | Outline each patch quad |
| `--border-color` | `R,G,B` | `0,255,0` | Border colour |
| `--border-thickness` | int | 1 | Border thickness (pixels) |
| `--alpha` | float | 1.0 | Global patch opacity (0.0–1.0) |
| `--opaque [V]` | float (0–1) | off | Texture mode: paint texels with confidence > `V` at full opacity, drop the rest. Bare `--opaque` ⇒ `0.1` |
| `--scale` | float | 1.0 | Shrink/grow each quad about its centre |
| `--upscale` | float | 1.0 | Upscale the output canvas (and projected coords) for legibility |
| `--backface-cull / --no-backface-cull` | flag | on | Skip patches whose normal faces away from the camera |
| `--images` | str (repeatable) | all | Only render images whose name contains this substring |

## Fill Modes

| Mode | What it draws | Best for evaluating |
|------|---------------|---------------------|
| `texture` | Warps the per-point RGBA patch bitmap onto the projected quad | Photometric/patch quality — texture should continue the image. **Requires patch bitmaps** (`bitmaps=true`). |
| `normal` | Flat-shades each quad by its world normal, `(n + 1) / 2` | The normal field's smoothness/sanity |
| `flat` | Flat-shades each quad by the point's reconstruction colour | Coverage / point colours |
| `wire` | Outline only, no fill | Geometry and coverage without occluding the image |

## Behaviour Notes

- **Painter's algorithm only.** Patches are sorted back-to-front by depth and
  composited in that order; there is *no* depth buffer, so a distant patch can
  show through a nearer one.
- **`--scale` breaks texture alignment.** Scaling a quad away from `1.0` moves
  its texture off the image content it came from, so use `1.0` for alignment
  checks; `--scale` is only for de-cluttering geometry/normal views.
- **Texture alpha.** By default each texel is composited with the bitmap's
  per-pixel cross-view confidence alpha (the source image bleeds through where
  confidence is low). `--opaque [V]` instead paints every texel whose normalised
  confidence exceeds `V` at full opacity and drops the rest, so any texture/image
  mismatch is real rather than masked by bleed-through. Bare `--opaque` uses
  `V=0.1`; `--opaque 0` keeps every texel with any confidence at all.
- **Projection** goes through the camera model's `ray_to_pixel`, so distorted
  and fisheye cameras (e.g. `OPENCV_FISHEYE` rigs) are handled. Each patch is
  drawn as a straight-edged quad, so a large patch on a fisheye does not follow
  the lens curvature within the quad.

## Output

One image per rendered frame, named `<image-stem>_<mode>.png`. The image's full
relative path is preserved with `/` replaced by `__`, so per-sensor rig frames
with the same basename (e.g. `fisheye_left/frame_05` and
`fisheye_right/frame_05`) do not collide.

## Usage Examples

```bash
# Textured patches with outlines (needs bitmaps)
sfm render-patches out.sfmr -o renders/ --border

# Honest alignment check — no confidence bleed-through
sfm render-patches out.sfmr -o renders/ --opaque

# Normal field on small images, scaled up for legibility
sfm render-patches out.sfmr -o renders/ --mode normal --upscale 3

# Just two views, wireframe geometry
sfm render-patches out.sfmr -o renders/ --mode wire --images _08 --images _12
```
