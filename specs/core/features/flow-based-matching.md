# Flow-Based Feature Matching

Flow-based matching finds SIFT feature correspondences in an ordered image
sequence, such as the frames of a video, without comparing descriptors across
every image pair. It computes dense optical flow between each pair of adjacent
images, moves each image's keypoints along that flow into the images that
follow, and matches each moved keypoint to a nearby keypoint in the target image
whose descriptor is close enough. Keypoints are carried up to five images ahead
by default, so each image is matched to the next five, not only to its
neighbour. `sfm match --flow` and `sfm solve --flow-match` use it. Both write
the matches into a COLMAP database and run COLMAP's geometric verification on
them; `sfm match --flow` does this in a temporary database and saves the
verified result as a `.matches` file, while `sfm solve --flow-match` writes into
the database the solve reads. This spec gives the measurements that set its
parameters, the matching pipeline, its cost and its limits.

## The Idea

Traditional SfM feature matching compares SIFT descriptors between image pairs to find
correspondences. For N images this is O(N^2) pairs, and each pair requires comparing
thousands of 128-dimensional descriptors. For video sequences with hundreds or thousands
of frames, this is the dominant cost.

Video sequences have a structural advantage: adjacent frames have small displacement and
high visual overlap. Dense optical flow can exploit this temporal coherence to find
correspondences without descriptor comparison, then extend those matches to wider
baselines through flow chaining.

This approach is essentially what traditional VFX tracking software (Nuke, PFTrack,
SynthEyes, etc.) has done for decades: track features across frames using optical flow
or template matching, then feed the resulting tracks into a solver. The difference here
is assembling it from SfM components (SIFT keypoints, dense DIS flow, descriptor
validation) rather than using a dedicated tracker, which lets us leverage the existing
feature extraction and reconstruction pipeline.

## Empirical Observations

Testing with `sfm flow` on the Seoul Bull dataset (23k SIFT keypoints per frame,
full-resolution images) established the following baseline behavior:

### Flow quality vs frame separation

| Separation | Flow median | Hit rate | Desc filter | Filtered spatial |
|------------|-------------|----------|-------------|-----------------|
| 1 frame | 5.5 px | 75% | 62% pass | 0.56 px median |
| 2 frames | 10.0 px | 70% | - | 0.64 px median |
| 3 frames | 12.7 px | 67% | - | 0.68 px median |
| 10 frames | 54.9 px | 52% | 26% pass | 0.87 px median |
| 20 frames | 86.9 px | 34% | 26% pass | 0.94 px median |
| 40 frames | 57.0 px | 21% | 1% pass | 1.48 px median |

"Hit rate" = fraction of image1 keypoints whose advected position lands within 3px of
an image2 keypoint. "Desc filter" = fraction of hits with L2 descriptor distance <= 100,
the `sfm flow` diagnostic's default `--descriptor-threshold`. The matcher's threshold
is 250, chosen from the descriptor distance histograms (Observation 2); pass rates at
250 were not recorded. "Filtered spatial" = median hit distance for hits that pass
L2 <= 100. All figures in the table and the observations below are from the default
flow preset unless they name the high-quality preset.

### Observations

1. **Adjacent-frame flow is excellent.** Default preset gives sub-pixel accuracy (0.56px
   median hit distance after the L2 <= 100 filter) with 75% hit rate. The high-quality
   preset on an adjacent pair gave 0.24px median; that run is not in the table.

2. **Descriptor filtering separates signal from noise.** The descriptor distance
   histogram is bimodal: a true-match peak at low distances and a noise peak at higher
   distances. A threshold of L2 <= 250 cleanly separates them. At the diagnostic's
   stricter L2 <= 100, 62% of hits pass at 1 frame apart and 26% at 10 frames, and
   those that pass still have sub-pixel accuracy.

3. **Spatial hit distance alone is insufficient.** At wide baselines, the hit distance
   distribution flattens to near-uniform across 0-3px, meaning most spatial hits are
   coincidental. Descriptor filtering is essential for quality.

4. **High-quality preset matters for wide baselines.** At 40 frames apart, default
   preset finds 50 matches that pass L2 <= 100 (the table's 1%) while high-quality
   finds 647. The multi-scale refinement handles large displacements (130px+ median)
   much better.

5. **Flow field structure is informative.** The flow X-direction histogram becomes bimodal with
   increasing baseline as foreground/background parallax separates. This could be used
   to detect scene structure or estimate difficulty.

## Matching Pipeline Design

### Overview

The matcher is
[_flow_matching.py](../../../src/sfmtool/feature_match/_flow_matching.py)
(`flow_match_sequential`), built on the Rust optical-flow module and driven by
`sfm match --flow` and `sfm solve --flow-match`; `sfm flow` provides the
diagnostics described here.

```python
flow_match_sequential(
    image_paths: list[Path],
    sift_paths: list[Path],
    preset: str = "default",            # "fast", "default" or "high_quality"
    descriptor_threshold: float = 250.0,
    window_size: int = 5,
    max_feature_count: int | None = None,
    trace_path: Path | None = None,
) -> dict[tuple[int, int], np.ndarray]
```

`image_paths` is the sequence in order and `sift_paths` the matching `.sift`
files. `preset` selects the optical flow preset. `window_size` is how many
images ahead each image's keypoints are carried, so pairs `(i, j)` with
`1 <= j - i <= window_size` are matched. `max_feature_count` reads only the
first that many features of each `.sift` file. `trace_path`, when set, receives
a Chrome Trace Event Format JSON timeline of the flow, advection and matching
calls. The result maps each image index pair `(i, j)`, `i < j`, that has at
least one match to an `(M, 2)` uint32 array of `(feature in i, feature in j)`
index pairs, sorted by the first index. A sequence of fewer than two images
returns an empty dict.

`_run_flow_matching` in
[_run.py](../../../src/sfmtool/feature_match/_run.py) calls it, writes the
matches into the COLMAP database and runs geometric verification on the matched
pairs. The command-line options that set the matcher's parameters are the same
on `sfm match --flow` and `sfm solve --flow-match`:

| Option | Parameter | Default |
|---|---|---|
| `--flow-preset {fast,default,high_quality}` | `preset` | `default` |
| `--flow-skip N` (N >= 1; 1 matches adjacent images only) | `window_size` | 5 |
| `--max-features N` | `max_feature_count` | all features |

`descriptor_threshold` and `trace_path` have no command-line option; the
commands always use the 250 default and write no trace.

All images must have the same width and height, because each adjacent flow is
computed between two images of one size. The matcher does not check this up
front: the first adjacent pair whose sizes differ raises `ValueError` from the
flow computation, and no matches are returned or written.

The sequence order is the order of the images' workspace-relative paths,
compared character by character
([`_run_flow_matching`](../../../src/sfmtool/feature_match/_run.py) sorts them
before calling the matcher). Frame numbers therefore need zero padding for name
order to be capture order: `frame_10.jpg` sorts before `frame_2.jpg`. Images
from several directories form one sequence, one directory after another, so the
last images of one directory are matched against the first images of the next.

The pipeline combines two stages:

1. **Flow-based candidate generation** via a sliding window over adjacent flows
2. **Descriptor-filtered validation** on all flow-based candidates

### Sliding Window Flow Matching

The implementation uses a sliding window approach that handles both adjacent and
wide-baseline matching in a single O(N) sweep:

1. For each consecutive frame pair (i, i+1), compute dense optical flow
2. **Advect** all tracked keypoint positions one hop forward through the new adjacent
   flow field — each advection is O(keypoints), just bilinear lookup per point
3. Match all window source images against the current frame by finding the K=5
   nearest keypoints in the target frame within a 10px candidate radius of the
   advected position, then keeping the one with the best descriptor match
4. When the window exceeds `window_size`, drop the oldest entry

With `window_size=5`, this produces matches at skip=1 through skip=5 for every frame.
Each window entry holds, for one source image, its advected keypoint positions
(N, 2), a validity mask (N,), its original keypoint positions (N, 2) and its
descriptors (N, 128 bytes), so the window's memory is O(window_size × N_features),
dominated by the descriptors. Beyond the window, the matcher holds the current
image's keypoints and descriptors, the current and next grayscale images, the
current adjacent flow field and the one being computed; it never keeps a flow
field after its hop.

#### Matching one window entry against the current image

For each window entry, `_flow_match_from_advected` matches its source image against
the current image:

1. Take the entry's advected positions whose validity mask is still set. A keypoint
   whose advected position leaves the image at any hop is dropped for good.
2. Find the K=5 nearest keypoints in the current image within a 10px candidate radius
   of each advected position, using a KD-tree (`nearest_k_within_radius`).
3. Among those candidates, keep the one with the lowest L2 descriptor distance,
   provided that distance is at most `descriptor_threshold` (250 by default).
4. Deduplicate: if several source keypoints match the same target keypoint, keep the
   pair with the lowest descriptor distance.

Steps 3 and 4 are `match_candidates_by_descriptor` in `sfmtool._sfmtool.matching`.
`_flow_match_pair` in the same Python module runs the same steps for one image pair
from a full flow field; only the tests call it.

The descriptor comparison is only between spatially-matched pairs (not all-vs-all),
so it's O(K) per pair where K is the number of keypoints, not O(K^2).

### Descriptor Filtering

The descriptor filter is what makes flow-based matching reliable at wide baselines.
Without it, spatial proximity alone has a high false positive rate: at 20 frames
apart, 74% of the spatial hits fail the diagnostic's L2 <= 100 test (the table's 26%
pass rate). With it, the surviving matches have genuine descriptor agreement and sub-pixel
spatial accuracy.

A single threshold of L2 <= 250 is used for all baselines. The descriptor distance
histogram is bimodal (true-match peak at low distances, noise peak at higher distances),
and 250 cleanly separates them across all tested baselines.

### Error Accumulation

Per-frame advection error ~0.5px accumulates as ~sqrt(N) * 0.5px for random errors,
giving ~1.6px at 10 frames — well within the 10px candidate radius. The descriptor
filter catches any advection errors that survive the radius by keeping only the
best-matching candidate.

## Future Directions: Wide-Baseline Pair Selection

The current implementation uses a fixed-size sliding window. Potential improvements:

### Accumulated displacement trigger

Track the cumulative flow magnitude along the sequence. When the accumulated median
displacement since the last wide-baseline computation exceeds a threshold (e.g., 50px),
trigger additional wide-baseline pairs. This adapts to camera speed.

### Covisibility-driven

After initial matching, build a covisibility graph from shared tracks. Compute
additional wide-baseline flows only between pairs with sufficient but incomplete
overlap.

## Cost Analysis

### Per-pair costs

| Operation | Cost (2880x2880) | Notes |
|-----------|-----------------|-------|
| Adjacent flow (default) | ~0.15-0.3s | Rayon parallel, SIMD |
| Adjacent flow (high_quality) | ~0.6-1.0s | More pyramid levels, larger patches |
| Keypoint advection (23k pts, per hop) | <0.01s | Bilinear lookup per point |
| Descriptor load + filter | ~0.05s | Read .sift files, L2 distance |

The flow timings are for the CPU path. The matcher calls `compute_optical_flow`
without `use_gpu`, so it runs the flow on the GPU whenever one is available
(`gpu_available()` in `sfmtool._sfmtool.flow`) and on the CPU otherwise; neither
command has an option to choose. With a GPU, pyramid levels smaller than the
preset's `gpu_min_pixels` still run on the CPU
([gpu-optical-flow.md](gpu-optical-flow.md)).

The per-image costs do not add up to the wall time, because the flow runs on one
background thread: while the main thread advects the window into image j,
matches it against image j and reads the features of image j + 1, the flow from
image j to image j + 1 is already being computed. The time per image is therefore close to
the larger of the flow time and the load, advect and match time, not their sum.

## Limitations

### Scene assumptions

Flow-based matching assumes temporal coherence — it works for video sequences where
consecutive frames overlap significantly. It does not help for:

- Unordered image collections (no temporal relationship)
- Large scene jumps or camera repositioning within a sequence
- Very fast camera motion where adjacent frames have little overlap

### Occlusion handling

Points that become occluded between frames produce incorrect flow and cannot be
recovered through chaining. The descriptor filter catches most of these (the flow
sends the point somewhere wrong, where the descriptor won't match), but occluded
points are a permanent loss in the chain.

### Textureless regions

Optical flow is unreliable in textureless regions (sky, uniform walls). However,
SIFT keypoints are rarely detected in such regions, so this has limited practical
impact on keypoint-based matching.

### Repetitive texture

Flow can be correct (converges to the nearest local minimum) but land on the wrong
instance of a repetitive pattern (e.g., bricks, tiles). The descriptor filter helps
here since different instances may have slightly different descriptors, but this
remains a failure mode shared with all local matching methods.

## Relationship to Existing Pipeline

Flow-based matching is one of the matching methods the user picks on `sfm match`
(`--flow`, alongside `--exhaustive`, `--sequential` and `--cluster`) or on
`sfm solve` (`--flow-match`); one run uses one method. It suits video sequences.
Unordered collections need one of the descriptor-based methods, since they have no
temporal order to follow. A pair that flow matching fails on, for example after a
cut in the video, gets no matches; no descriptor matching runs in its place.

The output of flow-based matching is the same as descriptor matching: a set of
(image_i, feature_j, image_k, feature_l) correspondences that feed into track
building and SfM solving.
