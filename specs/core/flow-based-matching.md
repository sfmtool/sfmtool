# Flow-Based Feature Matching

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
an image2 keypoint. "Desc filter" = fraction of hits with L2 descriptor distance <= 100.
"Filtered spatial" = median hit distance for descriptor-filtered matches.

### Observations

1. **Adjacent-frame flow is excellent.** Default preset gives sub-pixel accuracy (0.58px
   median) with 75% hit rate. High-quality preset improves to 0.24px median.

2. **Descriptor filtering separates signal from noise.** The descriptor distance
   histogram is bimodal: a true-match peak at low distances and a noise peak at higher
   distances. A threshold of L2 <= 250 cleanly separates them. At 1 frame apart, 62%
   pass; at 10 frames, 26% pass but those that pass still have sub-pixel accuracy.

3. **Spatial hit distance alone is insufficient.** At wide baselines, the hit distance
   distribution flattens to near-uniform across 0-3px, meaning most spatial hits are
   coincidental. Descriptor filtering is essential for quality.

4. **High-quality preset matters for wide baselines.** At 40 frames apart, default
   preset finds 50 descriptor-filtered matches while high-quality finds 647. The
   multi-scale refinement handles large displacements (130px+ median) much better.

5. **Flow field structure is informative.** The flow X-direction histogram becomes bimodal with
   increasing baseline as foreground/background parallax separates. This could be used
   to detect scene structure or estimate difficulty.

## Matching Pipeline Design

### Overview

The pipeline combines two stages:

1. **Flow-based candidate generation** via a sliding window over adjacent flows
2. **Descriptor-filtered validation** on all flow-based candidates

### Sliding Window Flow Matching

The implementation uses a sliding window approach that naturally handles both adjacent
and wide-baseline matching in a single O(N) sweep with O(window_size) memory:

1. For each consecutive frame pair (i, i+1), compute dense optical flow
2. **Advect** all tracked keypoint positions one hop forward through the new adjacent
   flow field — each advection is O(keypoints), just bilinear lookup per point
3. Match all window source images against the current frame by finding the nearest
   keypoint in the target frame within spatial tolerance (3px default) of the
   advected position
4. When the window exceeds `window_size`, drop the oldest entry

With `window_size=5`, this produces matches at skip=1 through skip=5 for every frame.
Memory usage is O(window_size × N_features) — only tracked keypoint positions are stored,
not full-resolution flow fields.

#### Per-pair matching

For each source/target pair with a precomputed flow:

1. Advect all SIFT keypoints from source through the flow field
2. Filter to advected points that land in-bounds
3. Find nearest keypoint in target within spatial tolerance using KD-tree
4. Filter by descriptor L2 distance <= threshold (default 250)
5. Deduplicate: if multiple sources match the same target, keep the pair with the
   lowest descriptor distance

The descriptor comparison is only between spatially-matched pairs (not all-vs-all),
so it's O(K) per pair where K is the number of keypoints, not O(K^2).

### Descriptor Filtering

The descriptor filter is what makes flow-based matching reliable at wide baselines.
Without it, spatial proximity alone has a high false positive rate (74% at 20 frames
apart). With it, the surviving matches have genuine descriptor agreement and sub-pixel
spatial accuracy.

A single threshold of L2 <= 250 is used for all baselines. The descriptor distance
histogram is bimodal (true-match peak at low distances, noise peak at higher distances),
and 250 cleanly separates them across all tested baselines.

### Error Accumulation

Per-frame advection error ~0.5px accumulates as ~sqrt(N) * 0.5px for random errors,
giving ~1.6px at 10 frames — well within the 3px spatial tolerance. The descriptor
filter catches any advection errors that exceed the tolerance.

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

Flow-based matching would complement, not replace, the existing descriptor matching:

- **Video sequences**: flow-based matching as the primary method, with descriptor
  matching as fallback for pairs where flow fails (detected via low hit rate or
  poor descriptor agreement)
- **Unordered collections**: descriptor matching only (no temporal structure)
- **Mixed**: use frame timestamps or EXIF data to identify sequential runs within
  an unordered collection, apply flow-based matching within runs

The output of flow-based matching is the same as descriptor matching: a set of
(image_i, feature_j, image_k, feature_l) correspondences that feed into track
building and SfM solving.
