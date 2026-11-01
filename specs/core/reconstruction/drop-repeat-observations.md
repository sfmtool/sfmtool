# Drop Repeat Observations

A point's track is the list of its observations: the feature in each
photograph that sees the point. A sound track sees its point at most once in
each photograph. A GLOMAP solve can hand back a point whose track holds two or
more observations in one image, each a different feature of that image, and the
Track Table counts them in its `repeat_observations` column
([track-table.md](../analysis/track-table.md)). This operation keeps, in each
such image, the one observation that agrees best with the point and drops the
others, so the point is seen once per image. Every reconstruction read from
COLMAP into `.sfmr` goes through it.

It is a **subtraction, not a solve**, like
[prune-covered-observations.md](prune-covered-observations.md). No point moves
and none is removed, and each point keeps every image it was seen in; only its
observation list is shorter.

## Rust API

The operation is an inherent method on `SfmrReconstruction` in
[repeat_observations.rs](../../../crates/sfmtool-core/src/reconstruction/repeat_observations.rs),
bound as `SfmrReconstruction.drop_repeat_observations()` on
`sfmtool.reconstruction`, which returns the new reconstruction and a dict of
the two counts.

```rust
/// What `drop_repeat_observations` did.
pub struct RepeatObservationsReport {
    /// Observations dropped, over every point.
    pub observations_dropped: usize,
    /// Points that lost at least one observation.
    pub points_changed: usize,
}

impl SfmrReconstruction {
    pub fn drop_repeat_observations(
        &self,
        progress: &Progress<'_>,
    ) -> Result<(SfmrReconstruction, RepeatObservationsReport), ReconstructionError>;
}
```

It takes a value and returns a value, rather than editing in place, because
its callers build a reconstruction and then pass it through a sequence of such
steps (the COLMAP import runs it between `from_data` and the classification of
points at infinity). It takes no options: the rule has no threshold to choose.

```rust
let (cleaned, report) = recon.drop_repeat_observations(&Progress::none())?;
println!(
    "dropped {} observations from {} points",
    report.observations_dropped, report.points_changed
);
```

It fails only when a `.sift` file the pixels have to come from cannot be read
(`ReconstructionError::SiftRead`), or when `progress` asks it to stop
(`ReconstructionError::Cancelled`).

## Which observation is kept

For each point and each image in which its track holds more than one
observation, the observations of that image are ranked by their reprojection
error against the point, the same error the stored RMS is made of
([sfmr-file-format.md](../../formats/sfmr-file-format.md#points3dreprojection_errorsnfloat32zst)):
the point projected through the image's pose and its own intrinsics, per-image
lens corrections included, and the pixel distance to the observation's
keypoint. A point at infinity (`w = 0`) projects its bearing through the
rotation alone. The lowest error is kept.

- An observation the camera cannot image (behind it, or outside the lens's
  domain), or whose pixel is not known (NaN), ranks after every observation
  that projects.
- Equal errors keep the observation that comes first in the point's track, and
  so does a group in which no observation projects. For a solve's output the
  track order is the order COLMAP stored, so the result is the same on every
  run.

## What the value that comes back holds

- **The observation columns take one row selection.** The tracks, the feature
  indexes, the inline keypoints, the observation confidences and the
  observation readings all drop the same rows, and the per-image map from
  feature to point is rebuilt from what is left. A point whose reference
  observation is dropped keeps its patch bitmap and gets `-1` as its
  reference.
- **A changed point is scored again.** Its RMS reprojection error is computed
  over the observations it keeps, from the same pixels the ranking used; an
  observation that does not project is left out, and a point with none left
  stores `0.0`. A point that lost nothing keeps its stored error.
- **The depth statistics follow the tracks.** The per-image depth statistics
  and histograms count one entry per observation, so they are derived again
  from the shortened tracks. The derivation also fills each missing normal
  with the point's mean viewing normal; the operation keeps every normal the
  value had instead, missing ones included, since this operation is about
  tracks.
- **A value with no repeat comes back as it is**, content hashes included, and
  no file is read.

The pixels are the inline `keypoints_xy` column when the value carries one,
which every value built by `SfmrReconstruction.from_data` does when its `.sift`
files are present. Otherwise they are read from the `.sift` files of the images
the changed points are seen in.

## Where it runs

`colmap_binary_to_rust_sfmr` and `pycolmap_to_rust_sfmr` in
[io.py](../../../src/sfmtool/colmap/io.py) are the two ways a COLMAP
reconstruction becomes an `SfmrReconstruction`, and both run this on the value
`from_data` builds, before points at infinity are classified, so the noise that
classification measures from the stored errors is not raised by the dropped
observations. They print `Dropped N repeat observation(s) from M point(s)` when
anything was dropped. Every `sfm solve` (incremental, global and rig), `sfm
from-colmap-bin` and `sfm densify` reaches it through them.

The file's metadata does not record the count. Its `tool_options` holds the
options a tool was run with, not what an import found, and the format has no
field for an import note.

A `.sfmr` file written before the import ran this can still hold repeat
observations, and nothing cleans them on load. `sfm xform --bundle-adjust`
through pycolmap exports the input's tracks and reads the same tracks back,
so it neither adds nor removes them, and does not run this.
