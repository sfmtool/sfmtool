# Reconstruction Specifications

Operations on reconstruction data itself. Implemented in
`crates/sfmtool-core/src/reconstruction/`.

| Document | Description |
|----------|-------------|
| [edited-reconstruction.md](edited-reconstruction.md) | A reconstruction as a value: the image table, the point set, the two heavy columns they share, and the edited form that is a shared immutable base plus its point edits, with materialisation and the row map. |
| [move-camera.md](move-camera.md) | Putting one image at a pose a caller names, and settling the tracks it observes around it: the ones two or more pixels see re-triangulated, a bearing only it sees turned with the camera, everything else kept, and a residual pair measured before and after. |
| [switch-camera-model.md](switch-camera-model.md) | Switching cameras of a reconstruction to another camera model by fitting the new model to the old one, with poses, points and keypoints untouched, and the reprojection errors compared before and after on one fixed set of observations. |
| [outermost-keypoint.md](outermost-keypoint.md) | The keypoint of each camera that lies furthest from its principal point, among the observations and among every feature detected in the images' `.sift` files, as a radius and an incidence angle: how far out the photographs reach, shown wherever a spline domain is edited. |
| [bundle-adjust.md](bundle-adjust.md) | Bundle-adjusting a whole reconstruction: what goes into the kernel, camera by camera, the constraints and representations it honours, the points it leaves unsupported and deletes, and the patch frames it rescales with their depth. |
| [batch-triangulation-api.md](batch-triangulation-api.md) | Batch triangulation carrying per-point observability diagnostics, and the classifier over them. |
| [triangulation-rules.md](triangulation-rules.md) | Reading every point of a track set from its observations at one geometry, with the per-track rules (floor, cheirality, bar, few) held once for every caller, and the reconstruction-level re-triangulation over them that honours a value's own point constraints. |
| [add-image-to-tracks.md](add-image-to-tracks.md) | Adding one image's observations to the tracks it can see: per point, a visibility check, a search for the new keypoint against the consensus of the existing observations, a photometric verdict by a selectable rule, a positional gate, and the write-back, with nothing moved. |
| [prune-covered-observations.md](prune-covered-observations.md) | Retiring every observation a finer tracked one covers in the same image, and dropping the points left with too few: a subtraction over the value, with nothing re-solved. |
| [point-correspondence.md](point-correspondence.md) | Finding the same 3D point across two reconstructions, and merging their tracks. |
