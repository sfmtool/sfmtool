# Edit Specifications

One spec per **edit family** of the SfM Explorer: the operations that give a
loaded reconstruction a new version. Each is a small spec in the shape of
[`resect-image.md`](resect-image.md) -- invocation, the mechanism it wraps,
what the version's label says, testing, non-goals -- because the shared part is
elsewhere: the value and its overlay are
[`../../core/reconstruction/edited-reconstruction.md`](../../core/reconstruction/edited-reconstruction.md),
the version list and its cursor are [`../document-model.md`](../document-model.md)
and [`../edit-history.md`](../edit-history.md), and the write is
[`../saving.md`](../saving.md).

Every family's mechanism is a **pure function in `sfmtool-core`**, bound through
`sfmtool-py` so the same edit is available offline; what a spec here adds is the
invocation, the version's label and the history entry.

| Document | Description |
|----------|-------------|
| [bundle-adjust.md](bundle-adjust.md) | Refine every pose and every point of the selected reconstruction against its observations, from the Edit menu, with the shared focal held or released. |
| [move-camera.md](move-camera.md) | Move one image's pose by hand: camera view with the camera coming along, so every navigation input moves it, with a live residual readout and one version when the lock is released. |
| [commit-track.md](commit-track.md) | Write the active bench track into the reconstruction: the one step of the bench that touches the file, replacing the point it came from or creating one, with the bench and the value stated in one version. |
| [prune-covered-observations.md](prune-covered-observations.md) | Retire every observation a finer tracked one covers in the same photograph, and drop the points left with too few, from the reconstruction row's context menu, with nothing re-solved. |
| [resect-image.md](resect-image.md) | Re-estimate one image's pose against structure held out from it, from the image row's context menu, and keep the answer as the node's next version. |
| [retriangulate-point.md](retriangulate-point.md) | Re-solve structure at the poses and the lens the value already holds: one point from its context menu in the 3D viewport, or every point from the reconstruction row's, as the node's next version. |

The two edits that predate this directory are specced with the document model
they were built to exercise: **delete point** (the point edit) and **delete
image** (the bulk edit), in
[`../document-model.md`](../document-model.md) § "Two kinds of edit".
