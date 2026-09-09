# Edit Specifications

One spec per **edit family** of the SfM Explorer: the operations that give a
loaded reconstruction a new version. Each is a small spec in the shape of
[`../resect-image.md`](../resect-image.md) -- invocation, the mechanism it wraps,
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
| [add-observation.md](add-observation.md) | Add one observation of the selected point to the image on screen, at a right-clicked pixel, placed by the embed pass's own photometric kernel and followed by a re-triangulation of the track. |
| [create-point.md](create-point.md) | Create a 3D point at a right-clicked pixel, at infinity along that pixel's ray, with a one-observation track and a patch sized by a prompt. |

The two edits that predate this directory are specced with the document model
they were built to exercise: **delete point** (the point edit) and **delete
image** (the bulk edit), in
[`../document-model.md`](../document-model.md) § "Two kinds of edit".
