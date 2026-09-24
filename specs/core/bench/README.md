# Bench Specifications

The bench beside a reconstruction, and the things put on it to be worked on.
Implemented in `crates/sfmtool-core/src/bench/`, bound as
`sfmtool._sfmtool.bench`.

| Document | Description |
|----------|-------------|
| [bench.md](bench.md) | The bench value: the ordered list of labelled items, the active one per kind, where a label is minted from and how a collision is suffixed, and why every operation returns a new bench. |
| [editable-track.md](editable-track.md) | The editable track: its observations with their provenance, per-stage measurements and verdicts, the two representations it passes through, the evaluation that measures it at either and the transitions between them, the descriptor search that grows one from an index of the capture, and the steps that grow, judge, split and finally commit one as a point. |
| [track-at-pixel.md](track-at-pixel.md) | Building a track at a pixel: the cascade of four ways of finding the other photographs' sightings (the `.matches` clusters, the neighbours' affine transfer, a plane sweep and the SIFT index's constellation query), the shared finish that anchors the patch on the pixel, grows, cleans and gates it, and the report and refusal a query returns. |
