# Bench Specifications

The bench beside a reconstruction, and the things put on it to be worked on.
Implemented in `crates/sfmtool-core/src/bench/`, bound as
`sfmtool._sfmtool.bench`.

| Document | Description |
|----------|-------------|
| [bench.md](bench.md) | The bench value: the ordered list of labelled items, the active one per kind, where a label is minted from and how a collision is suffixed, and why every operation returns a new bench. |
| [editable-track.md](editable-track.md) | The editable track: its observations with their provenance, per-stage measurements and verdicts, the two representations it passes through, and the steps that grow, judge, split and finally commit one as a point. |
