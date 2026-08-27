# The camera model registry

**Status:** Implemented — the registry and both `SfmrCamera` conversions live in
`crates/sfmtool-core/src/camera/intrinsics/registry.rs`; the
[`CameraModel`](../../formats/sfmr-file-format.md#3-cameras-camerasmetadatajsonzst)
enum and its accessors stay in `camera/intrinsics.rs`. Tests in
`camera/intrinsics/tests.rs`.

## Purpose

Two types describe a camera in this workspace, and there is exactly one place
where they meet. This spec says what each is for, why they are not merged, and
the invariant that lets the boundary between them be generated rather than
written twice.

## The two types

```
sfmr-format   SfmrCamera { model: String, width, height,
  (bottom)                 parameters: BTreeMap<String, f64> }
     ▲
     │ depends on
     │
sfmtool-core  CameraModel::Pinhole { focal_length_x: f64, … }   (15 variants)
```

`sfmr-format` does not depend on `sfmtool-core`. The dependency runs one way
and must stay that way.

**`SfmrCamera`** (`sfmr-format/src/types.rs`) is the **wire type**. It is a
stringly-typed bag because that is literally the shape of the
`cameras/metadata.json.zst` payload, and because the parameter names are
COLMAP's. Its consumers are the I/O crates — `sfmr-format`, `sfmr-colmap`,
`camrig-format` — which read and write cameras without needing to know what a
projection is.

**`CameraModel`** (`sfmtool-core/src/camera/intrinsics.rs`) is the
**computation type**. It is a closed enum so that every projection, Jacobian,
distortion and GPU-mesh code path is exhaustively matched by the compiler.

### Why they are not merged

- Pushing `CameraModel` down into `sfmr-format` inverts the layering: the
  bottom crate of the workspace would inherit the algorithm layer's
  dependencies, and `camrig-format` and `sfmr-colmap` would gain a transitive
  dependency on geometry code they never call.
- Using `SfmrCamera` for computation loses exhaustiveness. Adding a variant
  would stop being a compile error and start being a runtime `KeyError`
  equivalent — `parameters["focal_length_x"]` in a hot loop.
- The map can carry a model this build does not know. That is what makes
  `TryFrom<&SfmrCamera>` fallible and `From<&CameraIntrinsics>` infallible, and
  it is a deliberate forward-compatibility property of the format: an unknown
  model is an error at the boundary, not a panic in a solver.

## The naming invariant

> For every fixed-arity camera model, the **struct field name is byte-identical
> to the serialized parameter name**.

`CameraModel::Pinhole { focal_length_x, … }` serializes as
`"focal_length_x"`, and nothing else. This holds for all 13 fixed-arity
variants without exception, in both directions.

The invariant is what makes the registry possible: the parameter-name strings
need not be written at all, because they are recoverable from the field
identifiers. The authoritative human-readable list of models and parameter
names remains the table in
[../../formats/sfmr-file-format.md](../../formats/sfmr-file-format.md) §3; this spec
governs how the code realizes it.

## The registry

`camera/intrinsics/registry.rs` declares every model once, in a
`camera_models!` invocation with two blocks:

```rust
camera_models! {
    fixed_arity {
        Pinhole => "PINHOLE" {
            focal_length_x, focal_length_y, principal_point_x, principal_point_y
        },
        …
    }
    custom {
        SfmtoolFisheye => "SFMTOOL_FISHEYE" as SFMTOOL_FISHEYE,
        SfmtoolPinhole => "SFMTOOL_PINHOLE" as SFMTOOL_PINHOLE,
    }
}
```

From that one table the macro generates:

- `CameraModel::model_name()` — over **all** variants, fixed and custom.
- `fixed_arity_params(&CameraModel) -> BTreeMap<String, f64>` — the write
  half, with `stringify!` supplying each key.
- `fixed_arity_param_names(&CameraModel) -> &'static [&'static str]` — the same
  keys in **declaration** order, which is what `CameraModel::parameter_names()`
  hands a parameter table. A `BTreeMap` can only offer lexicographic order,
  which separates related terms and sorts `bspline_c10` before `bspline_c2`.
- `fixed_arity_from_sfmr(&str, &BTreeMap<…>) -> Result<CameraModel, …>` — the
  read half, with the same `stringify!` supplying each lookup, and an
  `UnknownModel` error for any name not in the table.
- `MODEL_COUNT`, the number of registered models — test-only, since its whole
  job is letting the test corpus assert its own completeness.
- One `&'static str` constant per custom model, so its name is not respelled at
  the dispatch site.

For a fixed-arity model the serialized key is therefore **never written as a
string at all**. The name appears twice as an *identifier* — the enum's field
declaration and the registry entry — and those two are checked against each
other by the compiler (see the table below), so they cannot drift apart. The
old arrangement wrote each key as a string literal in two independent matches
that nothing compared.

### What the compiler enforces

The registry is not a convention; each of these is a build failure:

| Mistake | Caught by |
|---|---|
| New variant not added to the registry | `model_name` and `fixed_arity_params` become non-exhaustive matches |
| Registry lists a field the variant does not have | unknown field in a struct pattern / expression |
| Registry omits a field the variant does have | `pattern does not mention field` / `missing field in initializer` |
| Read and write disagree about a name | not representable — both derive from the same identifier |

The last row is the point of the exercise. Before the registry, the two
directions were independent 100+ line matches and a one-sided edit produced a
camera that wrote but would not read back, with nothing watching.

### The `custom` block

A model whose parameter list is **not** a fixed set of `f64` fields cannot be
derived from field identifiers and is registered as `custom`. It contributes
its name to `model_name()` and to `MODEL_COUNT`, but its serialization — and
its parameter-name list — is hand-written and intercepted at the top of each
conversion, before the fixed-arity path is reached. The generated code carries
an `unreachable!` arm for custom variants naming the interception, rather than
absorbing them into a `_` arm.

The two sfmtool spline models, `SFMTOOL_FISHEYE` and `SFMTOOL_PINHOLE`, are the
custom models today. Each carries a variable-length parameter list
(`bspline_c0..bspline_c{N−1}` behind a declared `bspline_coeff_count`) and a
read path that performs real validation — declared count against stray keys, the
`MIN_BSPLINE_COEFFS` floor, the domain end. They differ only in the name of that
domain end (`bspline_theta_max` against `bspline_rho_max`), so one `get_bspline`
helper takes the key as a parameter and serves both. That asymmetry is the
models' design, specified in
[../../formats/sfmtool-camera-models.md](../../formats/sfmtool-camera-models.md), and
must not be flattened into the table.

## Testing requirements

- **Round-trip, every model.** Each registered model round-trips
  `CameraIntrinsics → SfmrCamera → CameraIntrinsics` to an equal value.
- **The fixture list is complete.** The test corpus that drives the round-trip
  asserts its own length against `MODEL_COUNT`, and that the models in it are
  distinct. A new variant cannot be registered (which the compiler already
  forces) and then silently left untested.
- **Names are pinned.** A golden test asserts the exact model-name string and
  the exact sorted parameter-name set for every registered model. Deriving both
  directions from one identifier means a renamed field renames the on-disk key
  too, and every round-trip test would still pass — the same hazard, and the
  same remedy, as `entry_names_are_pinned` in the format crates.
- **`parameter_names()` is a permutation of the written keys**, for every
  registered model. The two are one list for a fixed-arity model, but the
  custom models' names are hand-written twice — once to serialize, once to
  order — and this is what holds those two copies together.
- **Custom-model validation** is unchanged and specified with the model, in
  [../../formats/sfmtool-camera-models.md](../../formats/sfmtool-camera-models.md).

## Out of scope

The COLMAP interop layer (`sfmr-colmap`) keeps its own model-name mapping,
including the `EQUIDISTANT_FISHEYE` ↔ `SIMPLE_RADIAL_FISHEYE`-with-`k=0`
carrier rule and the `SFMTOOL_FISHEYE` export rejection. That is a translation
between two external conventions, not a restatement of this one, and it is
specified in [../../formats/sfmr-file-format.md](../../formats/sfmr-file-format.md).
