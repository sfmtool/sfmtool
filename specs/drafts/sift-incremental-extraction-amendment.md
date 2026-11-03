# Incremental SIFT descriptor extraction (amendment)

**Status:** Draft

Amends two standing specs, which describe the shipped behaviour:
[`../core/features/sift.md`](../core/features/sift.md) (the SIFT detector and
its split detect/describe interface) and
[`../formats/sift-file-format.md`](../formats/sift-file-format.md) (the `.sift`
container, whose current version is 2). Both point back here. It builds on
`.sift` version 2, which adds the required `image_to_gray` object and the
per-image sharpness and gradient structure tensor
([sift-file-format.md § Format versions](../formats/sift-file-format.md#format-versions));
the growable layout here is version 3.

Today a `.sift` file is written once, in full: every keypoint in it carries a
descriptor. The in-memory interface is already lazy — `detect_keypoints` hands
the caller a keypoint pool plus the retained `ScaleSpace`, and
`compute_descriptor` realizes any one of them on demand — but nothing survives
the end of the process, so a *detect-many, describe-few* workflow cannot span
two CLI invocations. This draft proposes making the archive itself growable, and
the `.sift` format version 3 that expresses it on disk.

Two things ship together or not at all, because neither is useful alone: the
lifecycle (`sfm sift --detect` / `--describe`, and describe-on-demand inside
`sfm match`) and the on-disk layout (chunked descriptors, `described_count`, and
the identity hashes that survive an append).

## Part 1 — the design (amends `core/features/sift.md`)

The section below returns to `sift.md`, after "Lazy descriptors and
coarse-to-fine", when this ships.

Lazy descriptor fill must survive **across CLI commands** — one command detects the
keypoint pool, later commands describe more of it on demand — so the working state lives
on disk, not just in memory. We extend the `.sift` archive itself into a **growable**
container rather than adding a sidecar. The design rests on one structural choice:

**The `.sift` format already stores features sorted by descending size** (the existing
backends do this), and the incremental design depends on that ordering. Coarse-to-fine
always wants the largest keypoints first, so the set of *described* keypoints is always a
dense **prefix** `[0, M)` of the keypoint list. That
means we never need a sparse coverage mask — coverage is a single integer `M`
(`described_count`), and descriptors are stored as contiguous **range chunks** that tile
that prefix. (Sort ties broken deterministically — e.g. by response, then octave, then
`(y, x)` — so the order is reproducible.)

**Descriptor chunks as range-named tensor entries.** Instead of one `descriptors`
array, the archive holds a sequence of append-only chunk entries named by their
inclusive keypoint-index range:

```
descriptors.0-100.128.uint8       # first describe-batch: keypoints 0..=100  (101 rows)
descriptors.101-1000.128.uint8    # later batch appended:  keypoints 101..=1000 (900 rows)
descriptors.1001-4095.128.uint8   # ...
```

Chunks are contiguous and gap-free (`next.start == prev.end + 1`) because coverage only
ever grows as a prefix. Reading the full descriptor block = concatenate chunks in order;
reading the top-K (coarse-to-fine) = read only the chunks covering `[0, K)`, the natural
extension of `read_sift_partial`.

**Append data, rewrite two small files.** Each describe-batch:

1. **Appends** one new immutable `descriptors.<a>-<b>.128.uint8` entry to the ZIP. Bulk
   data is strictly append-only and never rewritten.
2. **Rewrites the two small mutable JSON entries** — `features/descriptors_metadata.json`
   (the coverage count `described_count`) and `content_hash.json` (hashes only).
   `metadata.json` and the keypoint/thumbnail arrays stay immutable, so the stable
   `feature_set_xxh128` and `metadata_xxh128` never change. Both rewritten files are tiny.

The integrity model evolves but stays verifiable. Today's `content_xxh128` is already a
*digest of digests* (it hashes the concatenation of each array's xxh128). We keep that
structure: `content_hash.json.zst` lists the per-entry digests, including one per
descriptor chunk, and `content_xxh128` is the hash of the concatenated digest list.
Appending a chunk therefore only appends one digest and rehashes the small digest list —
**no rehashing of existing data**. At every point the archive is fully verifiable; the
contract changes from "exactly `N` keypoints each with a descriptor" to "`N` keypoints
with descriptors for a verified prefix of length `described_count ∈ [0, N]`."

**Lifecycle / CLI flow** (proposed — `--detect` / `--describe` / `--top-k` are
not yet implemented; today `sfm sift --extract` writes a fully-described file):

```
sfm sift --detect images        # writes keypoints sorted by size; described_count = 0
sfm sift --describe -i images --top-k 1000   # appends descriptors.0-999; described_count = 1000
sfm match ...                   # triggers describe-on-demand for the keypoints it needs,
                                # appending further chunks; reuses any already on disk
```

A later command reads `described_count`, computes only the still-missing descriptors
(rebuilding the `ScaleSpace` from the source image — deterministic given params and the
`image_file_xxh128` already recorded), and appends them. The pyramid rebuild is the
price of cross-process laziness; it is paid only when new descriptors are actually
needed, and amortizes when a batch describes many keypoints at once.

**External consumers.** Tools that need a complete dense block (COLMAP export, bulk
matching) require `described_count == feature_count`, i.e. describe-all. Once fully
described, a version 3 file can simply be written in the version 2 layout (a single descriptors array) for
those consumers — that is an ordinary format conversion, not a special operation.

**Referential stability.** Appending descriptors changes the file's whole-file
`content_xxh128`, which would break any `.sfmr`/`.matches`/workspace reference that pinned
the `.sift` by that hash — even though the expanded file is a strict superset. The format
solves this with a **stable** `feature_set_xxh128` (over the immutable image + keypoints +
tool config, excluding descriptors) that references should use instead. Descriptor-dependent
consumers (matches) additionally verify the immutable `[0, M)` descriptor prefix they relied
on. Full definitions in Part 2 below.
This implies updating the `.sfmr`, `.matches`, and workspace specs to reference
`feature_set_xxh128`.

**Concurrency.** Appending to one ZIP plus rewriting its metadata is a single-writer
critical section. Wrap describe-and-append in an advisory file lock on the `.sift`
(monotonic prefix growth makes the lock window short and conflicts rare). Concurrent
*readers* are unaffected: they read `described_count` and the chunks present.

**Open implementation details to validate.**

- ZIP append mechanics with the `zip` crate: appending data entries is cheap (rewrites
  only the central directory), but *replacing* the two mutable JSON entries
  (`features/descriptors_metadata.json.zst`, `content_hash.json.zst`) needs either
  duplicate-name-last-wins or a central-directory rewrite — pick and pin the reader's
  resolution rule.
- This is a `.sift` **format version bump**; `read_sift` must handle both the legacy
  single-`descriptors` layout and the new chunked layout (legacy = one implicit
  `0-(N-1)` chunk with `described_count = N`).
- The normative on-disk definition (chunk naming grammar, `described_count`, the
  `component_xxh128` digest cache and recompute rule, concurrency) is Part 2 below;
  this part is the design rationale.

## Part 2 — the on-disk layout (amends `formats/sift-file-format.md`)

Version 3 of the `.sift` format. `metadata.version` becomes `3`; version 1 and 2
files stay valid and readable, and a fully described version 3 file may be written
back in the version 2 layout for consumers that want the plain single-array form.
Everything version 2 adds (`feature_options.image_to_gray` and the `sharpness` and
`gradient_structure_tensor_xx_yy_xy` fields of `metadata.json`) is unchanged in
version 3; see
[`../formats/sift-file-format.md`](../formats/sift-file-format.md). The
reader handles both (legacy = one implicit `0-(N-1)` chunk with
`described_count = N`).

### Summary of differences

| Aspect | version 2 | version 3 |
|--------|----------|----------|
| `metadata.version` | `2` | `3` |
| `features/descriptors_metadata.json` | absent (all keypoints described) | present; records `described_count` (prefix length `[0, described_count)`) |
| Descriptor entries | single `features/descriptors.{feature_count}.128.uint8` | append-only chunks `features/descriptors.{start}-{end}.128.uint8` |
| Mutability | write-once, immutable | only descriptors (appended) and `content_hash.json` (rewritten) change; `metadata.json` stays immutable |
| `content_xxh128` | digest-of-digests over all entries | `xxh128(feature_set_xxh128 ‖ descriptor_prefix_xxh128 ‖ described_count)` |
| `feature_set_xxh128` | absent | stable id, invariant across appends |
| `descriptor_prefix_xxh128` | absent | hash of the described prefix |
| `component_xxh128` | absent | cached per-entry digests |

### Entry changes

`feature_tool_metadata.json.zst` is unchanged from version 2, including its
required `feature_options.image_to_gray`.

`metadata.json.zst` — `version` is `3`, and the entry carries version 2's
`sharpness` and `gradient_structure_tensor_xx_yy_xy` readings. It is written once and never
changes, including across descriptor appends: the mutable descriptor-coverage
count lives in `features/descriptors_metadata.json`, not here, so `metadata.json`
and `metadata_xxh128` stay constant as descriptors are appended.

`content_hash.json.zst` — version 3 splits identity into a part that is stable
across descriptor appends and a part that grows, so that references survive
expansion (see [Stable identity](#stable-identity) below). Alongside
`metadata_xxh128` and `feature_tool_xxh128` it defines:

* `feature_set_xxh128`: **Stable across the file's entire life.** XXH128 of the
    concatenation of these 16-byte digests, in order — the version 2
    `content_xxh128` inputs minus the descriptors:
    1. xxh128 of `feature_tool_metadata.json` (uncompressed)
    2. xxh128 of `metadata.json` (uncompressed)
    3. xxh128 of `features/positions_xy.{feature_count}.2.float32` (uncompressed)
    4. xxh128 of `features/affine_shapes.{feature_count}.2.2.float32` (uncompressed)
    5. xxh128 of `thumbnail_y_x_rgb.128.128.3.uint8` (uncompressed)

    Because `metadata.json` is immutable (it no longer carries `described_count`),
    this digest is constant for the file's entire life. It excludes descriptors, so
    it identifies *which keypoints from which image with which tool config* and
    never changes as descriptors are filled in. This is the recommended value for
    other files to reference.
* `descriptor_prefix_xxh128`: XXH128 of the concatenation of the per-chunk digests
    of the descriptor chunk entries covering `[0, described_count)`, in ascending
    `start` order (the xxh128 of the empty string when `described_count == 0`). For
    a given prefix length it is **reproducible from any later, expanded file** by
    truncating to that length, because chunks are append-only and never rewritten.
* `content_xxh128`: The exact-current-state hash, equal to
    `xxh128(feature_set_xxh128 ‖ descriptor_prefix_xxh128 ‖ u64_be(described_count))`
    (the two hashes as their 16-byte digests). Changes on every append. Because it
    is a pure function of the stable id and the reproducible prefix hash, a consumer
    that knows the `described_count` it pinned can re-derive and verify it against
    the current file.
* `component_xxh128`: (object) The individual xxh128 digest of each uncompressed
    entry (`feature_tool_metadata.json`, `metadata.json`, `positions_xy`,
    `affine_shapes`, `thumbnail`, and each `descriptors.{start}-{end}` chunk), keyed
    by entry name. Caches the per-entry digests so appending a chunk recomputes the
    hashes above without re-reading the large `positions_xy`, `affine_shapes`, or
    pre-existing descriptor chunks.

`features/descriptors.{start}-{end}.128.uint8.zst` replaces version 2's single
`features/descriptors.{feature_count}.128.uint8.zst`: one or more append-only
*chunk* entries that tile the described prefix `[0, described_count)`.

`features/descriptors_metadata.json.zst` — a new entry recording descriptor
coverage. JSON compressed with zstd, containing:

* `described_count`: (integer) The number of keypoints that have a stored
  descriptor. Descriptors cover the contiguous prefix `[0, described_count)` of
  the feature list, so `0 ≤ described_count ≤ feature_count`.

A descriptor append rewrites this entry (together with `content_hash.json`).
`described_count` is also derivable from the descriptor chunk entries
(`last_chunk.end + 1`, or `0` when there are none); a verifier cross-checks the
two. The value is folded into `content_xxh128` as a `u64`, so its integrity is
covered there. Version 1 and 2 files do not have this entry (all keypoints are
described).

### Incremental descriptor extraction

Version 3 makes a `.sift` file **growable**: a keypoint pool can be detected once and its
descriptors filled in incrementally, across multiple invocations, without re-reading or
rewriting the bulk data already on disk. This supports a *detect-many, describe-few*
workflow and coarse-to-fine matching (describe the largest keypoints first, finer ones only
where needed). This section is the normative on-disk definition.

#### Descriptor coverage

Descriptors cover the contiguous prefix `[0, described_count)` of the feature list. The
format's [descending-size feature ordering](../formats/sift-file-format.md#feature-ordering) makes this prefix exactly the
`described_count` largest features — the natural granularity for coarse-to-fine — so coverage
needs only the single `described_count` integer, with no sparse mask. A partial reader can
pull the top-K features (`positions_xy[0..K]`, `affine_shapes[0..K]`) together with exactly
the descriptor chunks covering `[0, K)`.

`positions_xy` and `affine_shapes` are written **once**, in full (shape `feature_count`);
only descriptors grow.

#### Descriptor chunk entries

Descriptors are stored as a sequence of append-only chunk entries, each named by the
**inclusive** keypoint-index range it covers:

```
features/descriptors.0-100.128.uint8.zst      # keypoints 0..=100   (101 rows)
features/descriptors.101-1000.128.uint8.zst   # keypoints 101..=1000 (900 rows)
features/descriptors.1001-4095.128.uint8.zst  # keypoints 1001..=4095
```

The leading `{start}-{end}` token replaces the row-count dimension of the
[extension shape convention](../formats/archive-container.md#entry-names-encode-shape-and-type): the row count is `end - start + 1`,
so an entry must contain `(end - start + 1) * 128 * 1` uncompressed bytes. The remaining
`.128.uint8` is the descriptor dimension and dtype as usual. Constraints:

* The first chunk starts at `0`.
* Chunks are contiguous and gap-free: each chunk's `start` equals the previous chunk's
  `end + 1`.
* The chunks jointly cover `[0, described_count - 1]`; equivalently
  `described_count = last_chunk.end + 1`, or `0` when there are no chunks.
* Chunks never overlap and are never rewritten once written.

A file with `described_count == 0` (detect-only) contains **no** `features/descriptors.*`
entries at all.

#### Appending a descriptor chunk

To describe keypoints `[a, b]` (where `a == described_count`):

1. Compute the descriptors for keypoints `a..=b`.
2. **Append** the immutable entry `features/descriptors.{a}-{b}.128.uint8.zst` to the ZIP.
   Existing entries are not touched.
3. Rewrite the two small mutable JSON entries:
   * `features/descriptors_metadata.json.zst`: set `described_count = b + 1`.
   * `content_hash.json.zst`: add the new chunk's digest to `component_xxh128`, then recompute
     `descriptor_prefix_xxh128` and `content_xxh128`. `feature_set_xxh128`, `metadata_xxh128`,
     and `metadata.json` are unchanged.

The ZIP-level mechanics of performing this append in place — how the central directory and
the replaced JSON entries are rewritten — and the locking or coordination needed for safe
concurrent writers are **implementation-defined**. The only requirement is that the resulting
file is a valid `.sift` as specified here, and that concurrent readers always observe a
consistent one.

#### Stable identity

A version 3 file's `content_xxh128` changes on every descriptor append, so other files that
pin a `.sift` by its whole-file hash (`.sfmr`, `.matches`, workspace caches) would see their
reference "break" the moment the file is expanded — even though the new file is a strict
superset of what they referenced. Reference `feature_set_xxh128` instead:

**Reference the stable id.** `feature_set_xxh128` is invariant for the file's entire life
and is recomputable from the immutable entries, so it is the correct value to record when a
consumer cares about *which keypoints* (the common case). It never breaks under descriptor
expansion or conversion to the version 2 layout. New and updated references SHOULD use it;
the `.sfmr`/`.matches`/workspace specs that currently store a `.sift` `content_xxh128` should
be updated to store `feature_set_xxh128`.

**Descriptor-dependent consumers verify the prefix.** A `.matches` file built from the first
`M` descriptors remains valid under expansion because chunks `[0, M)` are immutable. Such a
consumer should record `(feature_set_xxh128, M, descriptor_prefix_xxh128@M)` and re-verify by
recomputing `descriptor_prefix_xxh128` over `[0, M)` on the current file — which holds
regardless of how many additional descriptors were appended afterward. (This is also why no
whole-file hash history is needed: identity uses the stable hash, and descriptor state is
verified against the immutable prefix.)

### Image-to-gray conversion

`feature_options.image_to_gray` arrives in version 2 and is unchanged here. Its
definition (inputs, formula grammar, the optional `quantize_bits` and what each
feature tool records) is in
[sift-file-format.md § `feature_options.image_to_gray`](../formats/sift-file-format.md#feature_optionsimage_to_gray).

### Example

Against the version 1 example in
[`../formats/sift-file-format.md`](../formats/sift-file-format.md#using-cli-commands-to-pull-apart-a-sift-file),
a version 3 file of the same image, detected with 2464 keypoints
but only the first 1000 described so far, differs as follows — note `version`, the descriptor
*chunk* entries, the `features/descriptors_metadata.json` coverage file, and the extra
`content_hash.json` hash fields. The `metadata.json` is the version 1 file's with
`version` changed and version 2's two readings added (their values here are
illustrative):

```bash
$ unzip -l myimage.jpg.sift
 ...
 feature_tool_metadata.json.zst
 metadata.json.zst
 features/positions_xy.2464.2.float32.zst
 features/affine_shapes.2464.2.2.float32.zst
 features/descriptors.0-499.128.uint8.zst       # first describe-batch
 features/descriptors.500-999.128.uint8.zst     # appended later
 features/descriptors_metadata.json.zst
 thumbnail_y_x_rgb.128.128.3.uint8.zst
 content_hash.json.zst

$ jq . metadata.json
{
  "version": 3,
  "image_name": "myimage.jpg",
  "image_file_xxh128": "3748f9341bfdcc712beb2c5392664982",
  "image_file_size": 21584,
  "image_width": 270,
  "image_height": 480,
  "feature_count": 2464,
  "sharpness": 0.0041206,
  "gradient_structure_tensor_xx_yy_xy": [0.00093114, 0.00071402, -0.00008351]
}

$ jq . features/descriptors_metadata.json
{
  "described_count": 1000
}

$ jq . content_hash.json
{
  "metadata_xxh128": "a7b3c1d2e4f56789abcdef0123456789",
  "feature_tool_xxh128": "c220a90eb516a6654748c328f3403054",
  "feature_set_xxh128": "9f1c…",
  "descriptor_prefix_xxh128": "be77…",
  "content_xxh128": "4d0a…",
  "component_xxh128": {
    "feature_tool_metadata.json": "…",
    "metadata.json": "…",
    "features/positions_xy.2464.2.float32": "…",
    "features/affine_shapes.2464.2.2.float32": "…",
    "features/descriptors.0-499.128.uint8": "…",
    "features/descriptors.500-999.128.uint8": "…",
    "thumbnail_y_x_rgb.128.128.3.uint8": "…"
  }
}
```
