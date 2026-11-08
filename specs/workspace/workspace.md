# Workspace Specification

## Overview

An sfmtool workspace is a directory that serves as the root of a Structure from Motion
(SfM) project. It ties together images, extracted features, and reconstructions into a coherent unit. The
workspace makes iterative, interactive exploration of 3D structure easy and fun —
you shouldn't have to repeat options or remember paths between commands.

A workspace is identified by the presence of a `.sfm-workspace.json` file at its root.

## Interactive Structure from Motion Pipelines

SfM works across multiple stages, starting from a set of images and
discovering the 3D relationships between them over time. The result is an estimate
of a point cloud of surface points and the poses and parameters of all the image cameras.
Random sampling and consensus (RANSAC) is used at multiple stages of the pipeline,
and there are many ways that spurious solutions can sneak in.

SfM pipelines have a large number of tweakable steps and parameter values. The results
you get depend heavily on the input dataset, there isn't a single fixed pipeline that
will give great results across the board. This means that your process can look like

1. Set or adjust the options and parameter values for the pipeline.
2. Run the pipeline on your images.
3. Evaluate the output. If it's not good enough, go back to 1.

Instead of trying to design a highly reliable pipeline, the purpose of the workspace is to make
it easier to inspect and wire together the stages of the pipeline reactively. If you've computed
part of the pipeline and it goes off course, don't restart from the beginning. Instead, keep the
partial results to inspect and as inputs to build on. If you've got a rough approximation
of camera positions, you should be able to start from that, not begin afresh from the images.

## Data Files in a Workspace

A workspace holds data files for an SfM pipeline, with the goal that no data file spans
multiple steps, and different pipelines or runs through the pipeline can overlap and reuse
data. Some examples of this include:

1. The only pipeline step for video files is to convert them to images. That way, images
   are always individually referenceable as file paths relative in the workspace.
2. Every data format is intended to be written once and then left untouched. There are
   no mutable database files to incrementally update.
3. Every data format is inspectable by CLI commands (e.g. `sfm inspect`)
   and by the SfM Tool GUI.
4. We've defined file formats for SIFT features (`.sift`), feature matches (`.matches`),
   and SfM reconstructions (`.sfmr`).

## The `.sfm-workspace.json` File

### Creating a Workspace

```bash
# Initialize the current directory as a workspace
sfm ws init

# Initialize a new directory as a workspace
sfm ws init my_project

# Initialize with specific feature extraction settings
sfm ws init --feature-tool colmap --dsp my_project
sfm ws init --feature-tool opencv my_project
sfm ws init --feature-tool sfmtool my_project   # sfmtool's own Rust SIFT
sfm ws init --image-to-gray G --quantize-bits 8 my_project

# Move a version 1 workspace to version 2 and a new features folder
sfm ws upgrade my_project
```

The `sfm ws init` command creates the directory (if needed) and writes `.sfm-workspace.json`.
When `--feature-tool` is omitted, the default is `sfmtool` (its own Rust SIFT), so a
bare `sfm ws init` produces a sfmtool-shaped `feature_options` block — not the
COLMAP-shaped one shown in the File Format example below.

### File Format

```json
{
  "version": 2,
  "feature_tool": "colmap",
  "feature_type": "sift-colmap",
  "feature_options": {
    "max_image_size": 4096,
    "max_num_features": 8192,
    "domain_size_pooling": false,
    "estimate_affine_shape": false,
    "peak_threshold": 0.006666666666666667,
    "edge_threshold": 10.0,
    "upright": false,
    "normalization": "L1_ROOT",
    "use_gpu": true,
    "image_to_gray": {
      "formula": "0.2126*R + 0.7152*G + 0.0722*B",
      "quantize_bits": 8
    }
  },
  "feature_prefix_dir": "features/sift-colmap-c2405a72d0875a97223cbeccadab3291"
}
```

The example shows the `colmap` option shape. Each tool defines its own
`feature_options` keys: `opencv` uses the OpenCV SIFT parameter names
(`nfeatures`, `nOctaveLayers`, `contrastThreshold`, `edgeThreshold`, `sigma`),
and `sfmtool` uses the Rust SIFT's parameters (`octave_layers`,
`contrast_threshold`, `edge_threshold`, `max_num_features`, descriptor and
orientation settings, etc.). `use_gpu` (COLMAP only) is excluded from the
feature-cache hash since it does not affect feature output. In version 2 every
tool's `feature_options` also carry the `image_to_gray` object
([Version 2: `image_to_gray`](#version-2-image_to_gray)).

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `version` | integer | Format version of the workspace config file, `1` or `2`; a writer writes `2` ([Versions](#versions)) |
| `feature_tool` | string | Feature extraction tool: `"colmap"`, `"opencv"`, or `"sfmtool"` (the Rust SIFT in `sfmtool-core`; the default when `--feature-tool` is omitted) |
| `feature_type` | string | Feature type, which names the tool and any option that changes the output: `"sift-sfmtool"`, `"sift-opencv"`, or `"sift-colmap"` with `-dsp` for domain-size pooling and `-max<N>` for a feature cap other than 8192, as in `"sift-colmap-dsp-max4096"`; a workspace `sfm undistort` writes has `"sift-sfmtool-undistort"`. It is the start of the last component of `feature_prefix_dir` |
| `feature_options` | object | All parameters that affect feature output. Keys are tool-defined. See the `.sift` format spec for details |
| `feature_prefix_dir` | string | Relative path to the features subdirectory, including a content hash of the tool configuration |

The file has no other keys. An optional second file beside it in the workspace
root, `.sfm-named-image-lists.json`, names sets of the workspace's images by
pattern and frame range; it is described in
[`named-image-lists.md`](named-image-lists.md). The commands that take
`-r/--range` accept a set's name there
([`../cli/README.md`](../cli/README.md#the-shared--r--range-option)).

### Feature Prefix Directory

The `feature_prefix_dir` encodes both the feature type and the exact tool configuration into the
path. For example:

```
features/sift-colmap-d1245b460906df27ee4730273e0aba41
```

This is structured as `features/{feature_type}-{feature_cache_hash}`, where:

- `feature_type` identifies the descriptor format (e.g., `sift-colmap`, `sift-opencv`, `sift-sfmtool`)
- `feature_cache_hash` is a hash derived from `feature_tool`, `feature_type`, and
  `feature_options`, computed once when the folder is created
  ([The recorded folder name is authoritative](#the-recorded-folder-name-is-authoritative)).
  The spec does not prescribe a specific
  serialization algorithm — the implementation decides how to hash these values. It is a
  different value from the `feature_tool_xxh128` field stored inside each `.sift` file, which is
  the hash of that file's `feature_tool_metadata.json` bytes (see
  [Feature tool hash computation](../formats/sift-file-format.md#feature-tool-hash-computation)).
  Implementations should be deterministic so that reinitializing a workspace with the same
  settings produces the same hash and reuses cached features

This design means that if you change feature extraction settings (e.g., enable domain size pooling
or change `max_num_features`), the hash changes and features are stored in a separate directory.
Previously extracted features are not overwritten or invalidated — they remain available if you
switch back.

### Versions

`.sfm-workspace.json` has two versions. A reader reads both, and a writer writes
version 2.

| Aspect | version 1 | version 2 |
|--------|-----------|-----------|
| `version` | `1` | `2` |
| The grey conversion | the `sfmtool` tool records it as a `gray_formula` string in `feature_options`; `colmap` and `opencv` do not state it | every tool's `feature_options` carry the `image_to_gray` object at their top level, and no `gray_formula` |
| The other fields | as above | unchanged |

The version says which spelling a file uses without guessing from its keys, and
keeps one spelling per version rather than carrying `gray_formula` forward
beside the object.

#### Version 2: `image_to_gray`

A version 2 file's `feature_options` carry the same `image_to_gray` object a
version 2 `.sift` file records
([sift-file-format.md § `feature_options.image_to_gray`](../formats/sift-file-format.md#feature_optionsimage_to_gray)),
at their top level, for every feature tool, and each `.sift` file's
`feature_options` are a copy of the workspace's. The object is
`{"formula": …, "quantize_bits": …}`: a formula over the colour samples `R`, `G`
and `B` on `[0, 1]`, and an optional bit depth from 1 to 16 its value is rounded
to.

| `feature_tool` | `feature_options.image_to_gray` |
|----------------|---------------------------------|
| `sfmtool` | `{"formula": "0.2126*R + 0.7152*G + 0.0722*B"}` by default; a setting, which the `sfmtool` extractor honours for any formula and any `quantize_bits`, chosen with `sfm ws init --image-to-gray` and `--quantize-bits` |
| `colmap` | `{"formula": "0.2126*R + 0.7152*G + 0.0722*B", "quantize_bits": 8}`, fixed by the tool |
| `opencv` | `{"formula": "0.299*R + 0.587*G + 0.114*B", "quantize_bits": 8}`, fixed by the tool |
| `sfmtool-undistort` | its source's conversion ([`sfm undistort`](#sfm-undistort)); recorded, not settable |

For `colmap` and `opencv` the object records what the tool does rather than
choosing it, and extraction refuses one that differs.

#### Loading checks

`load_workspace_config`, the one reader of the file, refuses at load, naming the
file and what is wrong:

- a file with no `version` key;
- a `version` other than `1` or `2`;
- a version 2 file whose `feature_options` carry `gray_formula` at the top
  level, carry no top-level `image_to_gray`, or carry one that does not parse;
- a version 1 file whose `feature_options` carry `image_to_gray`, since the
  object belongs to version 2.

The version 2 rules apply to the top level of `feature_options` only;
`sfmtool-undistort`'s nested `source_feature_options` is a copy of its source's
record and may hold a version 1 `gray_formula`.

#### The recorded folder name is authoritative

`feature_prefix_dir` is stored in the file so that nothing downstream has to
compute it: every command, the viewer and every `.sfmr` use the recorded name as
it is, and `load_workspace_config` returns it without computing it again. A
folder name is computed only when a folder is created: by `sfm ws init`,
`sfm ws upgrade`, `sfm undistort`, `sfm sift --extract --tool` outside a
workspace, and the viewer's conversion of a version 1 record or switch of
another tool's record to the `sfmtool` defaults. Because `image_to_gray` is part
of the options a folder is named from, every folder created under version 2 is
new, for every tool, even with otherwise identical settings. Two writers that
name the same options differently cache the features twice, never wrongly.

What this gives up: a hand edit of `feature_options` does not move the folder,
so it can put `.sift` files made under different settings into one folder. Each
`.sift` file records its own `feature_tool_metadata`, so that is detectable
afterwards. To change the options, run `sfm ws init --force` or
`sfm ws upgrade`, which create a new folder.

#### Versions never mix

**No `.sift` file is written from version 1 settings, and no version 2 `.sift`
file goes into a folder that holds version 1 files.** Version 1 settings are a
version 1 workspace file or a record whose `feature_options` carry no
`image_to_gray`, such as a `.sfmr` made from a version 1 workspace. A version 1
workspace moves to version 2 only through `sfm ws upgrade` or `sfm ws init`,
both of which create a new folder.

**Reading version 1 files stays allowed everywhere.** A command that reads
`.sift` or `.sfmr` files in a version 1 workspace reads them in its recorded
folder, a `gray_formula` reading as `{"formula": <gray_formula>}` and a `colmap`
or `opencv` file with no stated conversion as its tool's row. A command that
only needed features already there, and finds every `.sift` present and
current, writes nothing and so does not refuse.

Every `.sift` file is written by one of these, and the rule is checked in each:

| Writer | Reached from | Reads its settings from | Writes into | Refuses when |
|--------|--------------|-------------------------|-------------|--------------|
| `image_files_to_sift_files`, which writes each `.sift` that is missing or older than its image | `sfm sift --extract`, and implicitly `sfm solve`, `sfm match` and `sfm to-colmap-db` | the options and folder its caller passes: the workspace file's, read through `load_workspace_config`; for `sfm to-colmap-db`, the reconstruction's record, a version 1 record converted in memory first (below); with `sfm sift --extract --tool` outside a workspace, the tool's defaults | the recorded `feature_prefix_dir`; for a converted record, the folder the converted options name; with `--tool`, a folder named from the tool's defaults | the tool is `sfmtool-undistort`; the options carry no `image_to_gray`; a `.sift` already in the folder is version 1; a `colmap` or `opencv` object differs from its tool's row |
| `build_sift_files` in `sfmtool-core` | its Python binding and the viewer | the options and folder it is handed; it never reads the workspace file | the folder it is handed | the options carry no `image_to_gray`; a `.sift` already in the folder is version 1 |
| the viewer's *Build SIFT Files*, through `build_sift_files` | the scene tree and the wire | the node's `.sfmr` record, not the workspace file, so the files go where the node looks for them | a version 2 `sfmtool` record: its recorded folder. A version 1 `sfmtool` record of an embedded-patches node: a new folder named from the record converted in memory as `sfm ws upgrade` converts one. A record naming another tool: the `sfmtool` defaults' folder | a version 1 record on a `sift_files` node ([index-files.md](../gui/index-files.md#building-the-sift-files)) |
| `sfm undistort` | `sfm undistort` | the input `.sfmr`'s record and the undistorted images | a new folder in its output | never, for version reasons |

Each writer checks only when it has something to write. The two that check a
folder for version 1 files, `image_files_to_sift_files` and `build_sift_files`,
both call `version_1_sift_files`
([`features/sift/files.rs`](../../crates/sfmtool-core/src/features/sift/files.rs),
bound in `sfmtool.sift`), so they agree on what counts: a file in the folder
itself, not a subfolder, whose name ends in `.sift` and whose metadata reads
with a version below 2. A file whose metadata cannot be read says nothing about
the folder and is passed over, and a folder that is not there holds nothing.

**`sfm to-colmap-db`** reads its settings from the `.sfmr`, not the workspace
file, so `sfm ws upgrade` cannot change them. When its record holds version 1
settings and some `.sift` file is missing or older than its image, it converts
the record in memory as the viewer does (`upgrade_feature_options`), extracts
into the folder named from the converted options, and reads every `.sift` file
from there, printing one line that names the old and new folders; the `.sfmr`
is not rewritten. This holds for every feature tool and both feature sources:
the export writes keypoints and descriptors from the `.sift` files and no
matches or tracks, so it reads all its features from one folder and needs no
correspondence with the reconstruction's feature indexes. With every file
present and current, nothing is converted.

The viewer differs on a `sift_files` node, and refuses a version 1 record there:
the node keeps using the reconstruction's feature indexes against the folder it
builds, so files in a new folder would break that correspondence.
`sfm to-colmap-db` has no such dependency.

<a id="sfm-undistort"></a>**`sfm undistort`** writes its own workspace
(`feature_tool` `sfmtool-undistort`, `feature_options` holding the source's
`source_feature_tool` and `source_feature_options` and the warp's `fit` and
`filter`) as version 2, with a top-level `image_to_gray` converted from the
source as `sfm ws upgrade` converts one: the source's own top-level object, else
its `gray_formula` as `{"formula": …}`, else the source tool's row, a version 1
`sfmtool-undistort` source being converted from its own nested record the same
way. Its `.sift` files are version 2 with that object, and their readings are
computed by decoding the undistorted image files it wrote, so they describe
those images. Its folder, `features/sift-sfmtool-undistort-{hash}`, is named over
its whole `feature_options` when it creates it, so undistorting again under
version 2 writes a new folder beside any older output.

#### Refusals

| Refusal | Raised by | Its message |
|---------|-----------|-------------|
| Version 1 settings | `image_files_to_sift_files`, `build_sift_files` | names the workspace file or the record, says it holds version 1 settings, and, for a workspace file, says to run `sfm ws upgrade`, which moves the workspace to version 2 and a new folder. For a `.sfmr`'s record it says the record is the `.sfmr`'s own, and never suggests `sfm ws upgrade` |
| Version 1 files in the folder | `image_files_to_sift_files`, `build_sift_files` | names the folder and the version 1 `.sift` files in it (the first five, and a count of the rest), and says to remove those files from the folder and run again, which extracts version 2 files in their place. The folder is checked only for version 2 settings, since version 1 settings are refused first, and `sfm ws upgrade` prints "nothing to do: already version 2" for a version 2 workspace file and never changes a `.sfmr`'s record, so the message is the same for every caller. It is the `Display` of `Version1SiftFiles`, which `sfmtool_core::features::sift::version_1_sift_files` returns and `sfmtool.sift.version_1_sift_files` binds |
| An `sfmtool-undistort` tool | `image_files_to_sift_files` | says the files of an undistorted workspace are made by `sfm undistort` and cannot be extracted, and to run `sfm undistort` again |
| `colmap` or `opencv` conversion differs | `image_files_to_sift_files` | names the tool and the one conversion it makes |
| Version 1 record on a `sift_files` node | the viewer | says the feature indexes name features of the old files, which a new folder would not hold, and suggests converting to embedded patches |
| Invalid workspace file at load | `load_workspace_config` | names the file and what is wrong: no `version`, the version found, or the key that breaks the version's rules |
| `sfm ws upgrade` cannot upgrade | `sfm ws upgrade`, changing nothing | names the file and says which: no workspace file, no `version`, a version other than 1 or 2, a `feature_tool` other than `sfmtool`, `colmap` or `opencv` (an `sfmtool-undistort` workspace is told to run `sfm undistort` again, since its `.sift` files cannot be extracted again), or a `gray_formula` that does not parse |

#### Migration

No command rewrites the workspace file unless the user runs one of these:

- **`sfm ws upgrade`** ([ws-upgrade-command.md](../cli/workspace/ws-upgrade-command.md))
  acts on the workspace found by the usual discovery from the directory it is
  given, or the current directory. It rewrites a version 1 file as version 2
  keeping every option, including hand-edited ones such as `peak_threshold` or
  `max_image_size`; converts `sfmtool`'s `gray_formula` into
  `image_to_gray: {"formula": <gray_formula>}` and gives a `colmap` or `opencv`
  workspace its tool's row; names a new `feature_prefix_dir` from the new
  options; prints the old and new folders; and leaves the old folder, its
  `.sift` files and the named image lists file on disk. On a version 2 file it
  says "nothing to do: already version 2" and changes nothing.
- **`sfm ws init` and `sfm ws init --force`** write version 2 with fresh
  options from the tool's defaults and the flags, the tool's `image_to_gray`
  row, and a new folder named from them. `--force` drops any hand-edited option,
  which is why the refusals point at `sfm ws upgrade`. The old folder stays on
  disk, and the named image lists file is not touched.
- **`sfm ws init --image-to-gray FORMULA --quantize-bits B`** sets the
  `sfmtool` tool's conversion before the folder is named. The formula is checked
  against the grammar and `B` must be an integer from 1 to 16; with
  `--feature-tool colmap` or `opencv` either flag is an error.

**`.sfmr` files.** `workspace.contents` copies the workspace's `feature_tool`,
`feature_type`, `feature_options` and `feature_prefix_dir`, so a `.sfmr` written
from a version 2 workspace holds the object in
`workspace.contents.feature_options`, and one written from a version 1 workspace
holds what that file held. `.sfmr` files that name an old folder still find
their `.sift` files there through their own record after an upgrade.

A `.sfmr` of version 15 carries each image's sharpness and gradient structure
tensor under one `image_to_gray`. A writer takes the conversion from the value
it writes, or else from `workspace.contents.feature_options.image_to_gray`, and
copies each image's readings from its version 2 `.sift` file when that file
records the same conversion; see
[`../formats/sfmr-file-format.md` § Where a writer gets the values](../formats/sfmr-file-format.md#where-a-writer-gets-the-values).

### Protection Against Nesting

`sfm ws init` prevents creating a workspace inside an existing workspace, since nested workspaces
create ambiguity about which workspace a file belongs to. Use `--force` to override this check
if you have a specific reason to nest.

## Workspace Directory Layout

A workspace has no mandatory internal structure beyond the `.sfm-workspace.json` file. Images can
live in any subdirectory. However, a typical workspace looks like this:

```
my_project/
├── .sfm-workspace.json
├── frames/                              # Images from video extraction
│   ├── scene_0001.jpg
│   ├── scene_0002.jpg
│   └── ...
├── photos/                              # Images from a camera
│   ├── DSC_0001.JPG
│   └── ...
├── frames/features/sift-colmap-d124.../  # Extracted features (auto-created)
│   ├── scene_0001.jpg.sift
│   ├── scene_0002.jpg.sift
│   └── ...
├── photos/features/sift-colmap-d124.../  # Features for photos (auto-created)
│   ├── DSC_0001.JPG.sift
│   └── ...
├── matches/                             # Default matches output directory
│   └── 20250115-00-sequential_1-50.matches
├── tvg-matches/                         # Matches with two-view geometries
│   └── 20250115-01-sequential_1-50-verified.matches
└── sfmr/                               # Default reconstruction output directory
    ├── 20250115-00-frames_1-50.sfmr
    ├── 20250115-00-frames_1-50-sift-index.kdf   # SIFT index (built on request)
    ├── 20250115-01-frames_1-100.sfmr
    └── 20250116-00-photos_1-88.sfmr
```

### Reconstruction Output

`.sfmr` reconstruction files can be placed anywhere within the workspace. When `sfm solve` is
run without an explicit output path, it writes to the `sfmr/` directory at the workspace root.
This is a default convention, not a requirement — commands that consume `.sfmr` files locate
the workspace through the embedded workspace reference, not by assuming a fixed output location.

### Feature Storage Convention

Features are stored alongside their source images, not in a single flat directory. The
`feature_prefix_dir` is appended to each image's parent directory:

```
{workspace}/{image_parent}/{feature_prefix_dir}/{image_basename}.sift
```

For an image at `frames/scene_0001.jpg` with `feature_prefix_dir` of
`features/sift-colmap-d1245b460906df27ee4730273e0aba41`, the `.sift` file is at:

```
frames/features/sift-colmap-d1245b460906df27ee4730273e0aba41/scene_0001.jpg.sift
```

This convention means:
- Multiple image directories within one workspace each get their own feature subdirectory
- Features stay close to their source images in the filesystem
- Different feature configurations coexist without conflict

### The SIFT Index

A reconstruction's descriptors can be indexed into one
[`.kdf` file](../formats/kdf-file-format.md), so that a patch of one photograph
can be looked for in every other one
([the constellation query](../core/features/kdf-constellation-query.md)). That
index is the **`.sfmr` file's sibling** and takes its name from it: the index of
`<dir>/<stem>.sfmr` is `<dir>/<stem>-sift-index.kdf`.

```
sfmr/20250115-00-frames_1-50.sfmr
sfmr/20250115-00-frames_1-50-sift-index.kdf
```

Beside the reconstruction rather than beside the features because the corpus it
holds is a statement about one reconstruction's images, in that
reconstruction's order: a match it returns names an image by its row in the
index and the observation that match becomes names an image by its row in the
reconstruction, and the two have to be the same number. Under this convention
two reconstructions saved in one directory have two indexes, and a
reconstruction whose images span several image directories still has one index
in one obvious place.

The file is optional and nothing creates it as part of `sfm ws init` or `sfm
sift`. The SfM Explorer opens the one beside a loaded `.sfmr` when it is there,
says on the node's Scene tree row whether it is still an index of that
reconstruction, and offers to build one where a person finds they need it
([`../gui/sift-index.md`](../gui/sift-index.md)).

## Workspace Discovery

Many commands need to find the workspace without being told explicitly. sfmtool uses upward
directory search: starting from the directory containing the input files, it walks up the
directory tree until it finds a `.sfm-workspace.json` file.

```
/home/user/my_project/.sfm-workspace.json    <-- found
/home/user/my_project/frames/scene_0001.jpg  <-- starting from here
```

This is analogous to how `git` finds the `.git` directory.

### Discovery in Commands

Different commands discover the workspace in different ways depending on their inputs:

- **`sfm sift --extract`**: Finds the workspace from the common parent of the provided image
  paths. If no workspace is found and `--tool` is not specified, the command fails with a
  helpful message suggesting `sfm ws init`.

- **`sfm solve`**: Finds the workspace from the common parent of the image directories being
  solved. The workspace provides feature tool settings needed to locate `.sift` files.

- **Commands that take `.sfmr` files** (e.g., `sfm inspect`, `sfm xform`, the GUI viewer):
  Resolve the workspace from metadata embedded in the `.sfmr` file itself (see next section).

## Workspace References in `.sfmr` Files

Every `.sfmr` file embeds workspace information in its `metadata.json.zst` under the `workspace`
key:

```json
{
  "workspace": {
    "absolute_path": "/home/user/my_project",
    "relative_path": "../..",
    "contents": {
      "feature_tool": "colmap",
      "feature_type": "sift-colmap",
      "feature_options": { ... },
      "feature_prefix_dir": "features/sift-colmap-d1245b460906df27ee4730273e0aba41"
    }
  }
}
```

This embeds a snapshot of the workspace configuration at the time the `.sfmr` file was written,
plus path information for relocating the workspace later.

### Path Resolution Strategy

When a `.sfmr` file is opened, the workspace directory is resolved using a three-step fallback
strategy:

1. **Relative path**: Join the `.sfmr` file's parent directory with `workspace.relative_path`.
   Check that the result contains `.sfm-workspace.json`. This handles the common case where the
   `.sfmr` file is inside the workspace or in a known relative location.

2. **Absolute path**: Try `workspace.absolute_path` directly. This handles the case where the
   `.sfmr` file has been moved outside the workspace but the workspace hasn't moved.

3. **Upward search**: Search upward from the `.sfmr` file's directory for
   `.sfm-workspace.json`. This handles the case where the `.sfmr` file is somewhere inside the
   workspace but neither the relative nor absolute paths are valid (e.g., after the workspace
   was moved to a new machine).

Each step validates that the candidate directory actually contains `.sfm-workspace.json` before
accepting it. A `.kdf` file's source verification resolves the workspace it records by the same
three steps, through the same function (`resolve_workspace_dir` in
[archive-io-crate.md](../formats/archive-io-crate.md)).

### Why Both Relative and Absolute Paths?

The dual-path approach handles different scenarios:

| Scenario | Which path works |
|----------|-----------------|
| `.sfmr` is inside the workspace, nothing moved | Relative path |
| Whole workspace copied to another machine | Relative path |
| `.sfmr` copied out of the workspace | Absolute path |
| Workspace moved, `.sfmr` still inside it | Upward search |

### Image Path Resolution

All image paths stored in the `.sfmr` file (`images/names.json.zst`) are relative to the
workspace directory, not relative to the `.sfmr` file. Once the workspace is resolved, an
image path like `frames/scene_0001.jpg` resolves to
`{resolved_workspace}/frames/scene_0001.jpg`.

This separation between "where is the workspace" and "where are images within the workspace"
is what makes `.sfmr` files portable. The workspace location can change, but the internal
structure stays consistent.

### Feature File Resolution

Similarly, `.sift` feature files are located by combining the workspace directory, the image's
parent directory, the `feature_prefix_dir`, and the image basename:

```
{workspace}/{image_parent}/{feature_prefix_dir}/{image_basename}.sift
```

The `feature_prefix_dir` is stored in the `.sfmr` metadata, so a reconstruction always knows
which specific feature configuration was used, even if the workspace has since been
reinitialized with different settings.

## How Commands Use the Workspace

The workspace eliminates repetition across commands. Here's what a session looks like:

```bash
# One-time setup
sfm ws init --dsp my_project

# Extract features — workspace provides the tool and options
sfm sift --extract my_project/frames/

# Solve — workspace provides feature locations
sfm solve -i my_project/frames/

# Try again with fewer features — workspace still provides the base config
sfm solve -i --max-features 500 my_project/frames/

# Try global SfM
sfm solve -g my_project/frames/

# Inspect a result — workspace resolved from the .sfmr file
sfm inspect my_project/sfmr/20260101-00-solve.sfmr

# View in the GUI — workspace resolved from the .sfmr file
pixi run gui -- my_project/sfmr/*.sfmr
```

Without the workspace, `sfm sift` would need `--tool colmap --dsp` every time. `sfm solve`
would need to know where features are stored. The GUI would need to be told where images live.
The workspace makes all of this implicit.

## Workspace Lifecycle

### Changing Feature Settings

The workspace config points to one feature configuration at a time, but the hash-based feature
directories mean you can freely switch between configurations without losing previous work.

```bash
# Start with default COLMAP settings
sfm ws init my_project
sfm sift --extract my_project/frames/
sfm solve -i my_project/frames/
# Features are in frames/features/sift-colmap-a1b2c3d4...

# Try domain size pooling — reinitialize the workspace
sfm ws init --dsp --force my_project
sfm sift --extract my_project/frames/
sfm solve -i my_project/frames/
# Features are in frames/features/sift-colmap-e5f6a7b8... (different hash)

# Switch back to the original settings
sfm ws init --force my_project
# The original features in frames/features/sift-colmap-a1b2c3d4... are still there
# No need to re-extract — sfm sift will find them cached
sfm sift --extract my_project/frames/
```

A version 1 workspace's folder cannot be returned to this way: `sfm ws init`
writes version 2, whose options carry `image_to_gray`, so its folder name is
always a new one, and `.sfmr` files from the version 1 workspace keep reading
their own folder through their records.

Each configuration produces a different `feature_prefix_dir` hash, so its features are stored
in a separate directory. Reinitializing the workspace changes which configuration is active
but does not touch any existing feature directories. This makes it cheap to experiment: try
different settings, compare the resulting reconstructions, and switch back without re-extracting.

### Existing `.sfmr` Files After Reinitializing

Each `.sfmr` file embeds a snapshot of the workspace configuration at the time it was created,
including the `feature_prefix_dir`. This means existing `.sfmr` files continue to resolve their
feature files correctly even after the workspace is reinitialized with different settings. The
`.sfmr` file doesn't depend on the current workspace config — it carries its own copy.

### Recovering Previous Settings

Since every `.sfmr` file embeds the full workspace configuration from when it was created
(the `workspace` key in `metadata.json.zst`), you can inspect any previous reconstruction to
recover the exact feature tool, options, and prefix directory that produced it. This acts as
an implicit history of workspace configurations — even if you've reinitialized the workspace
many times, each `.sfmr` file records what settings were active when it was built.

### Cleanup

Old feature directories are never deleted automatically. This is by design — they may be
referenced by existing `.sfmr` files, or you may want to switch back to that configuration
later. If you're sure a feature directory is no longer needed, delete it manually.

## Camera Intrinsics

A workspace can commit known-good camera intrinsics through optional
`camera_config.json` files. Resolution is closest-ancestor-wins from the image's
parent directory up to the workspace root. When present, the file is the sole
source of camera setup for those images, and `--camera-model` on the command
line is rejected for the invocation. See
[`camera-config.md`](camera-config.md) for the file format and semantics.

Multi-sensor rigs are described by an optional `rig_config.json` at the
workspace root; see [`rig-config.md`](rig-config.md).

## Design Principles

1. **Convention over configuration**: Sensible defaults mean most projects need only `sfm ws init`
   with no options.

2. **Don't repeat yourself**: Settings that apply to the whole project are stored once in the
   workspace, not passed on every command.

3. **Non-destructive iteration**: Changing feature settings creates a new feature directory
   rather than overwriting the old one. Multiple reconstructions coexist as separate `.sfmr`
   files.

4. **Portability**: Workspace-relative paths in `.sfmr` files mean reconstructions work across
   machines and operating systems.

5. **Discoverable**: Upward directory search means you don't need to be in the workspace root
   to run commands — any subdirectory works.

6. **Minimal footprint**: The workspace is a single JSON file. There's no database, no lock
   file, no hidden state beyond `.sfm-workspace.json`.
