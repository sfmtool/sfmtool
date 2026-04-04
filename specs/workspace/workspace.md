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
3. Every data format is inspectable by CLI commands (e.g. `sfm sift --print` or `sfm inspect`)
   and by the SfM Tool GUI.
4. We've defined file formats for SIFT features (`.sift`), feature matches (`.matches`),
   and SfM reconstructions (`.sfmr`).

## The `.sfm-workspace.json` File

### Creating a Workspace

```bash
# Initialize the current directory as a workspace
sfm init

# Initialize a new directory as a workspace
sfm init my_project

# Initialize with specific feature extraction settings
sfm init --feature-tool colmap --dsp my_project
sfm init --feature-tool opencv my_project
```

The `sfm init` command creates the directory (if needed) and writes `.sfm-workspace.json`.

### File Format

```json
{
  "version": 1,
  "feature_tool": "colmap",
  "feature_type": "sift",
  "feature_options": {
    "max_image_size": 4096,
    "max_num_features": 8192,
    "domain_size_pooling": false,
    "estimate_affine_shape": false,
    "peak_threshold": 0.006666666666666667,
    "edge_threshold": 10.0,
    "upright": false,
    "normalization": "L1_ROOT"
  },
  "feature_prefix_dir": "features/sift-colmap-d1245b460906df27ee4730273e0aba41"
}
```

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `version` | integer | Format version of the workspace config file. Must be `1` |
| `feature_tool` | string | Feature extraction tool: `"colmap"` or `"opencv"` |
| `feature_type` | string | Feature type: `"sift"` (future: `"surf"`, `"superpoint"`, etc.) |
| `feature_options` | object | All parameters that affect feature output. Keys are tool-defined. See the `.sift` format spec for details |
| `feature_prefix_dir` | string | Relative path to the features subdirectory, including a content hash of the tool configuration |

### Feature Prefix Directory

The `feature_prefix_dir` encodes both the feature type and the exact tool configuration into the
path. For example:

```
features/sift-colmap-d1245b460906df27ee4730273e0aba41
```

This is structured as `features/{feature_type}-{feature_tool_xxh128}`, where:

- `feature_type` identifies the descriptor format (e.g., `sift-colmap`, `sift-opencv`)
- `feature_tool_xxh128` is a hash derived from `feature_tool`, `feature_type`, and
  `feature_options`, computed once during `sfm init`. The spec does not prescribe a specific
  serialization algorithm — the implementation decides how to hash these values.
  Implementations should be deterministic so that reinitializing a workspace with the same
  settings produces the same hash and reuses cached features

This design means that if you change feature extraction settings (e.g., enable domain size pooling
or change `max_num_features`), the hash changes and features are stored in a separate directory.
Previously extracted features are not overwritten or invalidated — they remain available if you
switch back.

### Protection Against Nesting

`sfm init` prevents creating a workspace inside an existing workspace, since nested workspaces
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
  helpful message suggesting `sfm init`.

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
      "feature_type": "sift",
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
accepting it.

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
sfm init --dsp my_project

# Extract features — workspace provides the tool and options
sfm sift --extract my_project/frames/

# Solve — workspace provides feature locations
sfm solve -i my_project/frames/

# Try again with fewer features — workspace still provides the base config
sfm solve -i --max-features 500 my_project/frames/

# Try global SfM
sfm solve -g my_project/frames/

# Inspect a result — workspace resolved from the .sfmr file
sfm inspect my_project/sfmr/*.sfmr

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
sfm init my_project
sfm sift --extract my_project/frames/
sfm solve -i my_project/frames/
# Features are in frames/features/sift-colmap-a1b2c3d4...

# Try domain size pooling — reinitialize the workspace
sfm init --dsp --force my_project
sfm sift --extract my_project/frames/
sfm solve -i my_project/frames/
# Features are in frames/features/sift-colmap-e5f6a7b8... (different hash)

# Switch back to the original settings
sfm init --force my_project
# The original features in frames/features/sift-colmap-a1b2c3d4... are still there
# No need to re-extract — sfm sift will find them cached
sfm sift --extract my_project/frames/
```

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

## Design Principles

1. **Convention over configuration**: Sensible defaults mean most projects need only `sfm init`
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
