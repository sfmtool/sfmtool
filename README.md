# SfM Tool

[![CI](https://github.com/sfmtool/sfmtool/actions/workflows/ci.yml/badge.svg)](https://github.com/sfmtool/sfmtool/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/sfmtool/sfmtool/graph/badge.svg)](https://codecov.io/gh/sfmtool/sfmtool)

A tool for creating and exploring Structure from Motion (SfM) reconstructions,
built on [COLMAP](https://colmap.github.io/) and [OpenCV](https://opencv.org/).

Under construction - project is being restructured to better fit what it
evolved into.

## Components

- **`sfm` CLI** - Create workspaces, run SfM solves, and inspect reconstructions
- **SfM Explorer GUI** - View reconstructions in 3D, explore tracks and camera rays
- **Python bindings** - Access SfM functionality from Python via `sfmtool-py`

## Quick Start

```bash
# Create a workspace with your images
mkdir -p workspace/images
cp /path/to/images/*.jpg workspace/images/
cd workspace

# Initialize and solve
sfm ws init .
sfm solve -g images/

# Inspect the result
sfm inspect sfmr/[filename].sfmr
```

## Convert videos to image sequences

To try many different things with a particular 3D reconstruction, having the
data as a video complicates individual frame access. Therefore the `sfm` command expects you
to always convert to images first.

Here's an example of using [ffmpeg](https://www.ffmpeg.org/) to convert a
single video in the current directory into a sequence of images in a `frames`
subdirectory:

```bash
# Assumes there's one video in the current directory
VIDEO_FILE=$(echo *.mp4)

IMAGE_BASE=$(basename "$VIDEO_FILE")
IMAGE_BASE=${IMAGE_BASE%.*}

mkdir -p frames
ffmpeg -i "$VIDEO_FILE" \
    -qscale:v 2 \
    frames/"${IMAGE_BASE}_%04d.jpg"
```

## Building

This project uses [Pixi](https://pixi.sh/) for environment management with Rust and Python.

```bash
# Run the Sfm Explorer GUI
pixi run gui

# Run tests
pixi run test

# Serve docs locally
pixi run docs-serve
```

## Project Structure

```
crates/
  sfmtool-core/     Core project functionality
  sfm-explorer/     SfM Explorer GUI
  sfmtool-py/       Python bindings
  sfmr-format/      .sfmr reconstruction file I/O
  sfmr-colmap/      COLMAP integration
  sift-format/      SIFT feature file I/O
  matches-format/   Feature matches file I/O
```

## License

Apache-2.0
