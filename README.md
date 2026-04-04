# SfM Tool

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
sfm init .
sfm solve -g images/

# Inspect the result
sfm inspect sfmr/[filename].sfmr
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
  sfmtool-gui/      SfM Explorer GUI
  sfmtool-py/       Python bindings
  sfmr-format/      .sfmr reconstruction file I/O
  sfmr-colmap/      COLMAP integration
  sift-format/      SIFT feature file I/O
  matches-format/   Feature matches file I/O
```

## License

Apache-2.0
