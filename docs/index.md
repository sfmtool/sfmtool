# The Structure from Motion Tool

## About this Project

The goal of this project is to make creating and exploring Structure from Motion (SfM) fun.
I had a bunch of vacation datasets of interesting scenes sitting around, ready for photogrammetry and
Gaussian splats, but using existing tools wasn't as fun as I imagined it could be. I
also wanted a project to cut my teeth on agentic AI coding tools with no expectations
or restrictions. The result is this project, and I hope you enjoy it if you give it a try!

SfM Tool builds heavily on the work of others, especially [COLMAP](https://colmap.github.io/)
and [OpenCV](https://opencv.org/). Most of what's different is around the workflow you use
to create SfM reconstructions.

### The `sfm` CLI command

The CLI command `sfm` is the interface for creating and evaluating reconstructions.
Every reconstruction must be performed within an SfM workspace. This opinionated approach
makes it easier and more predictable to use the command, and means you don't need to
repeat options redundantly while using it.

Here's the simplest way to create a workspace and perform an SfM reconstruction with
a small image dataset.

```bash
# Create the workspace and an images directory inside of it
$ mkdir -p workspace/images

# Copy the images into the workspace
$ cp /path/to/images/*.jpg workspace/images/

# Enter the workspace
$ cd workspace

# Initialize the workspace
$ sfm init .
Initialized workspace: .../workspace
Configuration file: .../workspace/.sfm-workspace.json
  feature_tool: colmap
  estimate_affine_shape: False
  domain_size_pooling: False
  max_num_features: None

# Solve SfM for the specified images using the global GLOMAP solver
$ sfm solve -g images/
Running global SfM with GLOMAP...
Image files:
  C:\Dev\sfmtool\workspace\images\seoul_bull_sculpture_%02d.jpg (17 files, sequence 1-17)
...
Saved reconstruction to: .../workspace/sfmr/20260404-00-solve-seoul_bull_sculpture_1-17.sfmr
```

Now you can use a variety of commands to inspect the reconstruction.

```
# Overview of the reconstruction
$ sfm inspect sfmr/20260404-00-solve-seoul_bull_sculpture_1-17.sfmr
...

# Per-image reprojection error metrics
$ sfm inspect --metrics sfmr/20260404-00-solve-seoul_bull_sculpture_1-17.sfmr
...
```

### The SfM Explorer GUI

Once you have an .sfmr file, you can load it into the SfM Explorer GUi to view it in 3D.

## What is Structure from Motion?

In SfM, you start with a scene that is static and take photographs
of the scene from multiple different poses. Importantly, the poses should be in different
positions around the scene, not looking around from one position. Starting from just
the photographs, it solves for the structure of the scene and the camera poses at the same
time. The structure describes 3D points on surfaces that are visible from more than one
camera, and the motion describes camera intrinsics like focal length and lens distortion and
camera extrinsics like the image's position and orientation.

Recent research is focused on ideas like using feedforward networks to go straight from images
into 3D representations. This project is not about VGGT or similar techniques, but visualizing their
output in SfMm tool or using them as part of SfM would be interesting to explore.

## Installation

Coming soon...
