// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! A synthetic capture for the walk's tests: a scene of points with random
//! descriptors, seen by a pinhole camera along a path with known motion per
//! frame, and a [`FrameSource`] that writes each frame's version 2 `.sift`
//! files on a pool of workers, as the extracting source does, only for the
//! frames it is asked to prefetch or wait for, and records what it was asked.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use ndarray::{Array2, Array3};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use rand_distr::StandardNormal;
use sfmtool_sift_format::{
    write_sift, FeatureToolMetadata, SiftContentHash, SiftData, SiftMetadata, DESCRIPTOR_DIM,
    THUMBNAIL_SIZE,
};

use super::super::source::{Job, Threads, WorkerPool};
use super::super::{
    CaptureFrame, CaptureImage, CaptureSequence, FrameId, FrameSource, FrameSourceError,
};
use crate::features::sift::BuildSiftFilesError;
use crate::progress::Progress;

pub(super) const WIDTH: u32 = 640;
pub(super) const HEIGHT: u32 = 480;
const FOCAL: f64 = 500.0;
/// The size of a rig's second sensor, which differs from the reference's.
const SECOND_WIDTH: u32 = 320;
const SECOND_HEIGHT: u32 = 240;

/// The camera's x motion per frame for a step of about 1% of the long side:
/// the median displacement is `FOCAL · dx / z` at a median depth near 10.
pub(super) const ONE_PERCENT: f64 = 0.128;

/// What a frame's reference image shows.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum Content {
    /// Every scene point in view.
    Scene,
    /// Random features that match nothing.
    Junk,
    /// The points in view for which `(frame + point) % period < lifetime`:
    /// neighbouring frames share most, and frames `lifetime` or more apart
    /// share none.
    Window { period: usize, lifetime: usize },
    /// Only every `every`-th point.
    Subset { every: usize },
}

/// One frame of the synthetic capture.
#[derive(Clone, Debug)]
pub(super) struct Shot {
    pub(super) x: f64,
    pub(super) content: Content,
    /// The readings of each sensor: sharpness and `[Exx, Eyy, Exy]`.
    pub(super) readings: Vec<(f32, [f32; 3])>,
}

/// A synthetic capture: the scene, the frames, and how many sensors.
#[derive(Clone)]
pub(super) struct Synth {
    points: Vec<[f64; 3]>,
    descriptors: Vec<[u8; DESCRIPTOR_DIM]>,
    pub(super) shots: Vec<Shot>,
    pub(super) sensors: usize,
}

/// The readings of a sharp frame `t` of sensor `k`: a sharpness near 1 for
/// the reference sensor and near 5 for the second, and a tensor near
/// `[1, 1, 0]`, each with a few percent of variation, as a real capture's
/// readings vary.
pub(super) fn sharp(t: usize, k: usize) -> (f32, [f32; 3]) {
    let base = if k == 0 { 1.0 } else { 5.0 };
    let wobble = ((t * 37 + k * 11) % 9) as f32 / 100.0;
    let tilt = (((t * 53 + k * 7) % 9) as f32 - 4.0) / 100.0;
    (base * (0.96 + wobble), [1.0 + tilt, 1.0 - tilt, 0.0])
}

impl Synth {
    /// A wall of points spread along x from `x_min` to `x_max`, at depths
    /// from 6 to 14 in front of a camera at the origin looking along +z.
    pub(super) fn wall(seed: u64, x_min: f64, x_max: f64, density: f64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let n = ((x_max - x_min) * density) as usize;
        let points = (0..n)
            .map(|_| {
                [
                    rng.random_range(x_min..x_max),
                    rng.random_range(-4.0..4.0),
                    rng.random_range(6.0..14.0),
                ]
            })
            .collect();
        Self::with_points(&mut rng, points)
    }

    fn with_points(rng: &mut StdRng, points: Vec<[f64; 3]>) -> Self {
        let descriptors = points
            .iter()
            .map(|_| std::array::from_fn(|_| rng.random::<u8>()))
            .collect();
        Self {
            points,
            descriptors,
            shots: Vec::new(),
            sensors: 1,
        }
    }

    /// Frames along x at the positions `xs`, all sharp and showing the scene.
    pub(super) fn along(mut self, xs: impl IntoIterator<Item = f64>) -> Self {
        self.shots = xs
            .into_iter()
            .enumerate()
            .map(|(t, x)| Shot {
                x,
                content: Content::Scene,
                readings: vec![sharp(t, 0), sharp(t, 1)],
            })
            .collect();
        self
    }

    /// A camera that moves `speed` along x every frame for `frames` frames.
    pub(super) fn steady(seed: u64, frames: usize, speed: f64) -> Self {
        Self::steady_with_density(seed, frames, speed, 80.0)
    }

    /// [`Synth::steady`] with `density` points per unit of x, about 12
    /// times that many in view of each frame.
    pub(super) fn steady_with_density(seed: u64, frames: usize, speed: f64, density: f64) -> Self {
        let length = speed * frames as f64;
        Self::wall(seed, -12.0, length + 12.0, density).along((0..frames).map(|t| speed * t as f64))
    }

    /// One sequence of this capture's frames, with its `.sift` files in
    /// `dir`.
    pub(super) fn sequence(&self, dir: &Path) -> CaptureSequence {
        let frames = (0..self.shots.len())
            .map(|t| CaptureFrame {
                number: t as i64 + 1,
                images: (0..self.sensors)
                    .map(|k| CaptureImage {
                        image_path: dir.join(format!("cam{k}_{:04}.jpg", t + 1)),
                        sift_path: dir.join(format!("cam{k}_{:04}.jpg.sift", t + 1)),
                    })
                    .collect(),
            })
            .collect();
        CaptureSequence {
            patterns: (0..self.sensors)
                .map(|k| format!("cam{k}_%04d.jpg"))
                .collect(),
            frames,
        }
    }

    /// The features of sensor `k`'s image of frame `t`.
    fn features(&self, t: usize, k: usize) -> (Vec<[f32; 2]>, Vec<u8>, u32, u32) {
        let mut rng = StdRng::seed_from_u64((t * 7919 + k * 104_729 + 1) as u64);
        let shot = &self.shots[t];
        let content = if k == 0 { shot.content } else { Content::Junk };
        if content == Content::Junk {
            let (w, h) = if k == 0 {
                (WIDTH, HEIGHT)
            } else {
                (SECOND_WIDTH, SECOND_HEIGHT)
            };
            let n = 300;
            let positions = (0..n)
                .map(|_| {
                    [
                        rng.random_range(1.0..w as f32 - 1.0),
                        rng.random_range(1.0..h as f32 - 1.0),
                    ]
                })
                .collect();
            let descriptors = (0..n * DESCRIPTOR_DIM)
                .map(|_| rng.random::<u8>())
                .collect();
            return (positions, descriptors, w, h);
        }
        let mut positions = Vec::new();
        let mut descriptors = Vec::new();
        for (p, point) in self.points.iter().enumerate() {
            let shown = match content {
                Content::Window { period, lifetime } => (t + p) % period < lifetime,
                Content::Subset { every } => p % every == 0,
                _ => true,
            };
            if !shown {
                continue;
            }
            let (x, y, z) = (point[0] - shot.x, point[1], point[2]);
            let noise = |rng: &mut StdRng| 0.2 * rng.sample::<f64, _>(StandardNormal);
            let u = FOCAL * x / z + f64::from(WIDTH) / 2.0 + noise(&mut rng);
            let v = FOCAL * y / z + f64::from(HEIGHT) / 2.0 + noise(&mut rng);
            if u > 1.0 && u < f64::from(WIDTH) - 1.0 && v > 1.0 && v < f64::from(HEIGHT) - 1.0 {
                positions.push([u as f32, v as f32]);
                descriptors.extend_from_slice(&self.descriptors[p]);
            }
        }
        (positions, descriptors, WIDTH, HEIGHT)
    }

    /// Write sensor `k`'s `.sift` file of frame `t` at `path`, of `version`.
    pub(super) fn write(&self, t: usize, k: usize, path: &Path, version: u32) {
        let (positions, descriptors, width, height) = self.features(t, k);
        write_sift_file(
            path,
            &positions,
            descriptors,
            width,
            height,
            self.shots[t].readings[k],
            version,
        );
    }
}

/// Write a `.sift` file of the given features and readings.
pub(super) fn write_sift_file(
    path: &Path,
    positions: &[[f32; 2]],
    descriptors: Vec<u8>,
    width: u32,
    height: u32,
    (sharpness, tensor): (f32, [f32; 3]),
    version: u32,
) {
    let n = positions.len();
    let v2 = version >= 2;
    let feature_options = if v2 {
        serde_json::json!({"image_to_gray": {"formula": "G"}})
    } else {
        serde_json::json!({})
    };
    let data = SiftData {
        feature_tool_metadata: FeatureToolMetadata {
            feature_tool: "sfmtool".into(),
            feature_type: "sift".into(),
            feature_options,
        },
        metadata: SiftMetadata {
            version,
            image_name: path
                .file_stem()
                .unwrap_or_default()
                .to_string_lossy()
                .into_owned(),
            image_file_xxh128: "0".repeat(32),
            image_file_size: 1,
            image_width: width,
            image_height: height,
            feature_count: n as u32,
            sharpness: v2.then_some(f64::from(sharpness)),
            gradient_structure_tensor_xx_yy_xy: v2.then(|| tensor.map(f64::from)),
        },
        content_hash: SiftContentHash::default(),
        positions_xy: Array2::from_shape_vec((n, 2), positions.iter().flatten().copied().collect())
            .expect("n x 2"),
        affine_shapes: Array3::from_shape_fn((n, 2, 2), |(_, i, j)| if i == j { 1.0 } else { 0.0 }),
        descriptors: Array2::from_shape_vec((n, DESCRIPTOR_DIM), descriptors).expect("n x 128"),
        thumbnail_y_x_rgb: Array3::zeros((THUMBNAIL_SIZE, THUMBNAIL_SIZE, 3)),
    };
    write_sift(path, &data, 1).expect("the test's .sift file writes");
}

/// How the test source misbehaves, if at all.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum Fault {
    None,
    /// The worker fails on this frame of the first sequence, as an
    /// extraction whose write fails.
    FailFrame(usize),
    /// Write nothing.
    WriteNothing,
    /// Write version 1 files.
    WriteVersion1,
    /// Write this frame of the first sequence at another image size.
    Resize(usize),
    /// Set the cancel flag on the wait with this index, from 0, before
    /// waiting.
    CancelOnWait(usize),
}

/// What the test source was asked.
#[derive(Default)]
pub(super) struct Asked {
    /// The frames of each wait, in order.
    pub(super) waits: Vec<Vec<FrameId>>,
    /// Every frame asked to be prefetched.
    pub(super) prefetched: BTreeSet<FrameId>,
}

/// A [`FrameSource`] over [`Synth`]s whose workers write the `.sift` files of
/// the frames asked for, after a short wait that differs per frame so that
/// several workers finish out of order.
pub(super) struct SynthSource<'s> {
    pool: WorkerPool,
    threads: usize,
    asked: Asked,
    fault: Fault,
    cancel: Option<&'s AtomicBool>,
}

impl<'s> SynthSource<'s> {
    pub(super) fn new(synths: &[&Synth], threads: usize, fault: Fault) -> Self {
        let synths: Arc<Vec<Synth>> = Arc::new(synths.iter().map(|&s| s.clone()).collect());
        let job: Arc<Job> = Arc::new(move |id, images, _progress, wrote| {
            write_frame(&synths, id, images, fault, threads, wrote)
        });
        Self {
            pool: WorkerPool::new(job, Threads::Fixed(threads)),
            threads,
            asked: Asked::default(),
            fault,
            cancel: None,
        }
    }

    /// Set `flag` at the wait [`Fault::CancelOnWait`] names.
    pub(super) fn cancelled_by(mut self, flag: &'s AtomicBool) -> Self {
        self.cancel = Some(flag);
        self
    }

    pub(super) fn into_asked(self) -> Asked {
        self.asked
    }
}

/// Write the `.sift` files of one frame, as `fault` says.
fn write_frame(
    synths: &[Synth],
    id: FrameId,
    images: &[CaptureImage],
    fault: Fault,
    threads: usize,
    wrote: &mut bool,
) -> Result<(), FrameSourceError> {
    if threads > 1 {
        std::thread::sleep(Duration::from_micros(((id.frame * 7919) % 2000) as u64));
    }
    let first = id.sequence == 0;
    if first && fault == Fault::FailFrame(id.frame) {
        return Err(FrameSourceError::Extract {
            frame: id,
            error: BuildSiftFilesError::Write {
                path: images[0].sift_path.clone(),
                problem: "the disk is full".to_string(),
            },
        });
    }
    if fault == Fault::WriteNothing {
        return Ok(());
    }
    let version = if fault == Fault::WriteVersion1 { 1 } else { 2 };
    let synth = &synths[id.sequence];
    for (k, image) in images.iter().enumerate() {
        if let Some(parent) = image.sift_path.parent() {
            std::fs::create_dir_all(parent).expect("the test folder is made");
        }
        if first && k == 0 && fault == Fault::Resize(id.frame) {
            let readings = synth.shots[id.frame].readings[0];
            write_sift_file(
                &image.sift_path,
                &[[1.0, 1.0]],
                vec![0; 128],
                320,
                240,
                readings,
                2,
            );
        } else {
            synth.write(id.frame, k, &image.sift_path, version);
        }
        *wrote = true;
    }
    Ok(())
}

impl FrameSource for SynthSource<'_> {
    fn concurrency(&self) -> usize {
        self.threads
    }

    fn prefetch(&mut self, frames: &[(FrameId, &CaptureFrame)]) {
        self.asked
            .prefetched
            .extend(frames.iter().map(|&(id, _)| id));
        self.pool.prefetch(frames);
    }

    fn wait(
        &mut self,
        frames: &[(FrameId, &CaptureFrame)],
        progress: &Progress,
    ) -> Result<(), FrameSourceError> {
        let call = self.asked.waits.len();
        self.asked
            .waits
            .push(frames.iter().map(|&(id, _)| id).collect());
        if self.fault == Fault::CancelOnWait(call) {
            if let Some(flag) = self.cancel {
                flag.store(true, Ordering::Relaxed);
            }
        }
        self.pool.wait(frames, progress)
    }

    fn finish(&mut self) -> Vec<FrameId> {
        self.pool.finish()
    }
}

/// The `.sift` files under `dir`, sorted.
pub(super) fn sift_files(dir: &Path) -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .map(|entries| {
            entries
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.extension().is_some_and(|e| e == "sift"))
                .collect()
        })
        .unwrap_or_default();
    files.sort();
    files
}
