// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! One view's tile as colour planes, and the self-similarity reading its
//! blurred copies are read with.

use super::blur::{blur_tile, BlurScratch};
use crate::patch::self_similarity::{zncc_self_similarity_radius, PatchTile, SelfSimilarityParams};

/// One view's tile as planar colour channels in `f32`, with the samples that
/// carry data.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TilePlanes {
    /// `channels` planes of `side × side`, row-major.
    pub values: Vec<f32>,
    /// Per sample, row-major: whether it carries data.
    pub data: Vec<bool>,
    /// The tile's side, `R`.
    pub side: usize,
    /// The colour channels, 1 to 3.
    pub channels: usize,
}

impl TilePlanes {
    /// The colour planes of an interleaved `side × side × stride` `u8` tile,
    /// with `data` flagging the samples that carry data. One or two channels
    /// are grey, and grey and alpha; three or four are RGB, and RGB and alpha.
    /// Alpha is not a colour plane: the caller reads it into `data` where it
    /// says which samples carry data.
    ///
    /// # Panics
    ///
    /// Panics if `samples` or `data` do not cover the tile, or `stride` is 0.
    pub fn from_interleaved(samples: &[u8], side: usize, stride: usize, data: &[bool]) -> Self {
        assert!(stride > 0, "TilePlanes: no channels");
        assert_eq!(
            samples.len(),
            side * side * stride,
            "TilePlanes: samples do not cover the tile"
        );
        assert_eq!(
            data.len(),
            side * side,
            "TilePlanes: one data flag per sample"
        );
        let channels = if stride <= 2 { 1 } else { 3 };
        let n = side * side;
        let mut values = vec![0.0f32; channels * n];
        for (k, pixel) in samples.chunks_exact(stride).enumerate() {
            for c in 0..channels {
                values[c * n + k] = f32::from(pixel[c]);
            }
        }
        Self {
            values,
            data: data.to_vec(),
            side,
            channels,
        }
    }

    /// The same tile blurred isotropically by `sigma` grid px ([`blur_tile`]).
    pub fn blurred(&self, sigma: f64, scratch: &mut BlurScratch) -> Self {
        let mut out = Self::default();
        self.blur_into(sigma, &mut out, scratch);
        out
    }

    /// [`Self::blurred`] into `out`, reusing its buffers.
    pub(super) fn blur_into(&self, sigma: f64, out: &mut Self, scratch: &mut BlurScratch) {
        out.values.resize(self.values.len(), 0.0);
        out.data.clone_from(&self.data);
        out.side = self.side;
        out.channels = self.channels;
        blur_tile(
            &self.values,
            self.channels,
            self.side,
            &self.data,
            sigma,
            &mut out.values,
            scratch,
        );
    }
}

/// The whole-tile self-similarity ellipse matrix of the planar tile `values`
/// (`channels` colour planes of `side × side`), read the overlap way over the
/// samples `data` marks, with the default parameters; `None` where the
/// reading has no finite axes. The bench and the bindings read each view's
/// tile with it, and assess its blur ([`assess_blur`](super::assess_blur))
/// with it, so the lengths compared are of one reading.
pub fn read_tile_ellipse(
    values: &[f32],
    channels: usize,
    side: usize,
    data: &[bool],
) -> Option<[[f64; 2]; 2]> {
    let reading = zncc_self_similarity_radius(
        &PatchTile {
            values,
            channels: channels.min(3),
            width: side,
            height: side,
        },
        Some(data),
        [0, 0, side, side],
        &SelfSimilarityParams::default(),
    );
    let e = reading.ellipse;
    e.axes.iter().all(|a| a.is_finite()).then_some(e.matrix)
}
