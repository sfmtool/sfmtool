// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Each view's blur growth: its tile blurred by each of
//! [`GROWTH_PROBE_SIGMAS`] and each blurred tile's self-similarity ellipse
//! read, kept for every pair the view is blurred in.

use super::blur::{blur_tile, BlurScratch};
use super::tiles::TilePlanes;
use super::{BlurCovariance, BlurGrowth, GROWTH_PROBE_SIGMAS};

/// The growth of `tile`'s ellipse: the tile blurred isotropically by each of
/// [`GROWTH_PROBE_SIGMAS`] in turn into `out`, each blurred tile read by
/// `read`, the readings kept beside `ellipse`, the unblurred tile's
/// ([`BlurGrowth::from_ellipses`]). `None` where `read` gives no ellipse for
/// any of them.
///
/// `read` must read the ellipse the way `ellipse` was read, the same reading
/// of the same render, so that the growth is the reading's own and the
/// lengths a pair compares are of one reading.
///
/// # Panics
///
/// Panics if `tile`'s planes or data flags do not cover it.
pub fn read_growth(
    tile: &TilePlanes,
    ellipse: &[[f64; 2]; 2],
    mut read: impl FnMut(&[f32]) -> Option<[[f64; 2]; 2]>,
    out: &mut Vec<f32>,
    scratch: &mut BlurScratch,
) -> Option<BlurGrowth> {
    out.resize(tile.values.len(), 0.0);
    let mut probed = [[[0.0; 2]; 2]; GROWTH_PROBE_SIGMAS.len()];
    for (e, &sigma) in probed.iter_mut().zip(&GROWTH_PROBE_SIGMAS) {
        blur_tile(
            &tile.values,
            tile.channels,
            tile.side,
            &tile.data,
            BlurCovariance::isotropic(sigma),
            out,
            scratch,
        );
        *e = read(out)?;
    }
    Some(BlurGrowth::from_ellipses(ellipse, &probed))
}

/// The growth of each view of a track, read the first time a pair blurs the
/// view ([`read_growth`]) and kept for every other pair it is blurred in.
#[derive(Debug, Clone, Default)]
pub struct ViewGrowths {
    /// Per view: `None` until read, then the reading.
    read: Vec<Option<Option<BlurGrowth>>>,
    reads: usize,
    probe: Vec<f32>,
    scratch: BlurScratch,
}

impl ViewGrowths {
    /// Room for `views` views, none read.
    pub fn new(views: usize) -> Self {
        Self {
            read: vec![None; views],
            ..Self::default()
        }
    }

    /// View `view`'s growth: [`read_growth`] of `tile` against `ellipse` with
    /// `read` the first time it is asked for, the same answer after that.
    ///
    /// # Panics
    ///
    /// Panics if `view` is not below the count given to [`Self::new`].
    pub fn get(
        &mut self,
        view: usize,
        tile: &TilePlanes,
        ellipse: &[[f64; 2]; 2],
        read: impl FnMut(&[f32]) -> Option<[[f64; 2]; 2]>,
    ) -> Option<BlurGrowth> {
        if let Some(growth) = self.read[view] {
            return growth;
        }
        let growth = read_growth(tile, ellipse, read, &mut self.probe, &mut self.scratch);
        self.reads += GROWTH_PROBE_SIGMAS.len();
        self.read[view] = Some(growth);
        growth
    }

    /// How many blurred tiles have been read: one per probe for each view
    /// whose growth has been read.
    pub fn reads(&self) -> usize {
        self.reads
    }
}
