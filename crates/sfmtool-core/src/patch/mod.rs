// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Patch clouds: point-patch storage and normal refinement.

pub mod cloud;
pub mod cluster_refine;
pub mod display_bitmaps;
pub mod keypoint_localize;
pub mod keypoint_subpixel;
pub mod member_coherence;
pub mod normal_refine;
pub mod self_similarity;
pub mod spawn;
pub mod view_selection;

#[cfg(test)]
mod sampler_rule_tests;

pub use cloud::{PatchCloud, PatchCloudError};

use std::sync::atomic::{AtomicUsize, Ordering};

use crate::progress::Progress;

/// How a batch over a patch cloud reports the patches it has finished: a bump
/// of the caller's `done` counter per patch, for a poller on another thread,
/// and a `patches` count to `progress` about every hundredth of the way
/// through.
pub(crate) struct PatchCounter<'a, 'p> {
    total: usize,
    step: usize,
    finished: AtomicUsize,
    done: Option<&'a AtomicUsize>,
    progress: &'a Progress<'p>,
}

impl<'a, 'p> PatchCounter<'a, 'p> {
    /// A counter for a batch of `total` patches.
    pub(crate) fn new(
        total: usize,
        done: Option<&'a AtomicUsize>,
        progress: &'a Progress<'p>,
    ) -> Self {
        Self {
            total,
            step: (total / 100).max(1),
            finished: AtomicUsize::new(0),
            done,
            progress,
        }
    }

    /// One more patch is finished.
    pub(crate) fn finished(&self) {
        if let Some(done) = self.done {
            done.fetch_add(1, Ordering::Relaxed);
        }
        let n = self.finished.fetch_add(1, Ordering::Relaxed) + 1;
        if n.is_multiple_of(self.step) || n == self.total {
            self.progress
                .count(n as u64, Some(self.total as u64), "patches");
        }
    }
}
