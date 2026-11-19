// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The region pool: the bytes of copied regions one kernel call may hold at
//! once, across all its threads.

use std::sync::atomic::{AtomicU64, Ordering};

use crate::camera::mipmap::PHOTOGRAPH_TILE_SIZE;

/// The bytes of copied regions a kernel call's pool holds per thread of the
/// rayon pool: four RGB photograph tiles' worth, 3.1 MB.
pub const REGION_POOL_BYTES_PER_THREAD: u64 =
    4 * PHOTOGRAPH_TILE_SIZE as u64 * PHOTOGRAPH_TILE_SIZE as u64 * 3;

/// The bytes of copied regions one kernel call may hold at once.
///
/// The caller of a kernel makes one pool for that kernel call and passes it
/// in; a kernel that calls another passes its own pool down. A render entry
/// point reserves its regions from it: a reservation is one atomic
/// compare-and-add that succeeds or fails at once and never waits, so nested
/// parallel loops on many threads cannot block on the pool. A render whose
/// reservation fails reads through photograph tiles instead, with the same
/// values. The bytes go back to the pool when the copied regions are dropped.
pub struct RegionPool {
    capacity: u64,
    held: AtomicU64,
    peak: AtomicU64,
    copied: AtomicU64,
    tiled: AtomicU64,
}

/// What a [`RegionPool`] has been asked for so far.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RegionPoolStats {
    /// The most bytes held at once.
    pub peak_bytes: u64,
    /// The reads that copied their regions from the pool.
    pub copied_reads: u64,
    /// The reads that took the tile path because their reservation failed.
    pub tiled_reads: u64,
}

impl RegionPool {
    /// A pool of [`REGION_POOL_BYTES_PER_THREAD`] for each thread of the
    /// current rayon pool.
    pub fn for_kernel_call() -> Self {
        Self::with_capacity(REGION_POOL_BYTES_PER_THREAD * rayon::current_num_threads() as u64)
    }

    /// A pool of `bytes`.
    pub fn with_capacity(bytes: u64) -> Self {
        Self {
            capacity: bytes,
            held: AtomicU64::new(0),
            peak: AtomicU64::new(0),
            copied: AtomicU64::new(0),
            tiled: AtomicU64::new(0),
        }
    }

    /// The bytes the pool holds at most.
    pub fn capacity(&self) -> u64 {
        self.capacity
    }

    /// The bytes held now.
    pub fn held_bytes(&self) -> u64 {
        self.held.load(Ordering::Relaxed)
    }

    /// What the pool has been asked for so far.
    pub fn stats(&self) -> RegionPoolStats {
        RegionPoolStats {
            peak_bytes: self.peak.load(Ordering::Relaxed),
            copied_reads: self.copied.load(Ordering::Relaxed),
            tiled_reads: self.tiled.load(Ordering::Relaxed),
        }
    }

    /// Reserve `bytes`, or `None` at once when the pool cannot hold them.
    pub(crate) fn try_reserve(&self, bytes: u64) -> Option<Reservation<'_>> {
        let mut held = self.held.load(Ordering::Relaxed);
        loop {
            let total = held.checked_add(bytes).filter(|&t| t <= self.capacity)?;
            // A failed exchange means another thread changed the count
            // between the load and the exchange; the loop retries with the
            // new count and never waits.
            match self
                .held
                .compare_exchange_weak(held, total, Ordering::AcqRel, Ordering::Relaxed)
            {
                Ok(_) => {
                    self.peak.fetch_max(total, Ordering::Relaxed);
                    return Some(Reservation { pool: self, bytes });
                }
                Err(now) => held = now,
            }
        }
    }

    /// Count a read that copied its regions.
    pub(crate) fn count_copied(&self) {
        self.copied.fetch_add(1, Ordering::Relaxed);
    }

    /// Count a read that took the tile path.
    pub(crate) fn count_tiled(&self) {
        self.tiled.fetch_add(1, Ordering::Relaxed);
    }
}

/// Bytes reserved from a [`RegionPool`], given back when dropped.
pub(crate) struct Reservation<'p> {
    pool: &'p RegionPool,
    bytes: u64,
}

impl Drop for Reservation<'_> {
    fn drop(&mut self) {
        self.pool.held.fetch_sub(self.bytes, Ordering::AcqRel);
    }
}
