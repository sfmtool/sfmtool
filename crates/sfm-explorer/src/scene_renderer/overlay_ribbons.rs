// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The overlay ribbon store: one GPU buffer holding the instances of every
//! overlay figure the 3D view draws, a region per producer.
//!
//! A producer is the code that builds one overlay's instances: the bench
//! figure, and the drawn measurements. Each writes its instances into its own
//! region with [`OverlayRibbons::write`], and each target's pass draws its
//! regions out of the one buffer in the fixed region order
//! ([`ProducerId`]'s). `specs/gui/viewer-3d-bench-layer.md` § "The overlay
//! ribbon store" describes it.
//!
//! The regions lie end to end in [`ProducerId`] order, each `capacity`
//! instances long. A write that fits its region's capacity is one
//! `queue.write_buffer` of that region and nothing else. One that does not
//! doubles the region's capacity until it fits and lays the regions out
//! again; every region that moved is written again at its new place, from the
//! copy of its instances the store keeps for that purpose. When the regions
//! no longer fit the buffer, the buffer is replaced by one twice the size (or
//! more, by doubling), and every region is written into it. Neither the
//! buffer nor a region ever shrinks within a session, so a producer that
//! draws nothing for a while keeps its room.

use std::ops::Range;

use super::gpu_types::OverlayInstance;

/// The bytes of one instance in the buffer.
const STRIDE: u64 = std::mem::size_of::<OverlayInstance>() as u64;

/// The fewest instances a region is given when it first holds any.
const MIN_REGION_CAPACITY: u32 = 64;

/// The fewest instances the buffer is made for.
const MIN_BUFFER_CAPACITY: u32 = 256;

/// The code that writes one region of the store.
///
/// The variants are in the fixed order the regions lie in and are drawn in:
/// on the scene target the bench figure; on the measurements' layer the
/// measurements.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum ProducerId {
    /// The focused bench item's figure.
    BenchFigure,
    /// The selected and shown measurements.
    Measurements,
}

/// The texture a region's instances are drawn into.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum OverlayTarget {
    /// The scene texture, the EDL output, blended in linear light (the bench
    /// figure).
    Scene,
    /// The measurements' layer, blended on encoded values and painted over
    /// the scene by egui (`specs/gui/measurement-view.md` § "Drawing
    /// measurements").
    MeasurementLayer,
}

/// One producer's part of the buffer.
#[derive(Debug)]
struct Region {
    producer: ProducerId,
    target: OverlayTarget,
    /// Where its instances start in the buffer, in instances.
    offset: u32,
    /// How many instances it draws: as many as its last write handed it.
    count: u32,
    /// How many instances it has room for.
    capacity: u32,
    /// The instances of its last write, kept so that a repack can write the
    /// region again at its new place.
    instances: Vec<OverlayInstance>,
}

/// One `queue.write_buffer` the store issued, in bytes of the buffer.
#[cfg(test)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct IssuedWrite {
    pub(super) producer: ProducerId,
    pub(super) bytes: Range<u64>,
}

/// One buffer holding every overlay figure's instances, a region per
/// producer.
#[derive(Default)]
pub(crate) struct OverlayRibbons {
    /// Made on the first write that holds an instance, then replaced only by
    /// a larger one.
    buffer: Option<wgpu::Buffer>,
    /// How many instances `buffer` has room for.
    buffer_capacity: u32,
    /// In [`ProducerId`] order, which is their order in the buffer and the
    /// order they are drawn in.
    regions: Vec<Region>,
    /// Every write the store issued, oldest first, for the tests to read.
    #[cfg(test)]
    pub(super) issued: Vec<IssuedWrite>,
    /// The buffer's bytes as the issued writes left them: zeros where
    /// nothing has been written since the buffer was made.
    #[cfg(test)]
    mirror: Vec<u8>,
}

impl OverlayRibbons {
    /// Put `instances` in `producer`'s region, to be drawn into `target`.
    ///
    /// Instances within the region's capacity are one write of that region.
    /// More than that grow the region by doubling, lay the regions out again
    /// and write every region that moved. An empty slice writes nothing: the
    /// region draws nothing and keeps its capacity.
    pub(crate) fn write(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        producer: ProducerId,
        target: OverlayTarget,
        instances: &[OverlayInstance],
    ) {
        let index = self.region_index(producer, target);
        let region = &mut self.regions[index];
        region.target = target;
        region.instances.clear();
        region.instances.extend_from_slice(instances);
        region.count = instances.len() as u32;
        if instances.is_empty() {
            return;
        }
        if region.count <= region.capacity {
            self.write_region(queue, index);
            return;
        }

        let mut capacity = region.capacity.max(MIN_REGION_CAPACITY);
        while capacity < region.count {
            capacity *= 2;
        }
        region.capacity = capacity;
        let before: Vec<u32> = self.regions.iter().map(|r| r.offset).collect();
        let mut offset = 0;
        for region in &mut self.regions {
            region.offset = offset;
            offset += region.capacity;
        }

        if self.buffer.is_none() || offset > self.buffer_capacity {
            let mut buffer_capacity = self.buffer_capacity.max(MIN_BUFFER_CAPACITY);
            while buffer_capacity < offset {
                buffer_capacity *= 2;
            }
            self.buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("overlay ribbons"),
                size: u64::from(buffer_capacity) * STRIDE,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            self.buffer_capacity = buffer_capacity;
            #[cfg(test)]
            {
                self.mirror = vec![0; (u64::from(buffer_capacity) * STRIDE) as usize];
            }
            // A new buffer holds nothing, so every region is written into it.
            for i in 0..self.regions.len() {
                self.write_region(queue, i);
            }
        } else {
            for (i, offset_before) in before.into_iter().enumerate() {
                if i == index || self.regions[i].offset != offset_before {
                    self.write_region(queue, i);
                }
            }
        }
    }

    /// The index of `producer`'s region, made empty at its place in the
    /// fixed order if it has none yet.
    fn region_index(&mut self, producer: ProducerId, target: OverlayTarget) -> usize {
        match self
            .regions
            .binary_search_by(|region| region.producer.cmp(&producer))
        {
            Ok(index) => index,
            Err(index) => {
                // With no room yet, it starts where the region before it ends,
                // and moves nothing after it.
                let offset = index
                    .checked_sub(1)
                    .map_or(0, |i| self.regions[i].offset + self.regions[i].capacity);
                self.regions.insert(
                    index,
                    Region {
                        producer,
                        target,
                        offset,
                        count: 0,
                        capacity: 0,
                        instances: Vec::new(),
                    },
                );
                index
            }
        }
    }

    /// Write region `index`'s instances at its place in the buffer.
    fn write_region(&mut self, queue: &wgpu::Queue, index: usize) {
        let region = &self.regions[index];
        let Some(buffer) = &self.buffer else {
            return;
        };
        if region.instances.is_empty() {
            return;
        }
        let start = u64::from(region.offset) * STRIDE;
        let bytes: &[u8] = bytemuck::cast_slice(&region.instances);
        queue.write_buffer(buffer, start, bytes);
        #[cfg(test)]
        {
            let end = start + bytes.len() as u64;
            self.mirror[start as usize..end as usize].copy_from_slice(bytes);
            self.issued.push(IssuedWrite {
                producer: region.producer,
                bytes: start..end,
            });
        }
    }

    /// The instance ranges `target`'s pass draws, in the fixed region order,
    /// with the producer each is of. A region that holds nothing is left out.
    pub(crate) fn draws(
        &self,
        target: OverlayTarget,
    ) -> impl Iterator<Item = (ProducerId, Range<u32>)> + '_ {
        self.regions
            .iter()
            .filter(move |region| region.target == target && region.count > 0)
            .map(|region| (region.producer, region.offset..region.offset + region.count))
    }

    /// How many instances `target`'s pass draws.
    pub(crate) fn drawn_count(&self, target: OverlayTarget) -> u32 {
        self.draws(target)
            .map(|(_, range)| range.len() as u32)
            .sum()
    }

    /// Bind the buffer to vertex slot 1 of `pass` and draw each of
    /// `target`'s regions as camera-facing ribbon quads, in region order.
    ///
    /// The caller has set the pipeline, the bind group and the quad corners in
    /// slot 0.
    pub(crate) fn draw(&self, pass: &mut wgpu::RenderPass<'_>, target: OverlayTarget) {
        let Some(buffer) = &self.buffer else {
            return;
        };
        pass.set_vertex_buffer(1, buffer.slice(..));
        for (_, instances) in self.draws(target) {
            pass.draw(0..4, instances);
        }
    }

    /// The buffer every region lies in, once a write has made it.
    #[cfg(test)]
    pub(super) fn buffer(&self) -> Option<&wgpu::Buffer> {
        self.buffer.as_ref()
    }

    /// How many instances the buffer has room for.
    #[cfg(test)]
    pub(super) fn buffer_capacity(&self) -> u32 {
        self.buffer_capacity
    }

    /// `producer`'s region as `(offset, count, capacity)`, in instances.
    #[cfg(test)]
    pub(super) fn region(&self, producer: ProducerId) -> Option<(u32, u32, u32)> {
        self.regions
            .iter()
            .find(|region| region.producer == producer)
            .map(|region| (region.offset, region.count, region.capacity))
    }

    /// What the buffer holds, as the writes the store issued left it.
    ///
    /// Kept beside the buffer because the `noop` device the tests run on keeps
    /// no buffer contents to read back.
    #[cfg(test)]
    pub(super) fn contents(&self) -> &[u8] {
        &self.mirror
    }
}

#[cfg(test)]
mod tests;
