//! Logical invocation context and grid-stride indexing.

use core::num::NonZeroU32;

/// Logical invocation context surfaced to one dispatch-style kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PcuDispatchContext {
    global_invocation_id: u32,
    invocation_count: NonZeroU32,
}

impl PcuDispatchContext {
    /// Creates one logical invocation context, rejecting padded physical lanes.
    #[must_use]
    pub const fn new(global_invocation_id: u32, invocation_count: NonZeroU32) -> Option<Self> {
        if global_invocation_id >= invocation_count.get() {
            return None;
        }
        Some(Self {
            global_invocation_id,
            invocation_count,
        })
    }

    /// Returns this invocation's global logical identifier.
    #[must_use]
    pub const fn global_invocation_id(self) -> u32 {
        self.global_invocation_id
    }

    /// Returns the requested logical invocation count, independent of physical launch padding.
    #[must_use]
    pub const fn invocation_count(self) -> NonZeroU32 {
        self.invocation_count
    }

    /// Returns the indices this logical invocation covers in a grid-stride loop.
    ///
    /// The stride is the logical invocation count, even if a backend launches more physical
    /// lanes to fill its final workgroup. Callers may use this as a serial CPU reference for the
    /// portable one-dimensional grid-stride contract.
    #[must_use]
    pub const fn grid_stride_indices(self, extent: u64) -> PcuGridStrideIndices {
        let first = self.global_invocation_id as u64;
        PcuGridStrideIndices {
            next: if first < extent { Some(first) } else { None },
            stride: self.invocation_count,
            extent,
        }
    }
}

/// Overflow-safe serial reference for one logical invocation's grid-stride indices.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PcuGridStrideIndices {
    pub(super) next: Option<u64>,
    pub(super) stride: NonZeroU32,
    pub(super) extent: u64,
}

impl Iterator for PcuGridStrideIndices {
    type Item = u64;

    fn next(&mut self) -> Option<Self::Item> {
        let current = self.next?;
        self.next = current
            .checked_add(u64::from(self.stride.get()))
            .filter(|next| *next < self.extent);
        Some(current)
    }
}
