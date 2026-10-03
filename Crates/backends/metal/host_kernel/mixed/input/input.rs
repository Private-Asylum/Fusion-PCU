//! Stack-owned uploads and authentic retained resident input leases.
//!
//! A fresh uploaded buffer needs no additional Rust shared owner. Resident
//! storage retains its existing lease and an immutable borrow through execution.

#[rustfmt::skip]
use std::{
    cell::{
        Ref,
        RefCell,
    },
    rc::Rc,
};
use crate::MetalBuffer;

pub(super) enum Input {
    Uploaded(MetalBuffer),
    Resident(Rc<RefCell<MetalBuffer>>),
}

pub(super) enum View<'a> {
    Uploaded(&'a MetalBuffer),
    Resident(Ref<'a, MetalBuffer>),
}

impl Input {
    pub(super) fn borrow(&self) -> View<'_> {
        match self {
            Self::Uploaded(buffer) => View::Uploaded(buffer),
            Self::Resident(lease) => View::Resident(lease.borrow()),
        }
    }
}

impl View<'_> {
    pub(super) fn buffer(&self) -> &MetalBuffer {
        match self {
            Self::Uploaded(buffer) => buffer,
            Self::Resident(buffer) => buffer,
        }
    }
}
