//! CUDA backend scaffold for Fusion PCU.
//!
//! This crate reserves the workspace integration seam for a future CUDA implementation.
//! It currently exposes no executable backend, discovery provider, or capabilities and does
//! not load a CUDA runtime or require an installed SDK. Implementation remains deferred while
//! the common substrate is developed and validated against `ROCm`.

extern crate fusion_pcu_core as fusion_pcu;
