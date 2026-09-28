//! MLX backend scaffold for Fusion PCU on Apple silicon.
//!
//! This crate reserves the workspace integration seam for a future MLX implementation.
//! It currently exposes no executable backend, discovery provider, or capabilities and does
//! not load MLX or require Apple platform libraries. Implementation remains deferred while
//! the common substrate is developed and validated against `ROCm`.

extern crate fusion_pcu_core as fusion_pcu;
