//! PCU contract facade.
//!
//! The backing contract truth still comes from `fusion-pal` for now so provider crates can stay
//! wired the same way while `fusion-pcu` becomes the semantic front door.

pub use fusion_pal::contract::drivers::pcu::*;

pub mod caps {
    pub use fusion_pal::contract::drivers::pcu::caps::*;
}

pub mod device {
    pub use fusion_pal::contract::drivers::pcu::device::*;
}

pub mod error {
    pub use fusion_pal::contract::drivers::pcu::error::*;
}

pub mod invocation {
    pub use fusion_pal::contract::drivers::pcu::invocation::*;
}

pub mod ir {
    pub use fusion_pal::contract::drivers::pcu::ir::*;
}

pub mod protocol {
    pub use fusion_pal::contract::drivers::pcu::protocol::*;
}

pub mod unsupported {
    pub use fusion_pal::contract::drivers::pcu::unsupported::*;
}
