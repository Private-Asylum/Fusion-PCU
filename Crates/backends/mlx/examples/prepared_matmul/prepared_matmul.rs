//! Authentic annotated-source capture to exact prepared MLX GPU replay.

#[path = "source.rs"]
mod source;
#[path = "support/support.rs"]
mod support;

fn main() {
    support::run();
}
