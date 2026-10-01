#[path = "acquisition/acquisition.rs"]
mod acquisition;
#[path = "bootstrap/bootstrap.rs"]
mod bootstrap;
#[path = "output/output.rs"]
mod output;
#[path = "platform/platform.rs"]
mod platform;
#[path = "preparation/preparation.rs"]
mod prepare;
fn main() {
    if let Err(error) = bootstrap::build() {
        panic!("MLX native preparation refused: {error}");
    }
}
