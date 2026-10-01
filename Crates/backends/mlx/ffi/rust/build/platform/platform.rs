//! Decide from Cargo's target and host before executing any native tool.
pub fn native_build(host: &str, target: &str) -> Result<bool, String> {
    if target != "aarch64-apple-darwin" {
        return Ok(false);
    }
    if host != target {
        return Err(format!(
            "MLX native build requires Apple silicon host; unsupported cross compilation {host} -> {target}"
        ));
    }
    Ok(true)
}
