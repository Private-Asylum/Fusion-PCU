# MLX-owned checked unary peers

The workload is actual generic `#[pcu]` Neg/ReLU source, with low F16/BF16/E4M3FN/E5M2 specializations, all three underflow policies, Reject/Clamp, and extents257/65537. Each registration verifies three changing finite banks and the untouched output tail before and after its measurement; filtered registrations initialize their own bank.

Four peers retain the same explicit MLX GPU stream and MLX-owned custom primitive: source-prepared, ordinary source, detached neutral graph, and a directly prepared native checker control. Every warm call copies current host input into the exact UInt8/UInt16 physical carrier, realizes fresh payload/status sibling arrays, completes the explicit MLX stream and both sibling waits, scans all status records, and copies the terminal logical output prefix. The most recent payload owner survives until its checked release before the next call or prepared-owner drop. The control uses the same fixed C extension without source or graph admission. No peer calls a MetalSession or a CPU arithmetic fallback.

These finite-bank timings do not measure recovery/fatal fault latency or scalar broadcast. A separate allocation-census build counts only warm caller-thread Rust allocation/reallocation/free requests; C++/MLX/driver/device allocations are outside those counters. Ordinary source may have existing facade adaptation allocations; this workload does not claim a zero-allocation native execution boundary.

Use `cargo bench -p fusion-pcu-mlx --bench checked_unary -- --test` for semantic smoke, and add `--features allocation-census` for the separate Rust census. Native hardware qualification is recorded independently in the external MLX plan; a successful non-Mac skip is not native evidence.


The six-format extension adds F32/F64 to the same four physical peers, for576 registrations across Neg/ReLU, three underflow policies, Reject/Clamp and257/65537 lanes. Ordinary source is a real fourth peer; a separate allocation-census build reports caller Rust allocations and excludes SDK/native/device allocations. Native source proof1743 is closed with all576 semantic peers and all576 zero caller Rust census rows; the independently emitted F32/F64 bit/status oracle1740 and prior low4 exhaustive proof are distinct. No timings or Portable membership follow from semantic smoke.
