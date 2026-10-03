# Checked integer quotient/remainder host pairs

`cargo bench -p fusion-pcu-cpu --features source-div-rem --bench checked_div_rem`
measures eight primitive widths (`i/u8/16/32/64`) at one and 4096 logical invocations.
Each workload has four peers: an actual annotated prepared call, its ordinary source call,
an explicitly prepared IR diagnostic, and an independent Rust checked arithmetic control.
`-- --test` performs semantic smoke without statistical timing samples.

The native control uses primitive `checked_div` and `checked_rem`, without PCU preparation,
execution, discovery, or arithmetic traits. Its contract matches complete map preflight and
two-output publication. All peers retain caller-owned quotient and remainder buffers with
three preserved tail elements. Every invocation changes the first dividend, supplies both
fresh input slices, and observes both outputs. Zero and signed MIN/-1 reject without partial
publication; the latter also rejects the remainder operation, as PCU explicitly requires.

Discovery, ranking, IR construction, allocation and preparation occur outside timing.
Before each group, full output equality is checked. A 256-call warm census asserts exactly
zero allocations, reallocations and frees for each route; the ordinary source scoring count
must remain unchanged through the census and measured group. These census assertions do
not themselves establish a latency or speedup. Cold unsupported policy/type combinations,
all eight source routes, faults, complete schemas and retry are tested separately.
