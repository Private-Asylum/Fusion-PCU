# Checked CPU tensor peers

`checked_tensor` runs genuine ordinary `#[pcu(flag(strict))]` source beside a frozen
explicit-graph diagnostic and independently authored direct checked host arithmetic.
Each route creates a fresh owned result, copies it into the same caller output buffer,
and drops the result inside the measured boundary. Immutable shape ownership is cached
as `Rc<[usize]>` outside sampling; each route clones that reference and allocates one fresh
escaping data vector. Two changing input banks alternate.
Complete result bits are verified before and after sampling. Preparation and source cache
warmup occur outside steady-state sampling.

The loss peers cover F32/F64 at 16 and 4096 elements. The training peers cover a two-feature,
two-sample MatMul → ReLU → checked MSE → subtraction → ReLU backward → transpose MatMul
→ SGD update. The disconnected loss remains an observable checked effect. Every source
workload is an annotated function; the explicit graph is a separate diagnostic.

The direct peer independently authors operation order and storage handling, using the same
core checked scalar contract required by all three routes. It preserves separate nearest-even
arithmetic steps and first-fault order, without hardware unchecked arithmetic or contraction.
This is an exact-contract CPU peer, not a vendor BLAS throughput comparison.

Use `cargo bench -p fusion-pcu-cpu --features source-tensor --bench checked_tensor -- --test`
for an 18-case semantic smoke gate. No timing or speedup claim is recorded by that gate.
The executor's own warm allocation census is in `prepared_tensor`; owned result creation
here intentionally allocates escaping storage, and the source facade owns its cache metadata.
The source integration test measures exactly one allocation for warm F32/F64 loss and training,
including owned-output readback and drop. It does not claim allocation-free escaping ownership.

Native compound permission, backend precision permission, and portable reproduction remain
independent and explicitly unsupported by this checked plan. Default Boundary compounds
reject; strictness does not silently authorize native arithmetic.
