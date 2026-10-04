# CPU wide composition benchmark

Run the bounded smoke and allocation census with:

```sh
cargo bench -p fusion-pcu-cpu --features source-composed --bench wide_composed -- --test
```

The four I/U256/512 carriers run at 64 and 4096 elements. Genuine generic
`#[pcu]` source computes `stage = input + seed`, reloads the stage, multiplies
by the original input, then subtracts that original value into the output.
The seed is broadcast. Direct and grid source each have ordinary, explicitly
prepared, and detached graph preparation routes with Reject and Clamp policy.
Separate `flag(deterministic)` entries pair the same source and graph workload
under PortableV1. Normal entries remain independently identified; enabling
determinism does not silently enable Strict checking or suppress Clamp notices.
Healthy inputs change on every call; edge inputs put the maximum carrier value
in the final logical lane. Reject must preserve both complete output buffers;
Clamp must return its recovered fault and publish the useful intermediate and
final output. Output tails must survive both policies.

The handwritten control owns a separate lane loop, first-fault handling, and
transactional shadow buffers. It reuses the core per-operation checked/clamped
numerical law; it is an orchestration control, not an independent wide arithmetic
oracle. Independent fixed arithmetic goldens live in the wide composition tests.

Each of the 416 benchmark cases runs three changing-input comparisons and a
separate 64-call census before timing. The census checks the intermediate,
output, fault, and tails, requires zero warm Rust allocations/reallocations/frees,
and checks that no invocation rescoring occurs. Every measured route uses a
monomorphized closure and includes the same input-update work. Cold preparation,
assertions, and the census stay outside Criterion timing. The handwritten
control allocates its transaction buffers once before calls.

The smoke run establishes execution and warm resource behavior. Performance
measurements require a dedicated idle CPU window; no timing claims accompany
this benchmark.
