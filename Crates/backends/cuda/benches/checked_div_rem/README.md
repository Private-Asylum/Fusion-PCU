# Checked integer quotient/remainder authoring qualification

Canonical Criterion target `checked_div_rem` performs real ordinary `#[pcu]`
work for all signed/unsigned 8/16/32/64 widths. It calls the annotated direct
entry on every source iteration; generated IR is separately prepared for the
explicit control and the same admitted GPU kernel is directly launched by the
native control. Source modules also contain genuine grid-stride and Strict
entries exercised by the integration fixture. No CPU fallback or PortableV1
admission is implied. Division truncates toward zero; remainder retains the
dividend sign. Both zero divisor and signed MIN/-1 are fatal for both outputs.

Two sizes,65/65,536, eight widths and five routes yield80 semantic groups:
full-host source/explicit-IR/native and resident source/native. Each call
alternates two genuinely different host/resident input banks. The full quotient
and remainder, their caller tails, and native Rust plus core checked-division
oracles are checked after every completed call outside its timer. Host input
encoding is a typed borrowed byte view. Resident output readback is untimed;
full-host uploads and both output-prefix downloads remain timed.

Cold compilation, allocation, status initialization, staging growth, first
completed calls, independent input/oracle generation and source scoring are
separated. Warm source scoring must stay unchanged. Every route owns a retained
U64 success sentinel, resets after faults and includes kernel/event completion
plus status readback. Source/explicit/native host storage each retains two input
buffers, two N+2 output buffers and status. Resident controls each retain two
two-input physical banks, two N+2 outputs and status. There is no warm device
allocation or module load in this scope. Native ABI/unsafe launch lives only in
`native/ffi/ffi.rs`; it uses the exact cold-admitted four pointers plus status and
identical geometry. The native control is a same-lowering work baseline, not an
independent arithmetic algorithm; independent scalar oracles provide correctness.

`--test` is correctness-only and produces no statistics. Statistical mode retains
the existing hardware activity gate and20samples,500ms warmup,2s requested
measurement. `allocation-census` is a separate build and emits warm caller heap
counts; CUDA additionally asserts0symbol resolutions/module loads/device
allocations/frees and1explicit kernel. Resident census includes untimed complete
readback/oracle. Driver/library internal allocation is outside caller heap scope.

Integration `checked_div_rem_source` exercises fresh values/extrema, direct/grid/
Strict, first logical fault, signed overflow precedence, both-output host rollback,
prelaunch short-output rollback, resident fatal discard/reconstruction/retry and
escaped terminal outputs after input/cache drop. A65,536-lane eight-bit batch
checks every finite valid operand pair; invalid pairs are remapped only in that
successful batch and independently tested as terminal faults. That deliberate
separation prevents one fatal invocation from hiding all finite results.

Run authorized correctness fixtures serially and inspect actual device evidence
before making a numerical or performance claim. Current raw records live under
`.pcu-validation/2026-10-01/rocm-cuda-checked-div-rem/`; active user GPU workloads
must remain untouched. Backend plans distinguish pending/uncontended evidence.
