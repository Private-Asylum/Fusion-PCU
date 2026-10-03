# Encoded unary owner replacement

This benchmark compares genuine ordinary `#[pcu]` invocation, its emitted source IR,
an independently assembled graph and the direct MLX-owned kernel control. The four
formats, two unary operations, three underflow policies, two range policies and two
lengths produce 384 semantic peers. The finite changed inputs do not recover faults;
separate lifecycle tests cover recovery and fatal rejection.

All peers borrow preuploaded immutable inputs, execute into a fresh private MLX
array, wait for terminal completion, replace the prior output owner, read the exact
output prefix into an existing host buffer, and drop the replaced owner. Three
distinct input banks and sentinel tails are checked before and after every route,
including filtered registrations. The explicit peers retain the same cached shape
through an `Rc` clone. The ordinary peer uses genuine source-produced `PcuTensor`
owners and the frozen direct invocation route. Resident output length is exactly
the logical invocation count; this benchmark does not qualify larger-owner prefix
replacement.

`--test` runs semantic smoke checks. The separate `allocation-census` feature
reports caller-thread Rust allocations for one warmed matched call. It does not
observe native C++ or device allocator traffic, and no zero-allocation or in-place
execution claim is assumed. Ordinary routing may allocate for publication; the
actual census must be reported rather than attributed to mathematical admission.
Cold staging, source specialization, discovery, kernel compilation and policy
selection are outside the measured boundary. The GPU activity guard runs before
registration. Statistical results, operation costs, workspace and memory-budget
enforcement remain unclaimed until separately measured and admitted.
