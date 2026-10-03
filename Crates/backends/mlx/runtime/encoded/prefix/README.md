# Immutable resident output prefix

`MlxPreparedEncodedPrefix` freezes a retained MLX session, logical low-format tag,
positive prefix length and strictly larger previous output length. It warms the
exact physical UInt dtype and both shapes during cold preparation. Execution
accepts only matching arrays and creates a completed full-length output. A caller
may retain the same previous shape and carry the original checked operation's
recovered fault alongside that new output. Fatal arithmetic must be handled before
calling this publication seam. Equal-length replacement uses the existing direct
completed array path.

The authored C endpoint catches standard and unknown C++ exceptions. It invokes
the pinned MLX `slice_update` operation with exact matching UInt8/UInt16 dtypes,
one dimension, start zero, stop equal to the prefix length and stride one. No
conversion, arithmetic, raw buffer import, host materialization or provider change
is performed. The operation creates a lazy output whose inputs are the actual
old and prefix arrays. Rust holds separate official holders for both inputs and
the output through evaluation, explicit stream synchronization, wait and available
checks. Unknown completion poisons the actual session and retains all pending
holders plus their loaded image. Old owners are never marked as having been
written, although private GPU work may have been submitted.

The pinned MLX0.32.3 source (`64ea011cb65f14d9ce2737e60db9a4ae91ed7441`)
provides the relevant lifetime proof:

- `mlx/ops.cpp::slice_update` constructs a new `SliceUpdate` with `{src, upd}`.
- `mlx/backend/metal/indexing.cpp::SliceUpdate::eval_gpu` copies the source into
  its output, then copies the update into that private output prefix.
- `mlx/backend/common/copy.h::set_copy_output_data` allows buffer donation only
  when `is_donatable` succeeds.
- `mlx/array.h::array::is_donatable` requires both descriptor and data reference
  counts to equal one. The original escaped holder and separately retained
  official holder prevent that condition for the previous array.

The [official C++ operation documentation](https://ml-explore.github.io/mlx/build/html/cpp/ops.html)
describes the functional API; the pinned implementation and actual native sibling
tests establish the precise bounded ownership behavior used here. Copying bytes
does not broaden checked arithmetic or Portable admission. Larger resident input
prefix selection, shared facade integration, matching allocation census and
workspace/headroom enforcement are separate qualification steps.
