# Common U32 PIO Stream profile

This profile defines a deliberately small PCU Stream subset that a CPU reference consumer and the
RP2350 PIO adapter can both implement. It defines semantics and admission shape; it is not evidence
that a particular PIO build or board has passed hardware conformance.

## Program shape

A conforming program has exactly two ports, in this order:

1. An input port: U32, stream rate, input direction.
2. An output port: U32, stream rate, output direction.

Port names are descriptive metadata and are not constrained by this profile.

The program has no kernel resource bindings, no declared parameters, no runtime resource bindings,
and no runtime parameter values. It has exactly one pattern. An empty pattern sequence and a
sequence of multiple patterns are rejected. This is a one-word-in, one-word-out transform; it does
not specify framing, packet boundaries, buffering, or end-of-stream behavior. It makes no latency,
throughput, or clock-cycle guarantee.

The accepted patterns are `BitReverse`, `BitInvert`, `Increment`, `Decrement`, `ShiftLeft`,
`ShiftRight`, `ExtractBits`, `MaskLower`, and `ByteSwap32`. `AddParameter` and `XorParameter` are
outside the profile. Shifts accept counts 1 through 32. Extraction accepts width 1 through 32,
offset less than 32, and `offset + width <= 32`. `MaskLower` accepts widths 1 through 32.

## Operation semantics

- `BitReverse` reverses all 32 bits.
- `BitInvert` complements all 32 bits.
- `Increment` and `Decrement` wrap modulo 2^32.
- `ShiftLeft(n)` and `ShiftRight(n)` are logical shifts; shifting by 32 yields zero.
- `ExtractBits { offset, width }` selects that interval, with bit zero as the least significant bit,
  then right-aligns the result.
- `MaskLower(n)` returns the least significant `n` bits and clears the rest.
- `ByteSwap32` reverses the order of the four bytes.

Operations do not compose inside this profile. A caller needing two transforms must represent them
as separate programs or use a later profile that explicitly defines composition and its limits.

## Rejection reporting and reference vectors

`validate_pio_u32_stream_profile` reports the first static shape failure with
`PcuPioU32StreamProfileError`; `validate_pio_u32_stream_invocation` additionally rejects nonempty
runtime binding and parameter tables. Consumers should preserve these distinctions in diagnostics
instead of reducing every failure to “unsupported.”

The core exports `PCU_PIO_U32_STREAM_VECTORS` and
`execute_pio_u32_stream_reference` as reusable conformance inputs and a CPU reference. The core
tests check those vectors against the reference. Hardware conformance requires running the same
vectors on the board and comparing each output; this document does not claim that such a run has
occurred.
