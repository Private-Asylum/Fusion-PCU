# Strict MSE source/graph/native comparison

Canonical Criterion `strict_mse` executes actual `#[pcu]` F32/F64 loss beside a
frozen graph and independently launched identical ordered checker. Flattened
inputs of 65, 4096, and 65,536 change between physical banks. One active lane performs
ordered checked subtract/square/sum/divide; this bounded implementation proves
the prescribed contract, and is not a claim of reduction throughput optimization.

Complete output is checked against an independent integer sum of dyadic squares
and one rounded ratio outside timing. Full-host includes payload upload/readback,
terminal status/completion and fresh output release; resident excludes payload
transfers from timing, while still validating every output. Cold compilation is
reported separately. 20 samples, 500 ms warmup, 2 s requested measurement and
Criterion 95% intervals. Activity guards and separate census scope match the
ReLU derivative target. Historical records are not current acceptance.

The current permission extension adds all four independent compound/precision
combinations and all three underflow policies at the small boundary. Original
larger default-profile cases remain. Benchmark group and census labels retain
the requested tuple. New hardware acceptance is recorded separately in the
backend plan; these executable definitions alone do not establish device proof.
Warm census runs 64 changing-input calls for each source/graph/native route,
including matched completion, explicit output readback, oracle and release.
