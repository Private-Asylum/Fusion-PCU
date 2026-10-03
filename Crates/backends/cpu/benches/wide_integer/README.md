# Wide integer semantic pairs

`cargo bench -p fusion-pcu-cpu --bench wide_integer --features source-wide -- --test` executes144 paired semantic groups: six wide signed/unsigned representations, three checked operations, N1/N4096, and four real caller-owned output routes. Workloads are genuine generic annotated source; explicit IR is a diagnostic peer. Independent controls retain complete base256 products and exact signed-magnitude range checks, not core PCU arithmetic.

Both changing input states compare full outputs/tails. Each warm256-call source-prepared/ordinary/graph/native scope asserts zero alloc/realloc/free and no cold rescoring. No statistical samples or latency claim is collected by --test. Arithmetic, schema, overflow/underflow rollback, retry and exact offer exclusions are separate integration proofs.
