# Checked F32 Vulkan binary pairs

Every Add/Sub/Mul/Div workload runs actual `#[pcu]` source at N=1 and N=4096.
Each group pairs source preparation, ordinary source routing, an explicit graph
as an additional diagnostic, and a direct ash control. The control opens a
separate logical device on the same verified physical Vulkan UUID. It compiles
the audited integer GLSL arithmetic source through an independent cold compiler
path and calls no PCU lowering, preparation, admission or execution routine.
The numeric correctness oracle is the separate core integer-significand model.
Sharing the audited arithmetic shader is deliberate and does not constitute an
independent arithmetic implementation.

All four routes stage two fresh input buffers, execute the integer-only shader,
wait for a terminal fence, scan every private status in logical order, and publish
only after every status succeeds. Pipelines, coherent/cached mappings, descriptor
sets, commands and fences are retained. The ownership and synchronous host
boundary match. Output/padded tails and changed inputs are verified before timing.
The default underflow policy is IEEE-after-rounding with PCU faults instead of
IEEE default results. Other policies are exercised by the hardware test matrix.
F64, Clamp, portable reproducibility and other numerical options remain rejected.

Cold qualification asserts exactly zero warm Rust allocations, reallocations and
frees for all four routes. This census excludes allocations inside the Vulkan
driver. Native memory types/flags and physical allocation sizes are printed cold.
The GPU idle guard requires three consecutive readings <=5% from
`PCU_VULKAN_GPU_BUSY_PATH`, defaulting to the local RX 6900 XT counter. Compilation,
discovery, ranking, setup, clock probes and cold assertions are outside Criterion.

```sh
cargo bench -p fusion-pcu-vulkan --features hosted --bench checked_binary -- --test
cargo bench -p fusion-pcu-vulkan --features hosted --bench checked_binary
```

`--test` is a semantic smoke, not statistical timing evidence. Historical Neg
measurements do not qualify this binary implementation.

The `--test` semantic run takes no statistical timing samples and does not require
an idle GPU. All statistical runs retain the activity guard unchanged; desktop
or other compute activity must refuse measurements rather than alter the limit.
