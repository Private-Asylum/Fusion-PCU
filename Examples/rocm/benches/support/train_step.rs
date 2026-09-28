//! Direct HIP and rocBLAS peer for the linear-regression training graph.

use std::error::Error;
#[rustfmt::skip]
use std::time::{
    Duration,
    Instant,
};

#[rustfmt::skip]
use fusion_pcu_rocm::{
    DeviceBuffer,
    HipKernel,
    HipKernelArgument,
    HipCompletionBatch,
    HipRuntime,
    HipStreamHandle,
    Rocblas,
    RocblasSgemmHostTiming,
    RocmDiscovery,
    compile_hip_source_for_device,
};
use fusion_pcu::PcuObjectRef;
#[cfg(feature = "insights")]
#[rustfmt::skip]
use fusion_pcu::insights::{
    InsightClock,
    InsightLedger,
    InsightRecord,
    InsightStatus,
};

#[derive(Clone, Copy)]
pub struct TrainInputs<'a> {
    pub rows: usize,
    pub features: usize,
    pub samples: &'a [f32],
    pub target: &'a [f32],
    pub factor: &'a [f32],
    pub initial_weights: &'a [f32],
    pub rate: &'a [f32],
}

/// Opt-in wall-clock breakdown for the synchronous native training route.
///
/// `rocblas` and `hip_kernels` include their existing completion waits. These are host-observed
/// phase durations, not device-event timings.
#[derive(Clone, Copy, Debug, Default)]
pub struct NativeTrainTimings {
    pub rocblas: Duration,
    pub hip_kernels: Duration,
    /// Strict-route SGEMM host durations, accumulated over both training steps.
    pub forward_sgemm: Duration,
    pub gradient_sgemm: Duration,
    /// Sum of the rocBLAS setup, C call, device synchronization, and cleanup phases.
    pub forward_sgemm_host: RocblasSgemmHostTiming,
    /// Sum of the rocBLAS setup, C call, device synchronization, and cleanup phases.
    pub gradient_sgemm_host: RocblasSgemmHostTiming,
    /// Strict-route HIP host durations, including launch completion waits.
    pub delta_hip: Duration,
    pub scale_hip: Duration,
    pub update_hip: Duration,
    /// Strict-kernel host launch-call duration, accumulated over both steps.
    pub delta_launch: Duration,
    pub scale_launch: Duration,
    pub update_launch: Duration,
    /// Strict-kernel `completion.wait()` host duration, accumulated over both steps.
    pub delta_wait: Duration,
    pub scale_wait: Duration,
    pub update_wait: Duration,
    /// Host wall time for the batched delta + scale phase, including completion wait.
    pub batched_delta_scale: Duration,
    pub batched_delta_scale_launch: Duration,
    pub batched_delta_scale_wait: Duration,
    /// Host wall time for the batched update phase, including completion wait.
    pub batched_update: Duration,
    pub batched_update_launch: Duration,
    pub batched_update_wait: Duration,
    pub output_readback: Duration,
    pub total: Duration,
}

/// Caller-clock tick records for the strict fully-batched route, available with `insights`.
/// Per-step arrays are ordered as graph step 0 then graph step 1. Every phase record exposes
/// inclusive and exclusive ticks; `final_readback` includes result allocation and copy.
#[cfg(feature = "insights")]
#[derive(Clone, Copy, Debug, Default)]
#[allow(dead_code)] // Shared support module is also compiled by benches that do not use profiling.
pub struct NativeFullyBatchedTickProfile {
    pub forward_sgemm_enqueue: [InsightRecord; 2],
    pub delta_enqueue: [InsightRecord; 2],
    pub scale_enqueue: [InsightRecord; 2],
    pub gradient_sgemm_enqueue: [InsightRecord; 2],
    pub update_enqueue: [InsightRecord; 2],
    /// `batch.finish()`, including event recording and ownership transfer to completion.
    pub batch_finish: [InsightRecord; 2],
    pub completion_wait: [InsightRecord; 2],
    /// Dropping the completed batch and releasing its retained launch resources.
    pub postwait_drop: [InsightRecord; 2],
    pub final_readback: InsightRecord,
    pub whole_call: InsightRecord,
    pub status: InsightStatus,
}

const SOURCE: &str = r#"
extern "C" __global__ void training_error(
    const float *prediction, const float *target, const float *factor,
    float *error, unsigned int n) {
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) error[id] = (prediction[id] - target[id]) * factor[id];
}
extern "C" __global__ void training_delta(
    const float *prediction, const float *target, float *delta, unsigned int n) {
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) delta[id] = prediction[id] - target[id];
}
extern "C" __global__ void training_scale_error(
    const float *delta, const float *factor, float *error, unsigned int n) {
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) error[id] = delta[id] * factor[id];
}
extern "C" __global__ void training_update(
    const float *weights, const float *gradient, const float *rate,
    float *updated, unsigned int n) {
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) updated[id] = weights[id] - rate[id] * gradient[id];
}
extern "C" __global__ void training_update_strict(
    const float *weights, const float *gradient, float rate,
    float *updated, unsigned int n) {
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) {
        volatile float product = rate * gradient[id];
        updated[id] = weights[id] - product;
    }
}
"#;

#[derive(Clone, Copy)]
enum TrainingRoute {
    Combined,
    Strict { learning_rate: f32 },
}

/// Preallocated direct HIP/rocBLAS route; all graph inputs stay device resident.
pub struct NativeTrainStep {
    stream: HipStreamHandle,
    blas: Rocblas,
    error_kernel: HipKernel,
    delta_kernel: HipKernel,
    scale_error_kernel: HipKernel,
    update_kernel: HipKernel,
    strict_update_kernel: HipKernel,
    samples: DeviceBuffer,
    target: DeviceBuffer,
    factor: DeviceBuffer,
    delta: Option<DeviceBuffer>,
    rate: Option<DeviceBuffer>,
    prediction: DeviceBuffer,
    error: DeviceBuffer,
    gradient: DeviceBuffer,
    updated: DeviceBuffer,
    initial_weights: DeviceBuffer,
    weights: DeviceBuffer,
    rows: usize,
    features: usize,
    route: TrainingRoute,
}

#[cfg(feature = "insights")]
macro_rules! native_phase {
    ($timings:expr, $phase:expr, $step:expr, $operation:block) => {
        $timings.measure($phase, $step, || $operation.map_err(Into::into))
    };
}

#[cfg(not(feature = "insights"))]
macro_rules! native_phase {
    ($timings:expr, $phase:expr, $step:expr, $operation:block) => {
        $operation
    };
}

impl NativeTrainStep {
    /// Replace resident strict-route inputs without rebuilding kernels or intermediate storage.
    /// Previous executions must have completed before this method is called.
    #[allow(dead_code)] // Used by the volume bench; the fixed-input bench shares this module.
    pub fn replace_strict_inputs(
        &mut self,
        samples: &[f32],
        target: &[f32],
        initial_weights: &[f32],
    ) -> Result<(), Box<dyn Error>> {
        if !matches!(self.route, TrainingRoute::Strict { .. })
            || samples.len()
                != self
                    .rows
                    .checked_mul(self.features)
                    .ok_or("input extent overflow")?
            || target.len() != self.rows
            || initial_weights.len() != self.features
        {
            return Err("replacement strict inputs have an incompatible route or shape".into());
        }
        self.samples.copy_from(bytemuck::cast_slice(samples))?;
        self.target.copy_from(bytemuck::cast_slice(target))?;
        self.initial_weights
            .copy_from(bytemuck::cast_slice(initial_weights))?;
        Ok(())
    }

    pub fn prepare(
        discovery: &RocmDiscovery,
        device: PcuObjectRef,
        input: &TrainInputs<'_>,
    ) -> Result<Self, Box<dyn Error>> {
        Self::prepare_with_route(discovery, device, input, TrainingRoute::Combined)
    }

    /// Prepares a route-matched peer for PCU's strict `delta -> scale -> sgd_update` graph.
    ///
    /// Each pointwise stage has its own kernel boundary, and the update product is forced to
    /// round before subtraction so HIP cannot contract it into an FMA.
    pub fn prepare_strict(
        discovery: &RocmDiscovery,
        device: PcuObjectRef,
        input: &TrainInputs<'_>,
        learning_rate: f32,
    ) -> Result<Self, Box<dyn Error>> {
        Self::prepare_with_route(
            discovery,
            device,
            input,
            TrainingRoute::Strict { learning_rate },
        )
    }

    fn prepare_with_route(
        discovery: &RocmDiscovery,
        device: PcuObjectRef,
        input: &TrainInputs<'_>,
        route: TrainingRoute,
    ) -> Result<Self, Box<dyn Error>> {
        let TrainInputs {
            rows,
            features,
            samples,
            target,
            factor,
            initial_weights,
            rate,
        } = *input;
        let runtime = discovery.open_device(device)?;
        let image = compile_hip_source_for_device(&runtime, SOURCE)?;
        let module = runtime.load_module(&image)?;
        let error_kernel = module.function(c"training_error")?;
        let delta_kernel = module.function(c"training_delta")?;
        let scale_error_kernel = module.function(c"training_scale_error")?;
        let update_kernel = module.function(c"training_update")?;
        let strict_update_kernel = module.function(c"training_update_strict")?;
        let stream = runtime.create_stream()?;
        let mut blas = Rocblas::new(&runtime)?;
        // Match the PCU tensor assessor: GEMMs and dependent HIP kernels share one stream.
        blas.bind_stream(&stream)?;
        let samples_buffer = upload(&runtime, samples)?;
        let target_buffer = upload(&runtime, target)?;
        let factor_buffer = upload(&runtime, factor)?;
        let delta_buffer = match route {
            TrainingRoute::Combined => None,
            TrainingRoute::Strict { .. } => Some(allocate(&runtime, rows)?),
        };
        let rate_buffer = match route {
            TrainingRoute::Combined => Some(upload(&runtime, rate)?),
            TrainingRoute::Strict { .. } => None,
        };
        let prediction = allocate(&runtime, rows)?;
        let error = allocate(&runtime, rows)?;
        let gradient = allocate(&runtime, features)?;
        let updated = allocate(&runtime, features)?;
        // Keep the caller's starting point immutable and resident. Each execution writes its
        // two updates into separate reusable buffers without resetting the initial allocation.
        let initial_weights = upload(&runtime, initial_weights)?;
        let weights = allocate(&runtime, features)?;
        Ok(Self {
            stream,
            blas,
            error_kernel,
            delta_kernel,
            scale_error_kernel,
            update_kernel,
            strict_update_kernel,
            samples: samples_buffer,
            target: target_buffer,
            factor: factor_buffer,
            delta: delta_buffer,
            rate: rate_buffer,
            prediction,
            error,
            gradient,
            updated,
            initial_weights,
            weights,
            rows,
            features,
            route,
        })
    }

    pub fn execute_two(&self) -> Result<Vec<f32>, Box<dyn Error>> {
        self.execute_two_inner(None, false)
    }

    /// Run two strict-route steps while batching adjacent HIP pointwise launches.
    ///
    /// Delta and scale complete as one batch before the gradient SGEMM reads `error`. Each
    /// update completes before the following step reads its output as weights.
    pub fn execute_two_batched(&self) -> Result<Vec<f32>, Box<dyn Error>> {
        self.execute_two_inner(None, true)
    }

    /// Run the strict route with each complete graph step queued into one same-stream batch.
    ///
    /// The two SGEMMs and the dependent pointwise kernels share a batch. Each graph step waits
    /// once before the next step consumes its updated weights, matching the PCU step boundary.
    pub fn execute_two_fully_batched(&self) -> Result<Vec<f32>, Box<dyn Error>> {
        #[cfg(feature = "insights")]
        {
            self.execute_two_fully_batched_inner(&mut NoopFullyBatchedTiming)
        }
        #[cfg(not(feature = "insights"))]
        {
            self.execute_two_fully_batched_inner()
        }
    }

    /// Collect caller-clock phase records for the strict fully-batched route.
    ///
    /// Available only with the example crate's `insights` feature. These are host timings, not GPU
    /// event durations; caller-provided clock context changes invalidate samples per core policy.
    #[cfg(feature = "insights")]
    #[allow(dead_code)] // Used by the opt-in training insights bench.
    pub fn execute_two_fully_batched_profiled<C: InsightClock>(
        &self,
        clock: &mut C,
    ) -> Result<(Vec<f32>, NativeFullyBatchedTickProfile), Box<dyn Error>> {
        let mut ledger = InsightLedger::<_, 18, 2>::new(BorrowedInsightClock(clock));
        let result = ledger.scope(0, |ledger| {
            self.execute_two_fully_batched_inner(&mut CollectFullyBatchedTiming(ledger))
        })?;
        let records = ledger.records();
        Ok((
            result,
            NativeFullyBatchedTickProfile {
                forward_sgemm_enqueue: [records[1], records[9]],
                delta_enqueue: [records[2], records[10]],
                scale_enqueue: [records[3], records[11]],
                gradient_sgemm_enqueue: [records[4], records[12]],
                update_enqueue: [records[5], records[13]],
                batch_finish: [records[6], records[14]],
                completion_wait: [records[7], records[15]],
                postwait_drop: [records[8], records[16]],
                final_readback: records[17],
                whole_call: records[0],
                status: ledger.status(),
            },
        ))
    }

    #[allow(
        clippy::explicit_counter_loop, // Step index selects the corresponding fixed ledger slots.
        clippy::too_many_lines // Keep ordered graph phases beside their enqueue operations.
    )]
    fn execute_two_fully_batched_inner(
        &self,
        #[cfg(feature = "insights")] timings: &mut impl FullyBatchedTimingSink,
    ) -> Result<Vec<f32>, Box<dyn Error>> {
        let TrainingRoute::Strict { learning_rate } = self.route else {
            return Err("fully batched execution requires the strict native route".into());
        };
        let delta = self
            .delta
            .as_ref()
            .ok_or("strict route has no delta buffer")?;

        #[cfg(feature = "insights")]
        let mut step_index = 0;
        for (weights, updated) in [
            (&self.initial_weights, &self.weights),
            (&self.weights, &self.updated),
        ] {
            #[cfg(feature = "insights")]
            let step = step_index;
            let mut batch = HipCompletionBatch::new(&self.stream);
            native_phase!(timings, FullyBatchedPhase::ForwardSgemm, step, {
                self.blas.sgemm_into_batch(
                    &mut batch,
                    false,
                    false,
                    1,
                    self.rows,
                    self.features,
                    1.0,
                    weights,
                    1,
                    &self.samples,
                    self.features,
                    0.0,
                    &self.prediction,
                    1,
                )
            })?;
            native_phase!(timings, FullyBatchedPhase::DeltaEnqueue, step, {
                launch_binary_into_batch(
                    &self.delta_kernel,
                    &mut batch,
                    &self.prediction,
                    &self.target,
                    delta,
                    self.rows,
                )
            })?;
            native_phase!(timings, FullyBatchedPhase::ScaleEnqueue, step, {
                launch_binary_into_batch(
                    &self.scale_error_kernel,
                    &mut batch,
                    delta,
                    &self.factor,
                    &self.error,
                    self.rows,
                )
            })?;
            native_phase!(timings, FullyBatchedPhase::GradientSgemm, step, {
                self.blas.sgemm_into_batch(
                    &mut batch,
                    false,
                    true,
                    1,
                    self.features,
                    self.rows,
                    1.0,
                    &self.error,
                    1,
                    &self.samples,
                    self.features,
                    0.0,
                    &self.gradient,
                    1,
                )
            })?;
            native_phase!(timings, FullyBatchedPhase::UpdateEnqueue, step, {
                launch_scalar_update_into_batch(
                    &self.strict_update_kernel,
                    &mut batch,
                    weights,
                    &self.gradient,
                    learning_rate,
                    updated,
                    self.features,
                )
            })?;
            let mut completion = native_phase!(timings, FullyBatchedPhase::BatchFinish, step, {
                batch.finish()
            })?;
            let wait_result = native_phase!(timings, FullyBatchedPhase::CompletionWait, step, {
                completion.wait()
            });
            native_phase!(timings, FullyBatchedPhase::PostwaitDrop, step, {
                drop(completion);
                Ok::<(), Box<dyn Error>>(())
            })?;
            wait_result?;
            #[cfg(feature = "insights")]
            {
                step_index += 1;
            }
        }

        native_phase!(timings, FullyBatchedPhase::FinalReadback, 0, {
            let mut result = vec![0.0_f32; self.features];
            self.updated
                .copy_to(bytemuck::cast_slice_mut(&mut result))?;
            Ok::<Vec<f32>, Box<dyn Error>>(result)
        })
    }

    /// Run two strict-route batched steps while profiling host launch and completion waits.
    pub fn execute_two_batched_host_profiled(
        &self,
    ) -> Result<(Vec<f32>, NativeTrainTimings), Box<dyn Error>> {
        let started = Instant::now();
        let mut timings = NativeTrainTimings::default();
        let result = self.execute_two_inner(Some(&mut timings), true)?;
        timings.total = started.elapsed();
        Ok((result, timings))
    }

    /// Run two strict-route steps with HIP device-event durations for each batched kernel phase.
    ///
    /// The returned durations are delta+scale, update, delta+scale, update in milliseconds.
    /// SGEMM and readback remain outside the four measured device segments.
    pub fn execute_two_batched_device_timed(&self) -> Result<(Vec<f32>, Vec<f32>), Box<dyn Error>> {
        let TrainingRoute::Strict { learning_rate } = self.route else {
            return Err("batched execution requires the strict native route".into());
        };
        let delta = self
            .delta
            .as_ref()
            .ok_or("strict route has no delta buffer")?;
        let mut durations = Vec::with_capacity(4);

        for (weights, updated) in [
            (&self.initial_weights, &self.weights),
            (&self.weights, &self.updated),
        ] {
            // Row-major X*W is column-major W^T*X^T.
            self.blas.sgemm(
                false,
                false,
                1,
                self.rows,
                self.features,
                1.0,
                weights,
                1,
                &self.samples,
                self.features,
                0.0,
                &self.prediction,
                1,
            )?;
            durations.extend(launch_delta_scale_batched_timed(
                &self.delta_kernel,
                &self.scale_error_kernel,
                &self.stream,
                &self.prediction,
                &self.target,
                delta,
                &self.factor,
                &self.error,
                self.rows,
            )?);

            // Compute row-major X^T * error through its column-major transpose: error^T * X.
            self.blas.sgemm(
                false,
                true,
                1,
                self.features,
                self.rows,
                1.0,
                &self.error,
                1,
                &self.samples,
                self.features,
                0.0,
                &self.gradient,
                1,
            )?;
            durations.extend(launch_scalar_update_batched_timed(
                &self.strict_update_kernel,
                &self.stream,
                weights,
                &self.gradient,
                learning_rate,
                updated,
                self.features,
            )?);
        }

        let mut result = vec![0.0_f32; self.features];
        self.updated
            .copy_to(bytemuck::cast_slice_mut(&mut result))?;
        Ok((result, durations))
    }

    /// Run the same workload while recording host wall time around BLAS, HIP, and readback phases.
    ///
    /// The current native APIs wait inside every SGEMM and HIP launch, so these values include
    /// those waits and must not be interpreted as isolated GPU execution times.
    pub fn execute_two_profiled(&self) -> Result<(Vec<f32>, NativeTrainTimings), Box<dyn Error>> {
        let started = Instant::now();
        let mut timings = NativeTrainTimings::default();
        let result = self.execute_two_inner(Some(&mut timings), false)?;
        timings.total = started.elapsed();
        Ok((result, timings))
    }

    #[allow(clippy::too_many_lines)] // Keep profiled and ordinary execution on one workload path.
    fn execute_two_inner(
        &self,
        mut timings: Option<&mut NativeTrainTimings>,
        batched: bool,
    ) -> Result<Vec<f32>, Box<dyn Error>> {
        if batched && matches!(self.route, TrainingRoute::Combined) {
            return Err("batched execution requires the strict native route".into());
        }
        for (weights, updated) in [
            (&self.initial_weights, &self.weights),
            (&self.weights, &self.updated),
        ] {
            // Row-major X*W is column-major W^T*X^T.
            timed_sgemm(
                timings.as_deref_mut(),
                SgemmTiming::Forward,
                matches!(self.route, TrainingRoute::Strict { .. }),
                |host_timing| {
                    host_timing.map_or_else(
                        || {
                            self.blas
                                .sgemm(
                                    false,
                                    false,
                                    1,
                                    self.rows,
                                    self.features,
                                    1.0,
                                    weights,
                                    1,
                                    &self.samples,
                                    self.features,
                                    0.0,
                                    &self.prediction,
                                    1,
                                )
                                .map_err(Into::into)
                        },
                        |host_timing| {
                            self.blas
                                .sgemm_profiled(
                                    false,
                                    false,
                                    1,
                                    self.rows,
                                    self.features,
                                    1.0,
                                    weights,
                                    1,
                                    &self.samples,
                                    self.features,
                                    0.0,
                                    &self.prediction,
                                    1,
                                    host_timing,
                                )
                                .map_err(Into::into)
                        },
                    )
                },
            )?;
            match self.route {
                TrainingRoute::Combined => timed(
                    timings
                        .as_deref_mut()
                        .map(|timings| &mut timings.hip_kernels),
                    || {
                        launch_error(
                            &self.error_kernel,
                            &self.stream,
                            &self.prediction,
                            &self.target,
                            &self.factor,
                            &self.error,
                            self.rows,
                        )
                    },
                )?,
                TrainingRoute::Strict { .. } => {
                    let delta = self
                        .delta
                        .as_ref()
                        .ok_or("strict route has no delta buffer")?;
                    if batched {
                        timed_batched_native(
                            timings.as_deref_mut(),
                            BatchedNativeTiming::DeltaScale,
                            |timing| {
                                launch_delta_scale_batched(
                                    &self.delta_kernel,
                                    &self.scale_error_kernel,
                                    &self.stream,
                                    &self.prediction,
                                    &self.target,
                                    delta,
                                    &self.factor,
                                    &self.error,
                                    self.rows,
                                    timing,
                                )
                            },
                        )?;
                    } else {
                        timed_native(
                            timings.as_deref_mut(),
                            NativeTiming::Delta,
                            true,
                            |timing| {
                                launch_binary(
                                    &self.delta_kernel,
                                    &self.stream,
                                    &self.prediction,
                                    &self.target,
                                    delta,
                                    self.rows,
                                    timing,
                                )
                            },
                        )?;
                        timed_native(
                            timings.as_deref_mut(),
                            NativeTiming::Scale,
                            true,
                            |timing| {
                                launch_binary(
                                    &self.scale_error_kernel,
                                    &self.stream,
                                    delta,
                                    &self.factor,
                                    &self.error,
                                    self.rows,
                                    timing,
                                )
                            },
                        )?;
                    }
                }
            }
            // Compute the row-major X^T * error through its column-major transpose:
            // error^T * X. This matches the ROCm tensor adapter's rocBLAS orientation.
            timed_sgemm(
                timings.as_deref_mut(),
                SgemmTiming::Gradient,
                matches!(self.route, TrainingRoute::Strict { .. }),
                |host_timing| {
                    host_timing.map_or_else(
                        || {
                            self.blas
                                .sgemm(
                                    false,
                                    true,
                                    1,
                                    self.features,
                                    self.rows,
                                    1.0,
                                    &self.error,
                                    1,
                                    &self.samples,
                                    self.features,
                                    0.0,
                                    &self.gradient,
                                    1,
                                )
                                .map_err(Into::into)
                        },
                        |host_timing| {
                            self.blas
                                .sgemm_profiled(
                                    false,
                                    true,
                                    1,
                                    self.features,
                                    self.rows,
                                    1.0,
                                    &self.error,
                                    1,
                                    &self.samples,
                                    self.features,
                                    0.0,
                                    &self.gradient,
                                    1,
                                    host_timing,
                                )
                                .map_err(Into::into)
                        },
                    )
                },
            )?;
            match self.route {
                TrainingRoute::Combined => timed(
                    timings
                        .as_deref_mut()
                        .map(|timings| &mut timings.hip_kernels),
                    || {
                        launch_update(
                            &self.update_kernel,
                            &self.stream,
                            weights,
                            &self.gradient,
                            self.rate
                                .as_ref()
                                .ok_or("combined route has no rate buffer")?,
                            updated,
                            self.features,
                        )
                    },
                )?,
                TrainingRoute::Strict { learning_rate } => {
                    if batched {
                        timed_batched_native(
                            timings.as_deref_mut(),
                            BatchedNativeTiming::Update,
                            |timing| {
                                launch_scalar_update_batched(
                                    &self.strict_update_kernel,
                                    &self.stream,
                                    weights,
                                    &self.gradient,
                                    learning_rate,
                                    updated,
                                    self.features,
                                    timing,
                                )
                            },
                        )?;
                    } else {
                        timed_native(
                            timings.as_deref_mut(),
                            NativeTiming::Update,
                            true,
                            |timing| {
                                launch_scalar_update(
                                    &self.strict_update_kernel,
                                    &self.stream,
                                    weights,
                                    &self.gradient,
                                    learning_rate,
                                    updated,
                                    self.features,
                                    timing,
                                )
                            },
                        )?;
                    }
                }
            }
        }
        let mut result = vec![0.0_f32; self.features];
        timed(timings.map(|timings| &mut timings.output_readback), || {
            self.updated
                .copy_to(bytemuck::cast_slice_mut(&mut result))
                .map_err(Into::into)
        })?;
        Ok(result)
    }
}

#[cfg(feature = "insights")]
#[derive(Clone, Copy)]
enum FullyBatchedPhase {
    ForwardSgemm,
    DeltaEnqueue,
    ScaleEnqueue,
    GradientSgemm,
    UpdateEnqueue,
    BatchFinish,
    CompletionWait,
    PostwaitDrop,
    FinalReadback,
}

#[cfg(feature = "insights")]
trait FullyBatchedTimingSink {
    fn measure<T>(
        &mut self,
        phase: FullyBatchedPhase,
        step: usize,
        operation: impl FnOnce() -> Result<T, Box<dyn Error>>,
    ) -> Result<T, Box<dyn Error>>;
}

#[cfg(feature = "insights")]
struct NoopFullyBatchedTiming;

#[cfg(feature = "insights")]
impl FullyBatchedTimingSink for NoopFullyBatchedTiming {
    fn measure<T>(
        &mut self,
        _: FullyBatchedPhase,
        _: usize,
        operation: impl FnOnce() -> Result<T, Box<dyn Error>>,
    ) -> Result<T, Box<dyn Error>> {
        operation()
    }
}

#[cfg(feature = "insights")]
#[allow(dead_code)] // Shared support module is also compiled by benches that do not use profiling.
struct BorrowedInsightClock<'a, C>(&'a mut C);

#[cfg(feature = "insights")]
impl<C: InsightClock> InsightClock for BorrowedInsightClock<'_, C> {
    fn stamp(&mut self) -> fusion_pcu::insights::InsightStamp {
        self.0.stamp()
    }
}

#[cfg(feature = "insights")]
#[allow(dead_code)] // Constructed only by the opt-in training insights bench.
struct CollectFullyBatchedTiming<'a, C: InsightClock>(&'a mut InsightLedger<C, 18, 2>);

#[cfg(feature = "insights")]
impl<C: InsightClock> FullyBatchedTimingSink for CollectFullyBatchedTiming<'_, C> {
    fn measure<T>(
        &mut self,
        phase: FullyBatchedPhase,
        step: usize,
        operation: impl FnOnce() -> Result<T, Box<dyn Error>>,
    ) -> Result<T, Box<dyn Error>> {
        let point = phase.point(step);
        self.0.scope(point, |_| operation())
    }
}

#[cfg(feature = "insights")]
impl FullyBatchedPhase {
    #[cfg(feature = "insights")]
    const fn point(self, step: usize) -> usize {
        let per_step = match self {
            Self::ForwardSgemm => 0,
            Self::DeltaEnqueue => 1,
            Self::ScaleEnqueue => 2,
            Self::GradientSgemm => 3,
            Self::UpdateEnqueue => 4,
            Self::BatchFinish => 5,
            Self::CompletionWait => 6,
            Self::PostwaitDrop => 7,
            Self::FinalReadback => return 17,
        };
        1 + step * 8 + per_step
    }
}

fn timed<T>(
    duration: Option<&mut Duration>,
    operation: impl FnOnce() -> Result<T, Box<dyn Error>>,
) -> Result<T, Box<dyn Error>> {
    match duration {
        Some(duration) => {
            let started = Instant::now();
            let result = operation();
            *duration += started.elapsed();
            result
        }
        None => operation(),
    }
}

#[derive(Clone, Copy)]
enum NativeTiming {
    Delta,
    Scale,
    Update,
}

#[derive(Clone, Copy)]
enum BatchedNativeTiming {
    DeltaScale,
    Update,
}

fn timed_batched_native<T>(
    timings: Option<&mut NativeTrainTimings>,
    operation_timing: BatchedNativeTiming,
    operation: impl FnOnce(Option<&mut NativeTrainTimings>) -> Result<T, Box<dyn Error>>,
) -> Result<T, Box<dyn Error>> {
    if let Some(timings) = timings {
        let started = Instant::now();
        let result = operation(Some(&mut *timings));
        let elapsed = started.elapsed();
        match operation_timing {
            BatchedNativeTiming::DeltaScale => timings.batched_delta_scale += elapsed,
            BatchedNativeTiming::Update => timings.batched_update += elapsed,
        }
        timings.hip_kernels += elapsed;
        result
    } else {
        operation(None)
    }
}

#[derive(Default)]
struct HipLaunchHostTiming {
    launch_return: Duration,
    completion_wait: Duration,
}

fn timed_native<T>(
    timings: Option<&mut NativeTrainTimings>,
    operation_timing: NativeTiming,
    strict_route: bool,
    operation: impl FnOnce(Option<&mut HipLaunchHostTiming>) -> Result<T, Box<dyn Error>>,
) -> Result<T, Box<dyn Error>> {
    if let Some(timings) = timings {
        let mut hip_timing = HipLaunchHostTiming::default();
        let started = Instant::now();
        let result = operation(Some(&mut hip_timing));
        let elapsed = started.elapsed();
        match operation_timing {
            NativeTiming::Delta | NativeTiming::Scale | NativeTiming::Update => {
                timings.hip_kernels += elapsed;
                if strict_route {
                    match operation_timing {
                        NativeTiming::Delta => {
                            timings.delta_hip += elapsed;
                            timings.delta_launch += hip_timing.launch_return;
                            timings.delta_wait += hip_timing.completion_wait;
                        }
                        NativeTiming::Scale => {
                            timings.scale_hip += elapsed;
                            timings.scale_launch += hip_timing.launch_return;
                            timings.scale_wait += hip_timing.completion_wait;
                        }
                        NativeTiming::Update => {
                            timings.update_hip += elapsed;
                            timings.update_launch += hip_timing.launch_return;
                            timings.update_wait += hip_timing.completion_wait;
                        }
                    }
                }
            }
        }
        result
    } else {
        operation(None)
    }
}

#[derive(Clone, Copy)]
enum SgemmTiming {
    Forward,
    Gradient,
}

fn timed_sgemm<T>(
    timings: Option<&mut NativeTrainTimings>,
    operation_timing: SgemmTiming,
    strict_route: bool,
    operation: impl FnOnce(Option<&mut RocblasSgemmHostTiming>) -> Result<T, Box<dyn Error>>,
) -> Result<T, Box<dyn Error>> {
    let Some(timings) = timings else {
        return operation(None);
    };

    let mut host_timing = RocblasSgemmHostTiming::default();
    let started = Instant::now();
    let result = operation(Some(&mut host_timing));
    let elapsed = started.elapsed();
    timings.rocblas += elapsed;
    if strict_route {
        match operation_timing {
            SgemmTiming::Forward => timings.forward_sgemm += elapsed,
            SgemmTiming::Gradient => timings.gradient_sgemm += elapsed,
        }
    }
    add_sgemm_timing(
        match operation_timing {
            SgemmTiming::Forward => &mut timings.forward_sgemm_host,
            SgemmTiming::Gradient => &mut timings.gradient_sgemm_host,
        },
        host_timing,
    );
    result
}

fn add_sgemm_timing(total: &mut RocblasSgemmHostTiming, sample: RocblasSgemmHostTiming) {
    total.preflight += sample.preflight;
    total.rocblas_call += sample.rocblas_call;
    total.device_synchronize += sample.device_synchronize;
    total.cleanup += sample.cleanup;
}

#[allow(clippy::too_many_arguments)] // Keep both pointwise stages on one explicit batch.
fn launch_delta_scale_batched(
    delta_kernel: &HipKernel,
    scale_kernel: &HipKernel,
    stream: &HipStreamHandle,
    prediction: &DeviceBuffer,
    target: &DeviceBuffer,
    delta: &DeviceBuffer,
    factor: &DeviceBuffer,
    error: &DeviceBuffer,
    count: usize,
    mut timing: Option<&mut NativeTrainTimings>,
) -> Result<(), Box<dyn Error>> {
    let mut batch = HipCompletionBatch::new(stream);
    let launch_started = timing.as_ref().map(|_| Instant::now());
    launch_binary_into_batch(delta_kernel, &mut batch, prediction, target, delta, count)?;
    launch_binary_into_batch(scale_kernel, &mut batch, delta, factor, error, count)?;
    if let (Some(timing), Some(launch_started)) = (timing.as_deref_mut(), launch_started) {
        timing.batched_delta_scale_launch += launch_started.elapsed();
    }
    finish_batch_host_timed(&mut batch, timing, false)
}

fn finish_batch_host_timed(
    batch: &mut HipCompletionBatch,
    timing: Option<&mut NativeTrainTimings>,
    update: bool,
) -> Result<(), Box<dyn Error>> {
    let Some(timing) = timing else {
        return finish_batch(batch);
    };
    let mut completion = batch.finish()?;
    let started = Instant::now();
    let result = completion.wait();
    let elapsed = started.elapsed();
    if update {
        timing.batched_update_wait += elapsed;
    } else {
        timing.batched_delta_scale_wait += elapsed;
    }
    result?;
    Ok(())
}

#[allow(clippy::too_many_arguments)] // Keep both pointwise stages on one explicit batch.
fn launch_delta_scale_batched_timed(
    delta_kernel: &HipKernel,
    scale_kernel: &HipKernel,
    stream: &HipStreamHandle,
    prediction: &DeviceBuffer,
    target: &DeviceBuffer,
    delta: &DeviceBuffer,
    factor: &DeviceBuffer,
    error: &DeviceBuffer,
    count: usize,
) -> Result<Vec<f32>, Box<dyn Error>> {
    let mut batch = HipCompletionBatch::new_timed(stream);
    launch_binary_into_batch(delta_kernel, &mut batch, prediction, target, delta, count)?;
    launch_binary_into_batch(scale_kernel, &mut batch, delta, factor, error, count)?;
    finish_batch_timed(&mut batch)
}

fn launch_binary_into_batch(
    kernel: &HipKernel,
    batch: &mut HipCompletionBatch,
    a: &DeviceBuffer,
    b: &DeviceBuffer,
    output: &DeviceBuffer,
    count: usize,
) -> Result<(), Box<dyn Error>> {
    const BLOCK_SIZE: u32 = 64;
    let count_u32 = u32::try_from(count)?;
    let count_bytes = count_u32.to_ne_bytes();
    let args = [
        HipKernelArgument::Buffer(a),
        HipKernelArgument::Buffer(b),
        HipKernelArgument::Buffer(output),
        HipKernelArgument::Bytes(&count_bytes),
    ];
    // SAFETY: This matches the training_delta and training_scale_error ABIs. All arguments stay
    // alive through enqueue; the batch retains launch resources until its final completion.
    #[allow(unsafe_code)]
    unsafe {
        kernel.launch_into_batch(
            batch,
            [count_u32.div_ceil(BLOCK_SIZE), 1, 1],
            [BLOCK_SIZE, 1, 1],
            0,
            &args,
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)] // Keep the strict update ABI arguments explicit.
fn launch_scalar_update_batched(
    kernel: &HipKernel,
    stream: &HipStreamHandle,
    weights: &DeviceBuffer,
    gradient: &DeviceBuffer,
    learning_rate: f32,
    updated: &DeviceBuffer,
    count: usize,
    mut timing: Option<&mut NativeTrainTimings>,
) -> Result<(), Box<dyn Error>> {
    let mut batch = HipCompletionBatch::new(stream);
    let launch_started = timing.as_ref().map(|_| Instant::now());
    launch_scalar_update_into_batch(
        kernel,
        &mut batch,
        weights,
        gradient,
        learning_rate,
        updated,
        count,
    )?;
    if let (Some(timing), Some(launch_started)) = (timing.as_deref_mut(), launch_started) {
        timing.batched_update_launch += launch_started.elapsed();
    }
    finish_batch_host_timed(&mut batch, timing, true)
}

#[allow(clippy::too_many_arguments)] // Keep the strict update ABI arguments explicit.
fn launch_scalar_update_into_batch(
    kernel: &HipKernel,
    batch: &mut HipCompletionBatch,
    weights: &DeviceBuffer,
    gradient: &DeviceBuffer,
    learning_rate: f32,
    updated: &DeviceBuffer,
    count: usize,
) -> Result<(), Box<dyn Error>> {
    let count_u32 = u32::try_from(count)?;
    let learning_rate_bytes = learning_rate.to_ne_bytes();
    let count_bytes = count_u32.to_ne_bytes();
    let args = [
        HipKernelArgument::Buffer(weights),
        HipKernelArgument::Buffer(gradient),
        HipKernelArgument::Bytes(&learning_rate_bytes),
        HipKernelArgument::Buffer(updated),
        HipKernelArgument::Bytes(&count_bytes),
    ];
    // SAFETY: This matches the strict update ABI. Buffer lifetimes extend through enqueue, and
    // the batch retains launch resources until its final completion is waited.
    #[allow(unsafe_code)]
    unsafe {
        kernel.launch_into_batch(
            batch,
            [count_u32.div_ceil(256), 1, 1],
            [256, 1, 1],
            0,
            &args,
        )?;
    }
    Ok(())
}

fn launch_scalar_update_batched_timed(
    kernel: &HipKernel,
    stream: &HipStreamHandle,
    weights: &DeviceBuffer,
    gradient: &DeviceBuffer,
    learning_rate: f32,
    updated: &DeviceBuffer,
    count: usize,
) -> Result<Vec<f32>, Box<dyn Error>> {
    let count_u32 = u32::try_from(count)?;
    let learning_rate_bytes = learning_rate.to_ne_bytes();
    let count_bytes = count_u32.to_ne_bytes();
    let args = [
        HipKernelArgument::Buffer(weights),
        HipKernelArgument::Buffer(gradient),
        HipKernelArgument::Bytes(&learning_rate_bytes),
        HipKernelArgument::Buffer(updated),
        HipKernelArgument::Bytes(&count_bytes),
    ];
    let mut batch = HipCompletionBatch::new_timed(stream);
    // SAFETY: This matches the strict update ABI. Buffer lifetimes extend through enqueue, and
    // the batch retains launch resources until the final completion is waited.
    #[allow(unsafe_code)]
    unsafe {
        kernel.launch_into_batch(
            &mut batch,
            [count_u32.div_ceil(256), 1, 1],
            [256, 1, 1],
            0,
            &args,
        )?;
    }
    finish_batch_timed(&mut batch)
}

fn finish_batch(batch: &mut HipCompletionBatch) -> Result<(), Box<dyn Error>> {
    let mut completion = batch.finish()?;
    completion.wait()?;
    Ok(())
}

fn finish_batch_timed(batch: &mut HipCompletionBatch) -> Result<Vec<f32>, Box<dyn Error>> {
    let mut completion = batch.finish()?;
    completion.wait()?;
    Ok(batch.timings_ms())
}

fn launch_binary(
    kernel: &HipKernel,
    stream: &HipStreamHandle,
    a: &DeviceBuffer,
    b: &DeviceBuffer,
    output: &DeviceBuffer,
    count: usize,
    mut timing: Option<&mut HipLaunchHostTiming>,
) -> Result<(), Box<dyn Error>> {
    // Strict delta/scale mirror PCU tensor Sub/Mul dispatch, whose session uses 64 threads;
    // the combined historical route and strict SGD update retain their 256-thread geometry.
    const BLOCK_SIZE: u32 = 64;
    let count_u32 = u32::try_from(count)?;
    let count_bytes = count_u32.to_ne_bytes();
    let args = [
        HipKernelArgument::Buffer(a),
        HipKernelArgument::Buffer(b),
        HipKernelArgument::Buffer(output),
        HipKernelArgument::Bytes(&count_bytes),
    ];
    // SAFETY: The kernel takes three f32 pointers then a u32 length. The buffers are large enough,
    // the grid-stride tail is guarded, and waiting establishes lifetime-safe completion.
    #[allow(unsafe_code)]
    let mut completion = match timing.as_deref_mut() {
        Some(timing) => {
            let started = Instant::now();
            let completion = unsafe {
                kernel.launch(
                    stream,
                    [count_u32.div_ceil(BLOCK_SIZE), 1, 1],
                    [BLOCK_SIZE, 1, 1],
                    0,
                    &args,
                )
            };
            timing.launch_return += started.elapsed();
            completion?
        }
        None => unsafe {
            kernel.launch(
                stream,
                [count_u32.div_ceil(BLOCK_SIZE), 1, 1],
                [BLOCK_SIZE, 1, 1],
                0,
                &args,
            )?
        },
    };
    if let Some(timing) = timing {
        let started = Instant::now();
        let result = completion.wait();
        timing.completion_wait += started.elapsed();
        result?;
    } else {
        completion.wait()?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)] // Keep the HIP kernel arguments visible beside its ABI.
fn launch_scalar_update(
    kernel: &HipKernel,
    stream: &HipStreamHandle,
    weights: &DeviceBuffer,
    gradient: &DeviceBuffer,
    learning_rate: f32,
    updated: &DeviceBuffer,
    count: usize,
    mut timing: Option<&mut HipLaunchHostTiming>,
) -> Result<(), Box<dyn Error>> {
    let count_u32 = u32::try_from(count)?;
    let learning_rate_bytes = learning_rate.to_ne_bytes();
    let count_bytes = count_u32.to_ne_bytes();
    let args = [
        HipKernelArgument::Buffer(weights),
        HipKernelArgument::Buffer(gradient),
        HipKernelArgument::Bytes(&learning_rate_bytes),
        HipKernelArgument::Buffer(updated),
        HipKernelArgument::Bytes(&count_bytes),
    ];
    // SAFETY: The kernel takes two f32 pointers, one f32 scalar, one f32 pointer, then a u32
    // length. The buffers are large enough, the tail is guarded, and completion is awaited.
    #[allow(unsafe_code)]
    let mut completion = match timing.as_deref_mut() {
        Some(timing) => {
            let started = Instant::now();
            let completion = unsafe {
                kernel.launch(
                    stream,
                    [count_u32.div_ceil(256), 1, 1],
                    [256, 1, 1],
                    0,
                    &args,
                )
            };
            timing.launch_return += started.elapsed();
            completion?
        }
        None => unsafe {
            kernel.launch(
                stream,
                [count_u32.div_ceil(256), 1, 1],
                [256, 1, 1],
                0,
                &args,
            )?
        },
    };
    if let Some(timing) = timing {
        let started = Instant::now();
        let result = completion.wait();
        timing.completion_wait += started.elapsed();
        result?;
    } else {
        completion.wait()?;
    }
    Ok(())
}

fn allocate(runtime: &HipRuntime, elements: usize) -> Result<DeviceBuffer, Box<dyn Error>> {
    Ok(runtime.allocate(
        elements
            .checked_mul(size_of::<f32>())
            .ok_or("buffer size overflow")?,
    )?)
}

fn upload(runtime: &HipRuntime, values: &[f32]) -> Result<DeviceBuffer, Box<dyn Error>> {
    let mut buffer = allocate(runtime, values.len())?;
    buffer.copy_from(bytemuck::cast_slice(values))?;
    Ok(buffer)
}

fn launch_error(
    kernel: &HipKernel,
    stream: &HipStreamHandle,
    prediction: &DeviceBuffer,
    target: &DeviceBuffer,
    factor: &DeviceBuffer,
    error: &DeviceBuffer,
    count: usize,
) -> Result<(), Box<dyn Error>> {
    launch(kernel, stream, prediction, target, factor, error, count)
}

fn launch_update(
    kernel: &HipKernel,
    stream: &HipStreamHandle,
    weights: &DeviceBuffer,
    gradient: &DeviceBuffer,
    rate: &DeviceBuffer,
    updated: &DeviceBuffer,
    count: usize,
) -> Result<(), Box<dyn Error>> {
    launch(kernel, stream, weights, gradient, rate, updated, count)
}

fn launch(
    kernel: &HipKernel,
    stream: &HipStreamHandle,
    a: &DeviceBuffer,
    b: &DeviceBuffer,
    c: &DeviceBuffer,
    output: &DeviceBuffer,
    count: usize,
) -> Result<(), Box<dyn Error>> {
    let count_u32 = u32::try_from(count)?;
    let count_bytes = count_u32.to_ne_bytes();
    let args = [
        HipKernelArgument::Buffer(a),
        HipKernelArgument::Buffer(b),
        HipKernelArgument::Buffer(c),
        HipKernelArgument::Buffer(output),
        HipKernelArgument::Bytes(&count_bytes),
    ];
    // SAFETY: The kernels take four f32 pointers then a u32 length. All buffers are large enough,
    // the grid-stride tail is guarded, and waiting establishes lifetime-safe completion.
    #[allow(unsafe_code)]
    let mut completion = unsafe {
        kernel.launch(
            stream,
            [count_u32.div_ceil(256), 1, 1],
            [256, 1, 1],
            0,
            &args,
        )?
    };
    completion.wait()?;
    Ok(())
}
