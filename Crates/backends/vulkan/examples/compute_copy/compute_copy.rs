//! Fixed-buffer Vulkan copy proof. Checked arithmetic and modern source-facade integration
//! remain deferred; the SPIR-V lowerer rejects unsupported arithmetic rather than weakening it.

extern crate fusion_pcu_core as fusion_pcu;

use std::error::Error;
use std::fmt;
use std::num::NonZeroU32;

#[rustfmt::skip]
use fusion_pcu_vulkan::{
    PcuVulkanBackend,
    PcuVulkanDispatchReport,
    PcuVulkanError,
};
#[rustfmt::skip]
use fusion_pcu::{
    PcuBindingRef,
    PcuDispatchSubmission,
    PcuError,
    PcuInvocationBinding,
    PcuInvocationBuffer,
    PcuInvocationParameters,
    PcuInvocationShape,
    PcuInvocationTarget,
};
use fusion_pcu_macros::pcu_dispatch;
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::WindowEvent;
#[rustfmt::skip]
use winit::event_loop::{
    ActiveEventLoop,
    ControlFlow,
    EventLoop,
};
#[rustfmt::skip]
use winit::window::{
    Window,
    WindowId,
};

const WINDOW_WIDTH: f64 = 960.0;
const WINDOW_HEIGHT: f64 = 540.0;
const ELEMENT_COUNT: usize = 2048;
const INVOCATIONS: u32 = 250;

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), ExampleError> {
    let event_loop =
        EventLoop::new().map_err(|error| ExampleError::event_loop(error.to_string()))?;
    event_loop.set_control_flow(ControlFlow::Wait);

    let mut app = ExampleApp::new();
    event_loop
        .run_app(&mut app)
        .map_err(|error| ExampleError::event_loop(error.to_string()))?;

    app.finish()
}

struct ExampleApp {
    window: Option<Window>,
    result: Option<Result<ExampleReport, ExampleError>>,
}

impl ExampleApp {
    const fn new() -> Self {
        Self {
            window: None,
            result: None,
        }
    }

    fn finish(self) -> Result<(), ExampleError> {
        let Some(result) = self.result else {
            return Err(ExampleError::event_loop(
                "event loop exited before the Vulkan prototype test ran",
            ));
        };

        let report = result?;
        println!(
            "compute-copy: dispatched {} PCU invocations over {ELEMENT_COUNT} elements through Vulkan on {} ({:?}, groups {:?}, {} SPIR-V words, bound {}, sample output {:.2})",
            report.dispatch.execution.invocations,
            report.dispatch.device_name,
            report.dispatch.execution.resource_model,
            report.dispatch.execution.dispatch_groups,
            report.dispatch.spirv_words,
            report.dispatch.spirv_bound,
            report.sample_output,
        );
        Ok(())
    }

    fn initialize_once(&mut self, event_loop: &ActiveEventLoop) {
        if self.result.is_some() {
            event_loop.exit();
            return;
        }

        let attributes = Window::default_attributes()
            .with_title("Fusion Vulkan Example")
            .with_inner_size(LogicalSize::new(WINDOW_WIDTH, WINDOW_HEIGHT));

        let window = match event_loop.create_window(attributes) {
            Ok(window) => window,
            Err(error) => {
                self.result = Some(Err(ExampleError::window(error.to_string())));
                event_loop.exit();
                return;
            }
        };

        self.window = Some(window);
        self.result = Some(run_pcu_compute_test());
        event_loop.exit();
    }
}

impl ApplicationHandler for ExampleApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        self.initialize_once(event_loop);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        if matches!(event, WindowEvent::CloseRequested) {
            event_loop.exit();
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        if self.result.is_some() {
            event_loop.exit();
        }
    }
}

fn run_pcu_compute_test() -> Result<ExampleReport, ExampleError> {
    let mut source = [0_u32; ELEMENT_COUNT];
    let mut bias = [0_u32; ELEMENT_COUNT];
    let mut output = [0_u32; ELEMENT_COUNT];
    fill_inputs(&mut source, &mut bias)?;

    let bindings = parallel_copy_kernel_bindings();
    let builder = parallel_copy_kernel::<ELEMENT_COUNT>(&bindings).map_err(ExampleError::Pcu)?;
    let backend = PcuVulkanBackend::new().map_err(ExampleError::Vulkan)?;
    let invocations = NonZeroU32::new(INVOCATIONS).ok_or(ExampleError::BufferTooLarge)?;
    let dispatch = builder.with_ir(|kernel| {
        let mut invocation_bindings =
            parallel_float_invocation_bindings(&source, &bias, &mut output);
        backend
            .submit_dispatch(
                PcuDispatchSubmission {
                    kernel,
                    shape: PcuInvocationShape::invocations(invocations),
                },
                &mut invocation_bindings,
                PcuInvocationParameters::empty(),
            )
            .map_err(ExampleError::Vulkan)
    })?;
    verify_parallel_copy_output(&source, &output)?;
    let sample_output = last_output_sample(&output)?;

    Ok(ExampleReport {
        dispatch,
        sample_output,
    })
}

#[pcu_dispatch(kernel_id = 1, invocations = 250)]
fn parallel_copy_kernel<const N: usize>(input_a: &[f32], input_b: &[f32], output: &mut [f32]) {
    let mut id = context.global_invocation_id;
    let stride = context.invocation_count;
    while id < N {
        output[id] = input_a[id];
        id += stride;
    }
}

const fn parallel_float_invocation_bindings<'a>(
    source: &'a [u32; ELEMENT_COUNT],
    bias: &'a [u32; ELEMENT_COUNT],
    output: &'a mut [u32; ELEMENT_COUNT],
) -> [PcuInvocationBinding<'a>; 3] {
    [
        PcuInvocationBinding {
            target: PcuInvocationTarget::Binding(PcuBindingRef::new(0, 0)),
            buffer: PcuInvocationBuffer::WordsIn(source),
        },
        PcuInvocationBinding {
            target: PcuInvocationTarget::Binding(PcuBindingRef::new(0, 1)),
            buffer: PcuInvocationBuffer::WordsIn(bias),
        },
        PcuInvocationBinding {
            target: PcuInvocationTarget::Binding(PcuBindingRef::new(0, 2)),
            buffer: PcuInvocationBuffer::WordsOut(output),
        },
    ]
}

fn fill_inputs(
    source: &mut [u32; ELEMENT_COUNT],
    bias: &mut [u32; ELEMENT_COUNT],
) -> Result<(), ExampleError> {
    for index in 0..ELEMENT_COUNT {
        let value = f32::from(u16::try_from(index).map_err(|_| ExampleError::BufferTooLarge)?);
        source[index] = value.to_bits();
        bias[index] = (value * 0.5).to_bits();
    }
    Ok(())
}

fn verify_parallel_copy_output(
    source: &[u32; ELEMENT_COUNT],
    output: &[u32; ELEMENT_COUNT],
) -> Result<(), ExampleError> {
    for index in 0..ELEMENT_COUNT {
        let source = f32::from_bits(source[index]);
        let expected = source;
        let actual = f32::from_bits(output[index]);
        if actual.to_bits() != expected.to_bits() {
            return Err(ExampleError::ComputeMismatch {
                index,
                expected,
                actual,
            });
        }
    }
    Ok(())
}

fn last_output_sample(output: &[u32; ELEMENT_COUNT]) -> Result<f32, ExampleError> {
    output
        .last()
        .copied()
        .map(f32::from_bits)
        .ok_or(ExampleError::BufferTooLarge)
}

#[derive(Debug, Clone, PartialEq)]
struct ExampleReport {
    dispatch: PcuVulkanDispatchReport,
    sample_output: f32,
}

#[derive(Debug)]
enum ExampleError {
    EventLoop(String),
    Window(String),
    Pcu(PcuError),
    Vulkan(PcuVulkanError),
    BufferTooLarge,
    ComputeMismatch {
        index: usize,
        expected: f32,
        actual: f32,
    },
}

impl ExampleError {
    fn event_loop(message: impl Into<String>) -> Self {
        Self::EventLoop(message.into())
    }

    fn window(message: impl Into<String>) -> Self {
        Self::Window(message.into())
    }
}

impl fmt::Display for ExampleError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EventLoop(message) => write!(formatter, "event loop error: {message}"),
            Self::Window(message) => write!(formatter, "window error: {message}"),
            Self::Pcu(error) => write!(formatter, "pcu error: {error}"),
            Self::Vulkan(error) => write!(formatter, "pcu Vulkan error: {error}"),
            Self::BufferTooLarge => {
                formatter.write_str("buffer size does not fit Vulkan device size")
            }
            Self::ComputeMismatch {
                index,
                expected,
                actual,
            } => write!(
                formatter,
                "GPU compute verification failed at index {index}: expected {expected}, got {actual}"
            ),
        }
    }
}

impl Error for ExampleError {}
