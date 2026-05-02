use std::error::Error;
use std::fmt;

use fusion_pcu::backends::spirv::{
    lower_dispatch_to_spirv,
    PcuSpirvError,
    PcuSpirvFixedSink,
    PcuSpirvLoweringOptions,
    PcuSpirvModuleInfo,
};
use fusion_pcu::model::PcuDispatchKernelBuilder;
use fusion_pcu::runner::vulkan::{
    PcuVulkanError,
    PcuVulkanParallelF32Report,
    PcuVulkanRunner,
};
use fusion_pcu::{
    PcuBinding,
    PcuBindingAccess,
    PcuBindingStorageClass,
    PcuDispatchAluOp,
    PcuDispatchControlOp,
    PcuDispatchResourceOp,
    PcuDispatchValueOp,
    PcuError,
    PcuValueType,
};
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::WindowEvent;
use winit::event_loop::{
    ActiveEventLoop,
    ControlFlow,
    EventLoop,
};
use winit::window::{
    Window,
    WindowId,
};

const WINDOW_WIDTH: f64 = 960.0;
const WINDOW_HEIGHT: f64 = 540.0;
const ELEMENT_COUNT: usize = 256;
const LOCAL_SIZE_X: u32 = 64;
const SPIRV_WORD_CAPACITY: usize = 512;

type SpirvBuild = (PcuSpirvFixedSink<SPIRV_WORD_CAPACITY>, PcuSpirvModuleInfo);

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
                "event loop exited before the Vulkan runner test ran",
            ));
        };

        let report = result?;
        println!(
            "fusion-vulkan-example: dispatched {} parallel f32 lanes through PCU -> SPIR-V -> Vulkan on {} (queue family {}, {:?}, {} SPIR-V words, bound {}, sample output {:.2})",
            report.vulkan.element_count,
            report.device_name,
            report.queue_family_index,
            report.vulkan.resource_model,
            report.spirv_words,
            report.spirv_bound,
            report.vulkan.sample_output,
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
        self.result = Some(run_vulkan_runner_test());
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

fn run_vulkan_runner_test() -> Result<ExampleReport, ExampleError> {
    let (spirv, info) = compile_parallel_float_pcu_spirv()?;
    let mut source = [0.0_f32; ELEMENT_COUNT];
    let mut bias = [0.0_f32; ELEMENT_COUNT];
    let mut output = [0.0_f32; ELEMENT_COUNT];
    fill_inputs(&mut source, &mut bias)?;

    let runner = PcuVulkanRunner::new().map_err(ExampleError::Vulkan)?;
    let device_name = runner.physical_device_name().to_owned();
    let queue_family_index = runner.queue_family_index();
    let vulkan = runner
        .run_parallel_f32_spirv(spirv.as_slice(), LOCAL_SIZE_X, &source, &bias, &mut output)
        .map_err(ExampleError::Vulkan)?;
    verify_parallel_float_output(&source, &bias, &output)?;

    Ok(ExampleReport {
        device_name,
        queue_family_index,
        spirv_words: info.word_count,
        spirv_bound: info.bound,
        vulkan,
    })
}

fn compile_parallel_float_pcu_spirv() -> Result<SpirvBuild, ExampleError> {
    let bindings = [
        PcuBinding::value(
            Some("input_a"),
            0,
            0,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::ReadOnly,
            PcuValueType::f32(),
        ),
        PcuBinding::value(
            Some("input_b"),
            0,
            1,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::ReadOnly,
            PcuValueType::f32(),
        ),
        PcuBinding::value(
            Some("output"),
            0,
            2,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::WriteOnly,
            PcuValueType::f32(),
        ),
    ];
    let builder = PcuDispatchKernelBuilder::<8>::new(1, "main", [LOCAL_SIZE_X, 1, 1])
        .with_bindings(&bindings)
        .with_resource_op(PcuDispatchResourceOp::Load)
        .map_err(ExampleError::Pcu)?
        .with_resource_op(PcuDispatchResourceOp::Load)
        .map_err(ExampleError::Pcu)?
        .with_arithmetic_op(PcuDispatchAluOp::Mul)
        .map_err(ExampleError::Pcu)?
        .with_arithmetic_op(PcuDispatchAluOp::Add)
        .map_err(ExampleError::Pcu)?
        .with_value_op(PcuDispatchValueOp::Constant)
        .map_err(ExampleError::Pcu)?
        .with_arithmetic_op(PcuDispatchAluOp::Add)
        .map_err(ExampleError::Pcu)?
        .with_resource_op(PcuDispatchResourceOp::Store)
        .map_err(ExampleError::Pcu)?
        .with_control_op(PcuDispatchControlOp::Return)
        .map_err(ExampleError::Pcu)?;
    let kernel = builder.ir();
    let mut sink = PcuSpirvFixedSink::<SPIRV_WORD_CAPACITY>::new();
    let info = lower_dispatch_to_spirv(
        &kernel,
        PcuSpirvLoweringOptions::minimal_shader(),
        &mut sink,
    )
    .map_err(ExampleError::Spirv)?;
    Ok((sink, info))
}

fn fill_inputs(
    source: &mut [f32; ELEMENT_COUNT],
    bias: &mut [f32; ELEMENT_COUNT],
) -> Result<(), ExampleError> {
    for index in 0..ELEMENT_COUNT {
        let value = f32::from(u16::try_from(index).map_err(|_| ExampleError::BufferTooLarge)?);
        source[index] = value;
        bias[index] = value * 0.5;
    }
    Ok(())
}

fn verify_parallel_float_output(
    source: &[f32; ELEMENT_COUNT],
    bias: &[f32; ELEMENT_COUNT],
    output: &[f32; ELEMENT_COUNT],
) -> Result<(), ExampleError> {
    for index in 0..ELEMENT_COUNT {
        let expected = source[index].mul_add(2.0, bias[index]) + 1.0;
        let actual = output[index];
        if (actual - expected).abs() > f32::EPSILON {
            return Err(ExampleError::ComputeMismatch {
                index,
                expected,
                actual,
            });
        }
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq)]
struct ExampleReport {
    device_name: String,
    queue_family_index: u32,
    spirv_words: usize,
    spirv_bound: u32,
    vulkan: PcuVulkanParallelF32Report,
}

#[derive(Debug)]
enum ExampleError {
    EventLoop(String),
    Window(String),
    Pcu(PcuError),
    Spirv(PcuSpirvError),
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
            Self::Spirv(error) => write!(formatter, "spir-v lowering error: {error}"),
            Self::Vulkan(error) => write!(formatter, "vulkan runner error: {error}"),
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
