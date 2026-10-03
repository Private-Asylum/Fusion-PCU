//! Independent compiler/ash pointwise owner control. No PCU lowering or native executor calls.
#[path = "compile/compile.rs"]
mod compile;
#[rustfmt::skip]
use pcu_facade::{
    PcuExecutionFault,
    PcuExecutionFaultKind,
    PcuScalarType,
    PcuStableDeviceIdentity,
};
#[rustfmt::skip]
use super::{
    vk,
    NativeCopy,
    Owner,
    Result,
    Rc,
};

pub struct NativePointwise {
    copy: NativeCopy,
    right: Owner,
    output: Owner,
    status: Owner,
    shader: vk::ShaderModule,
    descriptor_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    descriptor_pool: vk::DescriptorPool,
    descriptors: vk::DescriptorSet,
    pool: vk::CommandPool,
    command: vk::CommandBuffer,
    fence: vk::Fence,
    extent: usize,
    groups: u32,
    unary: bool,
    status_stride: usize,
}

#[derive(Clone, Copy)]
pub struct NativeGeometry {
    pub input_bytes: [usize; 2],
    pub output_bytes: usize,
    pub status_stride: usize,
    pub groups: u32,
    pub unary: bool,
}

impl NativePointwise {
    pub fn new(
        identity: PcuStableDeviceIdentity,
        scalar: PcuScalarType,
        extent: u32,
        operation: u32,
        underflow: u32,
    ) -> Result<Self> {
        let words = compile::shader(scalar, extent, operation, underflow)?;
        let bytes = usize::from(scalar.bit_width() / 8)
            .checked_mul(extent as usize)
            .ok_or("native extent overflow")?;
        let lanes = match scalar.bit_width() {
            8 => 4,
            16 => 2,
            _ => 1,
        };
        Self::compiled(
            identity,
            &words,
            extent,
            NativeGeometry {
                input_bytes: [bytes; 2],
                output_bytes: bytes,
                status_stride: 1,
                groups: extent.div_ceil(lanes).div_ceil(64),
                unary: operation == 4,
            },
        )
    }

    /// Independent fixed-shape full forward/loss/reverse/update control.
    pub fn training(
        identity: PcuStableDeviceIdentity,
        scalar: PcuScalarType,
        policy: u32,
    ) -> Result<Self> {
        let words = compile::training(scalar, policy)?;
        let size = usize::from(scalar.bit_width() / 8);
        Self::compiled(
            identity,
            &words,
            2,
            NativeGeometry {
                input_bytes: [6 * size, 2 * size],
                output_bytes: 2 * size,
                status_stride: 1,
                groups: 1,
                unary: false,
            },
        )
    }

    pub fn compound(
        identity: PcuStableDeviceIdentity,
        scalar: PcuScalarType,
        constants: [u32; 9],
        counts: [u32; 3],
    ) -> Result<Self> {
        let words = compile::compound(scalar, constants)?;
        let size = usize::from(scalar.bit_width() / 8);
        let bytes = counts.map(|count| {
            (count as usize)
                .checked_mul(size)
                .ok_or("native extent overflow")
        });
        let [left, right, output] = bytes;
        Self::compiled(
            identity,
            &words,
            counts[2],
            NativeGeometry {
                input_bytes: [left?, right?],
                output_bytes: output?,
                status_stride: 3,
                groups: counts[2].div_ceil(64),
                unary: false,
            },
        )
    }

    #[allow(clippy::too_many_lines)] // Independent cold shader/native owner construction, outside measured work.
    fn compiled(
        identity: PcuStableDeviceIdentity,
        words: &[u32],
        extent: u32,
        geometry: NativeGeometry,
    ) -> Result<Self> {
        let copy = NativeCopy::new(identity, geometry.input_bytes[0])?;
        let context = &copy.context;
        let right = Owner::new(context, geometry.input_bytes[1])?;
        let output = Owner::new(context, geometry.output_bytes)?;
        let status = Owner::new(
            context,
            (extent as usize)
                .checked_mul(geometry.status_stride * 4)
                .ok_or("status overflow")?,
        )?;
        let unary = geometry.unary;
        let count = if unary { 3 } else { 4 };
        let device = &context.device;
        // SAFETY: Independent compiler produced valid words; context owns this logical device.
        let shader = unsafe {
            device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(words), None)?
        };
        let bindings: Vec<_> = (0..count)
            .map(|index| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(index)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
            })
            .collect();
        // SAFETY: Immutable stack/Vec create infos remain live through synchronous cold calls.
        let descriptor_layout = unsafe {
            device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                None,
            )?
        };
        // SAFETY: Descriptor layout belongs to this live device and is retained below.
        let pipeline_layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default().set_layouts(&[descriptor_layout]),
                None,
            )?
        };
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader)
            .name(c"main");
        // SAFETY: Compatible live shader/layout, no specialization pointers or native features.
        let pipeline = unsafe {
            device.create_compute_pipelines(
                vk::PipelineCache::null(),
                &[vk::ComputePipelineCreateInfo::default()
                    .stage(stage)
                    .layout(pipeline_layout)],
                None,
            )
        }
        .map_err(|(_, error)| error)?[0];
        // SAFETY: Pool sizes cover the one fixed set, whose descriptors are retained.
        let descriptor_pool = unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(1)
                    .pool_sizes(&[vk::DescriptorPoolSize::default()
                        .ty(vk::DescriptorType::STORAGE_BUFFER)
                        .descriptor_count(count)]),
                None,
            )?
        };
        // SAFETY: Pool and layout belong to the same context and are live.
        let descriptors = unsafe {
            device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(descriptor_pool)
                    .set_layouts(&[descriptor_layout]),
            )?
        }[0];
        // SAFETY: Copy command pool was created for this compute queue; query its family via
        // the retained independent context's cold copy preparation helper below.
        let pool = unsafe {
            device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(context.family)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )?
        };
        // SAFETY: Pool belongs to the context, primary command allocation count is one.
        let command = unsafe {
            device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )?
        }[0];
        // SAFETY: Unsigned fence belongs to this live device and has no extension pointers.
        let fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None)? };
        Ok(Self {
            copy,
            right,
            output,
            status,
            shader,
            descriptor_layout,
            pipeline_layout,
            pipeline,
            descriptor_pool,
            descriptors,
            pool,
            command,
            fence,
            extent: extent as usize,
            groups: geometry.groups,
            unary,
            status_stride: geometry.status_stride,
        })
    }

    pub fn upload(&self, input: &[u8]) -> Result<Owner> {
        let mut owner = Owner::new(&self.copy.context, input.len())?;
        owner.write(input);
        Ok(owner)
    }

    pub fn host(
        &mut self,
        left: &[u8],
        right: &[u8],
    ) -> std::result::Result<Owner, PcuExecutionFault> {
        self.copy.upload.write(left);
        if !self.unary {
            self.right.write(right);
        }
        self.execute(None)
    }

    #[allow(clippy::needless_pass_by_ref_mut)] // Exclusive native command/descriptor mutation is required despite immutable Rust handle fields.
    pub fn borrowed(
        &mut self,
        left: &Owner,
        right: &Owner,
    ) -> std::result::Result<Owner, PcuExecutionFault> {
        assert!(Rc::ptr_eq(&left.context, &self.copy.context));
        if !self.unary {
            assert!(Rc::ptr_eq(&right.context, &self.copy.context));
        }
        self.execute(Some([left, right]))
    }

    /// Host samples/targets with the preceding escaped native weights, without readback.
    pub fn mixed(
        &mut self,
        left: &[u8],
        right: &Owner,
    ) -> std::result::Result<Owner, PcuExecutionFault> {
        assert!(Rc::ptr_eq(&right.context, &self.copy.context));
        self.copy.upload.write(left);
        self.execute(Some([&self.copy.upload, right]))
    }

    #[allow(clippy::too_many_lines)] // Native binding, terminal synchronization and complete diagnostic law remain adjacent.
    fn execute(
        &self,
        inputs: Option<[&Owner; 2]>,
    ) -> std::result::Result<Owner, PcuExecutionFault> {
        let values = inputs.unwrap_or([&self.copy.upload, &self.right]);
        assert_eq!(values[0].bytes, self.copy.upload.bytes);
        if !self.unary {
            assert_eq!(values[1].bytes, self.right.bytes);
        }
        let buffer = |owner: &Owner| {
            [vk::DescriptorBufferInfo::default()
                .buffer(owner.buffer)
                .range(vk::WHOLE_SIZE)]
        };
        let infos = [
            buffer(values[0]),
            buffer(values[1]),
            buffer(&self.output),
            buffer(&self.status),
        ];
        let count = if self.unary { 3 } else { 4 };
        let writes: [_; 4] = core::array::from_fn(|index| {
            vk::WriteDescriptorSet::default()
                .dst_set(self.descriptors)
                .dst_binding(u32::try_from(index).unwrap())
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(
                    &infos[if self.unary && index != 0 {
                        index + 1
                    } else {
                        index
                    }
                    .min(3)],
                )
        });
        let device = &self.copy.context.device;
        // SAFETY: All prior calls terminal; fixed native owners and borrowed exact-session
        // inputs stay retained through terminal wait. Reset precedes descriptor updates.
        unsafe {
            device
                .reset_command_buffer(self.command, vk::CommandBufferResetFlags::empty())
                .unwrap();
            device.update_descriptor_sets(&writes[..count], &[]);
            device.reset_fences(&[self.fence]).unwrap();
            device
                .begin_command_buffer(self.command, &vk::CommandBufferBeginInfo::default())
                .unwrap();
            let before = [vk::MemoryBarrier::default()
                .src_access_mask(
                    vk::AccessFlags::HOST_WRITE
                        | vk::AccessFlags::TRANSFER_WRITE
                        | vk::AccessFlags::SHADER_WRITE,
                )
                .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)];
            device.cmd_pipeline_barrier(
                self.command,
                vk::PipelineStageFlags::HOST
                    | vk::PipelineStageFlags::TRANSFER
                    | vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &before,
                &[],
                &[],
            );
            device.cmd_bind_pipeline(self.command, vk::PipelineBindPoint::COMPUTE, self.pipeline);
            device.cmd_bind_descriptor_sets(
                self.command,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                &[self.descriptors],
                &[],
            );
            device.cmd_dispatch(self.command, self.groups, 1, 1);
            let after = [vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::HOST_READ)];
            device.cmd_pipeline_barrier(
                self.command,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &after,
                &[],
                &[],
            );
            device.end_command_buffer(self.command).unwrap();
            let completion = device
                .queue_submit(
                    self.copy.context.queue,
                    &[vk::SubmitInfo::default().command_buffers(&[self.command])],
                    self.fence,
                )
                .and_then(|()| device.wait_for_fences(&[self.fence], true, u64::MAX));
            if let Err(error) = completion {
                self.copy.context.poisoned.set(true);
                std::mem::forget(Rc::clone(&self.copy.context));
                panic!("native pointwise uncertain completion: {error:?}");
            }
        }
        for index in 0..self.extent {
            // SAFETY: Shader initializes every logical status; terminal compute-to-host
            // barrier and wait precede this mapped U32 read within the full status allocation.
            let status = unsafe {
                self.status
                    .mapped
                    .add(index * self.status_stride * 4)
                    .cast::<u32>()
                    .read_unaligned()
            };
            let kind = match status {
                0 => continue,
                1 => PcuExecutionFaultKind::InvalidFloatingOperand,
                2 => PcuExecutionFaultKind::ArithmeticUnderflow,
                3 => PcuExecutionFaultKind::ArithmeticOverflow,
                4 => PcuExecutionFaultKind::DivideByZero,
                _ => panic!("invalid native pointwise status"),
            };
            return Err(PcuExecutionFault {
                kind,
                invocation_id: index as u64,
                recovered: false,
            });
        }
        Ok(self.copy.copy_owned(&self.output).unwrap())
    }
}

impl Drop for NativePointwise {
    fn drop(&mut self) {
        if self.copy.context.poisoned.get() {
            return;
        }
        // SAFETY: Calls are synchronous/terminal; child resources release before retained Context.
        unsafe {
            let device = &self.copy.context.device;
            device.destroy_fence(self.fence, None);
            device.destroy_command_pool(self.pool, None);
            device.destroy_descriptor_pool(self.descriptor_pool, None);
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            device.destroy_descriptor_set_layout(self.descriptor_layout, None);
            device.destroy_shader_module(self.shader, None);
        }
    }
}
