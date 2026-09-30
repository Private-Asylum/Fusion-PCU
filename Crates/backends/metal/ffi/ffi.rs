//! Typed Objective-C ownership and all foreign/shared-memory operations live here.

#[cfg(target_os = "macos")]
mod native {
    #[rustfmt::skip]
    use std::{
        cell::Cell,
        marker::PhantomData,
        mem,
        ptr::{
            self,
            NonNull,
        },
        rc::Rc,
    };
    #[rustfmt::skip]
    use objc2::{
        rc::{
            autoreleasepool,
            Retained,
        },
        runtime::{
            AnyObject,
            NSObjectProtocol,
            ProtocolObject,
        },
        sel,
    };
    #[rustfmt::skip]
    use objc2_foundation::{
        NSError,
        NSString,
    };
    #[rustfmt::skip]
    use objc2_metal::{
        MTLCopyAllDevices,
        MTLBuffer,
        MTLCommandBuffer,
        MTLCommandBufferStatus,
        MTLCommandEncoder,
        MTLCommandQueue,
        MTLCompileOptions,
        MTLComputeCommandEncoder,
        MTLComputePipelineState,
        MTLDevice,
        MTLLibrary,
        MTLResourceOptions,
        MTLSize,
    };
    #[rustfmt::skip]
    use crate::{
        MetalDeviceFacts,
        MetalError,
    };

    fn nil(operation: &str) -> MetalError {
        MetalError::Runtime(format!("{operation} returned nil"))
    }

    fn error(failure: &NSError, operation: &str) -> MetalError {
        MetalError::Runtime(format!("{operation}: {}", failure.localizedDescription()))
    }

    fn facts(device: &ProtocolObject<dyn MTLDevice>) -> Result<MetalDeviceFacts, MetalError> {
        // Generated features declare signatures, not runtime availability. Check physical-fact
        // selectors before messaging older devices/OS versions. sel! uses objc2's typed cache.
        for selector in [
            sel!(registryID),
            sel!(hasUnifiedMemory),
            sel!(maxBufferLength),
        ] {
            if !device.respondsToSelector(selector) {
                return Err(MetalError::Unsupported);
            }
        }
        Ok(MetalDeviceFacts {
            name: device.name().to_string(),
            registry_id: device.registryID(),
            unified_memory: device.hasUnifiedMemory(),
            max_buffer_bytes: device.maxBufferLength() as u64,
        })
    }

    pub fn discover() -> Result<Vec<MetalDeviceFacts>, MetalError> {
        autoreleasepool(|_| {
            MTLCopyAllDevices()
                .iter()
                .map(|device| facts(&device))
                .collect()
        })
    }

    // MTL protocols may implement Send/Sync; the runtime contract remains deliberately
    // thread-confined. Retained owns +1 objects and no autoreleased borrow escapes a pool.
    pub struct Session {
        device: Retained<ProtocolObject<dyn MTLDevice>>,
        queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
        quarantined: Cell<bool>,
        _thread: PhantomData<Rc<()>>,
    }
    pub struct Buffer {
        object: Retained<ProtocolObject<dyn MTLBuffer>>,
        bytes: usize,
        _thread: PhantomData<Rc<()>>,
    }
    pub struct Pipeline {
        object: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
        maximum_threads: usize,
        _thread: PhantomData<Rc<()>>,
    }

    impl Session {
        pub fn ensure_quiescent(&self) -> Result<(), MetalError> {
            if self.quarantined.get() {
                Err(MetalError::Runtime(
                    "Metal session is quarantined after unknown completion".into(),
                ))
            } else {
                Ok(())
            }
        }

        pub fn open(index: usize) -> Result<(Self, MetalDeviceFacts), MetalError> {
            autoreleasepool(|_| {
                let devices = MTLCopyAllDevices();
                let device = devices
                    .iter()
                    .nth(index)
                    .ok_or_else(|| MetalError::Runtime("Metal device index unavailable".into()))?;
                let facts = facts(&device)?;
                let queue = device
                    .newCommandQueue()
                    .ok_or_else(|| nil("Metal command queue"))?;
                Ok((
                    Self {
                        device,
                        queue,
                        quarantined: Cell::new(false),
                        _thread: PhantomData,
                    },
                    facts,
                ))
            })
        }

        pub fn allocate(&self, bytes: usize) -> Result<Buffer, MetalError> {
            self.ensure_quiescent()?;
            autoreleasepool(|_| {
                let object = self
                    .device
                    .newBufferWithLength_options(bytes, MTLResourceOptions::StorageModeShared)
                    .ok_or_else(|| nil("Metal shared allocation"))?;
                if object
                    .contents()
                    .as_ptr()
                    .cast::<u32>()
                    .align_offset(mem::align_of::<u32>())
                    != 0
                {
                    return Err(MetalError::Runtime(
                        "Metal shared contents unavailable".into(),
                    ));
                }
                // Apple documents that newBufferWithLength:options: clears all values to zero.
                // https://developer.apple.com/documentation/metal/mtldevice/makebuffer(length:options:)
                Ok(Buffer {
                    object,
                    bytes,
                    _thread: PhantomData,
                })
            })
        }

        pub fn compile(&self, source: &str, entry: &str) -> Result<Pipeline, MetalError> {
            self.ensure_quiescent()?;
            autoreleasepool(|_| {
                let options = MTLCompileOptions::new();
                #[allow(
                    deprecated,
                    reason = "Preserve admitted fast-math-disabled compilation on the existing deployment range."
                )]
                options.setFastMathEnabled(false);
                let library = self
                    .device
                    .newLibraryWithSource_options_error(&NSString::from_str(source), Some(&options))
                    .map_err(|failure| error(&failure, "Metal compile"))?;
                let function = library
                    .newFunctionWithName(&NSString::from_str(entry))
                    .ok_or_else(|| nil("Metal function"))?;
                let object = self
                    .device
                    .newComputePipelineStateWithFunction_error(&function)
                    .map_err(|failure| error(&failure, "Metal pipeline"))?;
                let maximum_threads = object.maxTotalThreadsPerThreadgroup();
                if maximum_threads == 0 {
                    return Err(MetalError::Runtime(
                        "Metal pipeline reports no thread capacity".into(),
                    ));
                }
                Ok(Pipeline {
                    object,
                    maximum_threads,
                    _thread: PhantomData,
                })
            })
        }

        pub fn execute(
            &self,
            pipeline: &Pipeline,
            buffers: [&Buffer; 4],
            config: [u32; 2],
            count: usize,
        ) -> Result<(), MetalError> {
            self.ensure_quiescent()?;
            autoreleasepool(|_| {
                // Four buffer leases include repeated unary operands, plus device/queue/pipeline.
                // Typed ARC owners survive terminal wait even if the safe outer owners vanish.
                let leases: [Retained<AnyObject>; 7] = [
                    buffers[0].object.clone().into(),
                    buffers[1].object.clone().into(),
                    buffers[2].object.clone().into(),
                    buffers[3].object.clone().into(),
                    self.device.clone().into(),
                    self.queue.clone().into(),
                    pipeline.object.clone().into(),
                ];
                let command = self
                    .queue
                    .commandBuffer()
                    .ok_or_else(|| nil("Metal command buffer"))?;
                let encoder = command
                    .computeCommandEncoder()
                    .ok_or_else(|| nil("Metal encoder"))?;
                encoder.setComputePipelineState(&pipeline.object);
                // SAFETY: admitted MSL uses this four-buffer ABI and validated extents. setBytes
                // copies the complete initialized config now; all +1 owners survive completion.
                unsafe {
                    for (index, buffer) in buffers.iter().enumerate() {
                        encoder.setBuffer_offset_atIndex(Some(&buffer.object), 0, index);
                    }
                    encoder.setBytes_length_atIndex(
                        NonNull::from(&config).cast(),
                        mem::size_of_val(&config),
                        4,
                    );
                }
                encoder.dispatchThreads_threadsPerThreadgroup(
                    MTLSize {
                        width: count,
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: pipeline.maximum_threads.min(256).min(count),
                        height: 1,
                        depth: 1,
                    },
                );
                encoder.endEncoding();
                command.commit();
                command.waitUntilCompleted();
                match command.status() {
                    MTLCommandBufferStatus::Completed => Ok(()),
                    MTLCommandBufferStatus::Error => Err(command.error().map_or_else(
                        || MetalError::Runtime("Metal completion failed without NSError".into()),
                        |failure| error(&failure, "Metal completion"),
                    )),
                    // Unknown completion must keep command+encoder, all four buffers, device,
                    // queue and pipeline alive. No ARC owner is released until quiescence is
                    // proved; this permanent quarantine intentionally retains uncertain work.
                    _ => {
                        self.quarantined.set(true);
                        mem::forget(leases);
                        mem::forget(command);
                        mem::forget(encoder);
                        Err(MetalError::Runtime(
                            "Metal wait returned nonterminal status".into(),
                        ))
                    }
                }
            })
        }
    }

    impl Buffer {
        pub fn write_bytes(&self, offset: usize, bytes: &[u8]) -> Result<(), MetalError> {
            if offset
                .checked_add(bytes.len())
                .is_none_or(|end| end > self.bytes)
            {
                return Err(MetalError::InvalidExtent);
            }
            // SAFETY: safe caller establishes exclusive quiescent storage; checked byte extent,
            // initialized host input and disjoint shared allocation permit this bounded copy.
            unsafe {
                ptr::copy_nonoverlapping(
                    bytes.as_ptr(),
                    self.object.contents().as_ptr().cast::<u8>().add(offset),
                    bytes.len(),
                );
            }
            Ok(())
        }
        pub fn read_bytes(&self, offset: usize, bytes: &mut [u8]) -> Result<(), MetalError> {
            if offset
                .checked_add(bytes.len())
                .is_none_or(|end| end > self.bytes)
            {
                return Err(MetalError::InvalidExtent);
            }
            // SAFETY: safe caller establishes initialized quiescent storage; disjoint host
            // destination and checked extent permit copying after terminal completion.
            unsafe {
                ptr::copy_nonoverlapping(
                    self.object.contents().as_ptr().cast::<u8>().add(offset),
                    bytes.as_mut_ptr(),
                    bytes.len(),
                );
            }
            Ok(())
        }
        /// Resets every private status byte before each submission.
        pub fn fill_ones(&self) {
            // SAFETY: the caller owns this fresh unpublished Shared allocation and retains it;
            // the exact allocation extent is writable and no command has been committed yet.
            unsafe {
                ptr::write_bytes(
                    self.object.contents().as_ptr().cast::<u8>(),
                    0xff,
                    self.bytes,
                );
            };
        }
        /// Inspects initialized status in place after proven terminal completion.
        /// The callback returns only a status result; no Shared reference can escape.
        pub fn inspect_words(
            &self,
            count: usize,
            inspect: impl FnOnce(&[u32]) -> Result<(), MetalError>,
        ) -> Result<(), MetalError> {
            if count.checked_mul(4) != Some(self.bytes) {
                return Err(MetalError::InvalidExtent);
            }
            let contents = self.object.contents().as_ptr().cast::<u32>();
            if contents.align_offset(mem::align_of::<u32>()) != 0 {
                return Err(MetalError::Runtime(
                    "Metal shared contents unavailable".into(),
                ));
            }
            // SAFETY: the safe caller proved terminal completion and keeps this buffer alive;
            // every record was initialized before commit. The aligned exact slice is immutable
            // for this scoped inspection, and the callback cannot return a borrowed reference.
            inspect(unsafe { core::slice::from_raw_parts(contents, count) })
        }
        pub fn write(&self, words: &[u32]) -> Result<(), MetalError> {
            if words.len().checked_mul(4) != Some(self.bytes) {
                return Err(MetalError::InvalidExtent);
            }
            let contents = self.object.contents().as_ptr().cast::<u32>();
            if contents.align_offset(mem::align_of::<u32>()) != 0 {
                return Err(MetalError::Runtime(
                    "Metal shared contents unavailable".into(),
                ));
            }
            // SAFETY: exact aligned extent, initialized host words and exclusive unpublished
            // shared allocation; no active GPU command may access the same storage.
            unsafe {
                ptr::copy_nonoverlapping(words.as_ptr(), contents, words.len());
            }
            Ok(())
        }
        pub fn read(&self, count: usize) -> Result<Vec<u32>, MetalError> {
            if count.checked_mul(4) != Some(self.bytes) {
                return Err(MetalError::InvalidExtent);
            }
            let mut output = vec![0; count];
            let contents = self.object.contents().as_ptr().cast::<u32>();
            if contents.align_offset(mem::align_of::<u32>()) != 0 {
                return Err(MetalError::Runtime(
                    "Metal shared contents unavailable".into(),
                ));
            }
            // SAFETY: exact aligned initialized extent and terminal completion; Vec destination
            // is disjoint and owned on this thread, including the retained buffer lifetime.
            unsafe {
                ptr::copy_nonoverlapping(contents, output.as_mut_ptr(), count);
            }
            Ok(output)
        }
    }
    #[cfg(test)]
    mod tests {
        #[rustfmt::skip]
        use super::{
            MetalError,
            Session,
        };

        #[test]
        #[ignore = "Requires actual macOS Metal compiler/device; verifies NSError ownership and operational retry."]
        fn typed_compile_errors_survive_pool_and_allow_valid_submission_retry() {
            assert!(matches!(
                Session::open(usize::MAX),
                Err(MetalError::Runtime(_))
            ));
            let (session, _) = Session::open(0).unwrap();
            let Err(compiler_error) =
                session.compile("this is invalid Metal source", "migration_copy")
            else {
                panic!("invalid MSL unexpectedly compiled");
            };
            let source = r"#include <metal_stdlib>
using namespace metal;
kernel void migration_copy(device const uint* input [[buffer(0)]],
    device const uint* unused [[buffer(1)]], device uint* output [[buffer(2)]],
    device uint* records [[buffer(3)]], constant uint2& config [[buffer(4)]],
    uint id [[thread_position_in_grid]]) {
    if (id < config.x) { output[id] = input[id]; records[id] = 0u; }
}
";
            let Err(entry_error) = session.compile(source, "missing_migration_entry") else {
                panic!("missing entry unexpectedly resolved");
            };
            let pipeline = session.compile(source, "migration_copy").unwrap();
            let input = session.allocate(16).unwrap();
            // Exercise Apple's native zero-clearing promise before any host initialization.
            input
                .inspect_words(4, |words| {
                    assert_eq!(words, [0; 4]);
                    Ok(())
                })
                .unwrap();
            let output = session.allocate(16).unwrap();
            let records = session.allocate(16).unwrap();
            input.write(&[1, 2, 3, 4]).unwrap();
            records.fill_ones();
            records
                .inspect_words(4, |words| {
                    assert_eq!(words, [u32::MAX; 4]);
                    Ok(())
                })
                .unwrap();
            session
                .execute(&pipeline, [&input, &input, &output, &records], [4, 0], 4)
                .unwrap();
            assert_eq!(output.read(4).unwrap(), [1, 2, 3, 4]);
            assert_eq!(records.read(4).unwrap(), [0; 4]);
            // Both error strings survive every original pool and subsequent compile/dispatch.
            let MetalError::Runtime(description) = compiler_error else {
                panic!("compiler NSError lost Runtime translation");
            };
            assert!(description.starts_with("Metal compile:"), "{description}");
            assert!(description.len() > "Metal compile:".len());
            let MetalError::Runtime(description) = entry_error else {
                panic!("nil function lost Runtime translation");
            };
            assert_eq!(description, "Metal function returned nil");
            session.ensure_quiescent().unwrap();
        }
    }
}
#[cfg(target_os = "macos")]
pub use native::*;

#[cfg(not(target_os = "macos"))]
#[allow(
    clippy::unused_self,
    clippy::missing_const_for_fn,
    reason = "Portable stubs retain the non-const native instance API."
)]
mod unavailable {
    #[rustfmt::skip]
    use crate::{
        MetalDeviceFacts,
        MetalError,
    };

    pub struct Session;
    pub struct Buffer;
    pub struct Pipeline;

    pub fn discover() -> Result<Vec<MetalDeviceFacts>, MetalError> {
        Err(MetalError::Unsupported)
    }

    impl Session {
        pub fn ensure_quiescent(&self) -> Result<(), MetalError> {
            Err(MetalError::Unsupported)
        }
        pub fn open(_: usize) -> Result<(Self, MetalDeviceFacts), MetalError> {
            Err(MetalError::Unsupported)
        }
        pub fn allocate(&self, _: usize) -> Result<Buffer, MetalError> {
            Err(MetalError::Unsupported)
        }
        pub fn compile(&self, _: &str, _: &str) -> Result<Pipeline, MetalError> {
            Err(MetalError::Unsupported)
        }
        pub fn execute(
            &self,
            _: &Pipeline,
            _: [&Buffer; 4],
            _: [u32; 2],
            _: usize,
        ) -> Result<(), MetalError> {
            Err(MetalError::Unsupported)
        }
    }
    impl Buffer {
        pub fn write_bytes(&self, _: usize, _: &[u8]) -> Result<(), MetalError> {
            Err(MetalError::Unsupported)
        }
        pub fn read_bytes(&self, _: usize, _: &mut [u8]) -> Result<(), MetalError> {
            Err(MetalError::Unsupported)
        }
        pub fn fill_ones(&self) {}
        pub fn inspect_words(
            &self,
            _: usize,
            _: impl FnOnce(&[u32]) -> Result<(), MetalError>,
        ) -> Result<(), MetalError> {
            Err(MetalError::Unsupported)
        }
        pub fn write(&self, _: &[u32]) -> Result<(), MetalError> {
            Err(MetalError::Unsupported)
        }
        pub fn read(&self, _: usize) -> Result<Vec<u32>, MetalError> {
            Err(MetalError::Unsupported)
        }
    }
}
#[cfg(not(target_os = "macos"))]
pub use unavailable::*;
