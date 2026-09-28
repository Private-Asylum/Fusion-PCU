//! Conservative initialization facts for admitted dense scalar maps.

#[rustfmt::skip]
use super::{
    PcuBindingRef,
    PcuDispatchControlOp,
    PcuDispatchDataOp,
    PcuDispatchIndex,
    PcuDispatchKernelIr,
    PcuDispatchOp,
};

impl PcuDispatchKernelIr<'_> {
    /// Proves the initialized prefix of a binding after successful dense-map completion.
    ///
    /// This cold structural analysis recognizes only straight-line data instructions followed
    /// by an optional terminal return, or one grid-stride map with that same body shape. Every
    /// successful invocation must store its own logical element, and no instruction may read
    /// this binding. Unknown control flow and opaque effects return `None`.
    ///
    /// The caller must separately admit the kernel's SSA/type rules, launch geometry, resource
    /// access and physical aliasing. This fact does not authorize aliasing, storage donation,
    /// publication after failed completion, or reading an uninitialized allocation. In
    /// particular, elements beyond the returned prefix retain their original semantics: a
    /// larger host output still needs its untouched tail preserved.
    #[must_use]
    pub fn fully_written_binding_elements(
        &self,
        target: PcuBindingRef,
        submitted_invocations: u32,
    ) -> Option<u32> {
        if submitted_invocations == 0 {
            return None;
        }
        let ops = without_terminal_return(self.ops);
        let (body, index, extent) = match ops {
            [PcuDispatchOp::GridStrideLoop { extent, body }] if *extent != 0 => {
                (*body, PcuDispatchIndex::GridStrideId, *extent)
            }
            _ => (ops, PcuDispatchIndex::InvocationId, submitted_invocations),
        };
        let mut writes = false;
        for op in without_terminal_return(body) {
            let PcuDispatchOp::Data(data) = op else {
                return None;
            };
            match data {
                PcuDispatchDataOp::BindingLoad { binding, .. } if *binding == target => {
                    return None;
                }
                PcuDispatchDataOp::BindingStore {
                    binding,
                    index: store_index,
                    ..
                } if *binding == target => {
                    if *store_index != index {
                        return None;
                    }
                    writes = true;
                }
                _ => {}
            }
        }
        writes.then_some(extent)
    }
}

const fn without_terminal_return<'a, 'ir>(
    ops: &'a [PcuDispatchOp<'ir>],
) -> &'a [PcuDispatchOp<'ir>] {
    match ops {
        [
            body @ ..,
            PcuDispatchOp::Control(PcuDispatchControlOp::Return),
        ] => body,
        _ => ops,
    }
}

#[cfg(test)]
mod tests {
    #[rustfmt::skip]
    use super::{
        PcuBindingRef,
        PcuDispatchControlOp,
        PcuDispatchDataOp,
        PcuDispatchIndex,
        PcuDispatchOp,
    };
    #[rustfmt::skip]
    use crate::model::{
        PcuDispatchKernelBuilder,
        PcuDispatchValueId,
    };

    const OUTPUT: PcuBindingRef = PcuBindingRef::new(0, 1);

    fn store(index: PcuDispatchIndex) -> PcuDispatchOp<'static> {
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
            binding: OUTPUT,
            index,
            value: PcuDispatchValueId(0),
        })
    }

    fn coverage(ops: &[PcuDispatchOp<'_>], width: u32) -> Option<u32> {
        let builder = PcuDispatchKernelBuilder::<1>::new(1, "effects", [width, 1, 1]);
        let ir = super::PcuDispatchKernelIr {
            ops,
            ..builder.ir()
        };
        ir.fully_written_binding_elements(OUTPUT, width)
    }

    #[test]
    fn direct_coverage_is_only_the_submitted_prefix() {
        let ops = [store(PcuDispatchIndex::InvocationId)];
        assert_eq!(coverage(&ops, 65), Some(65));
        assert_eq!(coverage(&ops, 0), None);
        assert_eq!(coverage(&[], 65), None);
    }

    #[test]
    fn scalar_and_unknown_indices_do_not_prove_dense_initialization() {
        assert_eq!(
            coverage(&[store(PcuDispatchIndex::BindingElementZero)], 65),
            None
        );
        assert_eq!(
            coverage(&[store(PcuDispatchIndex::Value(PcuDispatchValueId(1)))], 65),
            None
        );
    }

    #[test]
    fn reads_require_original_contents_even_after_a_dense_store() {
        let read = PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
            result: PcuDispatchValueId(1),
            binding: OUTPUT,
            index: PcuDispatchIndex::InvocationId,
        });
        assert_eq!(
            coverage(&[read, store(PcuDispatchIndex::InvocationId)], 65),
            None
        );
        assert_eq!(
            coverage(&[store(PcuDispatchIndex::InvocationId), read], 65),
            None
        );
    }

    #[test]
    fn early_return_branch_and_opaque_effects_are_unknown() {
        let write = store(PcuDispatchIndex::InvocationId);
        let terminal = PcuDispatchOp::Control(PcuDispatchControlOp::Return);
        assert_eq!(coverage(&[write, terminal], 65), Some(65));
        assert_eq!(coverage(&[terminal, write], 65), None);
        assert_eq!(
            coverage(
                &[PcuDispatchOp::Control(PcuDispatchControlOp::Branch), write],
                65
            ),
            None
        );
        assert_eq!(
            coverage(&[PcuDispatchOp::Intrinsic { name: "opaque" }, write], 65),
            None
        );
    }

    #[test]
    fn grid_stride_coverage_uses_logical_extent_not_launch_width() {
        let body = [store(PcuDispatchIndex::GridStrideId)];
        let ops = [
            PcuDispatchOp::GridStrideLoop {
                extent: 1_024,
                body: &body,
            },
            PcuDispatchOp::Control(PcuDispatchControlOp::Return),
        ];
        assert_eq!(coverage(&ops, 250), Some(1_024));
        let invalid_body = [store(PcuDispatchIndex::InvocationId)];
        assert_eq!(
            coverage(
                &[PcuDispatchOp::GridStrideLoop {
                    extent: 1_024,
                    body: &invalid_body
                }],
                250
            ),
            None
        );
        assert_eq!(
            coverage(
                &[PcuDispatchOp::GridStrideLoop {
                    extent: 0,
                    body: &body
                }],
                250
            ),
            None
        );
    }
}
