//! Actual fourteen-width operand maps preserve logical roles, status and private publication.
#[path = "../wide_integer/oracle/oracle.rs"]
#[allow(dead_code)]
// This fixture needs the independent base-256 arithmetic and canonical carriers.
mod oracle;
#[path = "../../benches/integer_operands/source/source.rs"]
mod source;
#[rustfmt::skip]
use pcu_facade::{
    global,
    PcuBindingRef,
    PcuHostArgument,
    PcuHostKernelBackend,
    PcuPreparedHostKernel,
    PcuDispatchIntegerBinaryOp,
    PcuRangePolicy,
    PcuExecutionFault,
    PcuExecutionFaultKind,
    PcuI256,
    PcuU256,
    PcuI512,
    PcuU512,
};
#[rustfmt::skip]
use fusion_pcu_cpu::{
    PcuCpuHostBackend,
    PcuCpuPreparedHost,
};
use oracle::Wide;
const N: usize = 65;
const fn portable() -> bool {
    true
}
const fn numerical(ir: &mut pcu_facade::PcuDispatchKernelIr<'_>) {
    if portable() {
        ir.numerical_requirements.numerical_options.reproducibility =
            pcu_facade::PcuReproducibility::PortableV1;
    }
}
fn evaluate<T: Wide>(
    kind: usize,
    a: &[T],
    b: &[T],
    lane: usize,
) -> Result<T, PcuExecutionFaultKind> {
    let (op, left, right) = match kind {
        0 => (PcuDispatchIntegerBinaryOp::Add, a[lane], a[lane]),
        1 => (PcuDispatchIntegerBinaryOp::Mul, a[lane], a[lane]),
        2 => (PcuDispatchIntegerBinaryOp::Sub, a[lane], b[lane]),
        3 => (PcuDispatchIntegerBinaryOp::Sub, a[lane], a[0]),
        _ => (PcuDispatchIntegerBinaryOp::Sub, a[0], a[lane]),
    };
    oracle::evaluate(left, right, op)
}
fn explicit<T: Wide>(
    kind: usize,
    plan: &mut PcuCpuPreparedHost,
    a: &[T],
    b: &[T],
    out: &mut [T],
) -> Result<(), PcuExecutionFault> {
    let t = |slot| PcuBindingRef::new(0, slot);
    let result = match kind {
        0 => plan.call(&mut [
            PcuHostArgument::read_write(t(0), out),
            PcuHostArgument::read(t(1), a),
        ]),
        1 | 4 => plan.call(&mut [
            PcuHostArgument::read(t(0), &[] as &[T]),
            PcuHostArgument::read_write(t(1), out),
            PcuHostArgument::read(t(2), a),
        ]),
        2 => plan.call(&mut [
            PcuHostArgument::read(t(0), b),
            PcuHostArgument::read_write(t(1), out),
            PcuHostArgument::read(t(2), a),
        ]),
        _ => plan.call(&mut [
            PcuHostArgument::read(t(0), a),
            PcuHostArgument::read_write(t(1), out),
        ]),
    };
    result.map_err(|error| match error {
        fusion_pcu_cpu::PcuCpuHostError::Integer(
            fusion_pcu_cpu::PcuCpuCheckedIntegerError::Fault(f),
        ) => f,
        e => panic!("unexpected native error {e:?}"),
    })
}
trait BuildPlan {
    fn prepare(&self, b: &PcuCpuHostBackend) -> PcuCpuPreparedHost;
}
impl<const M: usize> BuildPlan for fusion_pcu_core::model::PcuDispatchKernelBuilder<'_, M> {
    fn prepare(&self, b: &PcuCpuHostBackend) -> PcuCpuPreparedHost {
        {
            let mut ir = self.ir();
            numerical(&mut ir);
            b.prepare_host_kernel(&ir).unwrap()
        }
    }
}
impl<const M: usize> BuildPlan for fusion_pcu_core::model::PcuGridStrideKernelBuilder<'_, M> {
    fn prepare(&self, b: &PcuCpuHostBackend) -> PcuCpuPreparedHost {
        self.with_ir(|k| {
            let mut ir = *k;
            numerical(&mut ir);
            b.prepare_host_kernel(&ir).unwrap()
        })
    }
}
fn compare<T: Wide>(
    kind: usize,
    range: PcuRangePolicy,
    mut plan: PcuCpuPreparedHost,
    mut ordinary: impl FnMut(&[T], &[T], &mut [T]) -> Result<(), PcuExecutionFault>,
) {
    let mut state = 0xcca1_041b_c777_55a1_u64;
    let mut a: [T; N] = std::array::from_fn(|_| {
        let mut bytes = [0; 64];
        for byte in &mut bytes[..T::BYTES] {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            *byte = state.to_le_bytes()[0];
        }
        T::from_bytes(bytes)
    });
    let b = std::array::from_fn::<_, N, _>(|lane| {
        if lane % 2 == 0 {
            oracle::maximum::<T>()
        } else {
            oracle::minimum::<T>()
        }
    });
    let sentinel = oracle::small::<T>(77);
    for bank in 0..3 {
        if bank == 1 {
            a.fill(oracle::small(3));
        } else if bank == 2 {
            a[2] = oracle::maximum();
            a[3] = oracle::minimum();
        }
        let mut want = [sentinel; N + 3];
        let mut notice = None;
        for (lane, value) in want.iter_mut().take(N).enumerate() {
            match evaluate(kind, &a, &b, lane) {
                Ok(result) => *value = result,
                Err(fault) => {
                    notice.get_or_insert_with(|| PcuExecutionFault {
                        kind: fault,
                        recovered: range == PcuRangePolicy::Clamp,
                        invocation_id: u64::try_from(lane).unwrap(),
                    });
                    *value = if fault == PcuExecutionFaultKind::ArithmeticUnderflow {
                        oracle::minimum()
                    } else {
                        oracle::maximum()
                    };
                }
            }
        }
        if range == PcuRangePolicy::Reject && notice.is_some() {
            want.fill(sentinel);
        }
        for route in 0..2 {
            let mut out = [sentinel; N + 3];
            let got = match route {
                0 => ordinary(&a, &b, &mut out),
                _ => explicit(kind, &mut plan, &a, &b, &mut out),
            };
            assert_eq!(got, notice.map_or(Ok(()), Err));
            assert_eq!(out, want);
        }
    }
}
fn width<T: Wide>(backend: &PcuCpuHostBackend) {
    macro_rules! case {
        ($name:ident,$bindings:ident,$ir:ident,$kind:expr,$range:ident,$call:expr) => {{
            let bindings = source::$bindings::<T>();
            let ir = source::$ir::<T, N>(&bindings).unwrap();
            compare::<T>($kind, PcuRangePolicy::$range, ir.prepare(backend), $call);
        }};
    }
    case!(
        doubled,
        doubled_bindings,
        doubled_ir,
        0,
        Reject,
        |a, _b, o| source::doubled::<T, N>(o, a).map_err(|e| e.arithmetic_fault().unwrap())
    );
    case!(
        squared,
        squared_bindings,
        squared_ir,
        1,
        Reject,
        |a, _b, o| source::squared::<T, N>(&[], o, a).map_err(|e| e.arithmetic_fault().unwrap())
    );
    case!(
        reordered,
        reordered_bindings,
        reordered_ir,
        2,
        Reject,
        |a, b, o| source::reordered::<T, N>(b, o, a).map_err(|e| e.arithmetic_fault().unwrap())
    );
    case!(
        independent,
        independent_bindings,
        independent_ir,
        3,
        Reject,
        |a, _b, o| source::independent::<T, N>(a, o).map_err(|e| e.arithmetic_fault().unwrap())
    );
    case!(grid, grid_bindings, grid_ir, 4, Reject, |a, _b, o| {
        source::grid::<T, N>(&[], o, a).map_err(|e| e.arithmetic_fault().unwrap())
    });
    case!(
        doubled_clamp,
        doubled_clamp_bindings,
        doubled_clamp_ir,
        0,
        Clamp,
        |a, _b, o| source::doubled_clamp::<T, N>(o, a).map_err(|e| e.arithmetic_fault().unwrap())
    );
    case!(
        squared_clamp,
        squared_clamp_bindings,
        squared_clamp_ir,
        1,
        Clamp,
        |a, _b, o| source::squared_clamp::<T, N>(&[], o, a)
            .map_err(|e| e.arithmetic_fault().unwrap())
    );
    case!(
        reordered_clamp,
        reordered_clamp_bindings,
        reordered_clamp_ir,
        2,
        Clamp,
        |a, b, o| source::reordered_clamp::<T, N>(b, o, a)
            .map_err(|e| e.arithmetic_fault().unwrap())
    );
    case!(
        independent_clamp,
        independent_clamp_bindings,
        independent_clamp_ir,
        3,
        Clamp,
        |a, _b, o| source::independent_clamp::<T, N>(a, o)
            .map_err(|e| e.arithmetic_fault().unwrap())
    );
    case!(
        grid_clamp,
        grid_clamp_bindings,
        grid_clamp_ir,
        4,
        Clamp,
        |a, _b, o| source::grid_clamp::<T, N>(&[], o, a).map_err(|e| e.arithmetic_fault().unwrap())
    );
}
#[test]
fn fourteen_width_portable_full_bits_source_prepared_roles() {
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Cpu,
        numerical_options: pcu_facade::PcuNumericalOptions {
            reproducibility: if portable() {
                pcu_facade::PcuReproducibility::PortableV1
            } else {
                pcu_facade::PcuReproducibility::Unspecified
            },
            ..Default::default()
        },
        ..Default::default()
    })
    .unwrap();
    let backend = PcuCpuHostBackend::detect();
    macro_rules! widths {($($t:ty),+)=>{$(width::<$t>(&backend);)+};}
    widths!(
        i8, u8, i16, u16, i32, u32, i64, u64, i128, u128, PcuI256, PcuU256, PcuI512, PcuU512
    );
    global::clear_thread_cache().unwrap();
    global::use_defaults().unwrap();
}

#[pcu_facade::pcu(invocations = 7, crate_path = ::pcu_facade, flag(deterministic), flag(clamp_range))]
fn local_portable<T: pcu_facade::PcuCheckedInteger>(output: &mut [T], input: &[T]) {
    let id = context.global_invocation_id;
    output[id] = input[id] + input[id];
}
#[test]
fn local_portable_clamp_overrides_unspecified_policy() {
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Cpu,
        ..Default::default()
    })
    .unwrap();
    let max = oracle::maximum::<PcuU512>();
    let sentinel = oracle::small::<PcuU512>(77);
    let mut output = [sentinel; 10];
    let error = local_portable(&mut output, &[max; 7]).unwrap_err();
    assert_eq!(
        error.arithmetic_fault(),
        Some(PcuExecutionFault {
            kind: PcuExecutionFaultKind::ArithmeticOverflow,
            recovered: true,
            invocation_id: 0,
        })
    );
    assert_eq!(output[..7], [max; 7]);
    assert_eq!(output[7..], [sentinel; 3]);
    let binding = local_portable_bindings::<PcuU512>();
    let graph = local_portable_ir::<PcuU512>(&binding).unwrap();
    assert_eq!(
        graph
            .ir()
            .numerical_requirements
            .numerical_options
            .reproducibility,
        pcu_facade::PcuReproducibility::PortableV1
    );
    global::clear_thread_cache().unwrap();
    global::use_defaults().unwrap();
}
