//! Requested-header `PortableV1` proof; eligibility is separate from actual GPU conformance.
extern crate pcu_facade as fusion_pcu;
#[path = "../../benches/checked_low_precision/oracle/oracle.rs"]
#[allow(dead_code)]
mod oracle;
#[path = "../../benches/strict_matmul/selection.rs"]
mod selection;
#[path = "../../benches/checked_low_precision/source/source.rs"]
#[allow(dead_code)]
mod source;
#[rustfmt::skip]
use fusion_pcu::{
    global, PcuExecutionError, PcuExecutionFaultKind, PcuFloatUnderflowPolicy,
    PcuF16Bits, PcuBf16Bits, PcuF8E4M3FnBits, PcuF8E5M2Bits,
    PcuBindingRef, PcuHostArgument, PcuHostKernelBackend, PcuPreparedHostKernel,
};
use oracle::Format;
fn selected() {
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Cuda,
        device: Some(0),
        block_size: 256,
        ..Default::default()
    })
    .unwrap();
    global::clear_thread_cache().unwrap();
}
#[allow(clippy::needless_pass_by_value)] // Classify and consume one terminal execution result.
fn fault(result: Result<(), PcuExecutionError>, index: u64, kind: PcuExecutionFaultKind) {
    assert!(
        matches!(&result,Err(PcuExecutionError::ArithmeticFault(f)) if f.invocation_id==index&&f.kind==kind&&!f.recovered),
        "{result:?}"
    );
}
#[allow(clippy::too_many_lines)] // Keep full encoding output and all observed fault classes together.
fn profile<T: Format>() {
    const N: usize = 65536;
    const SMALL: usize = 65;
    let (_, backend, _) = selection::selected_device();
    macro_rules! operation {($module:ident,$op:expr)=>{{
    for policy in [PcuFloatUnderflowPolicy::IeeeAfterRounding,PcuFloatUnderflowPolicy::RejectSubnormalResult,PcuFloatUnderflowPolicy::AllowGradualUnderflow] {
        let mut left=Vec::with_capacity(N);let mut right=Vec::with_capacity(N);let mut expected=Vec::with_capacity(N);let mut witnesses=Vec::new();
        for index in 0..N {
            let a=T::from(u16::try_from(if T::SIGN==0x80 {index/256}else{index}).unwrap());
            let b=T::from(u16::try_from(if T::SIGN==0x80 {index%256}else{(index.wrapping_mul(32749)+17)%65536}).unwrap());
            let (a,b,value)=match oracle::reference(a,b,$op,policy) {
                Ok(v)=>{assert_eq!(v.bits(),oracle::independent(a,b,$op));(a,b,v)},
                Err(error)=>{let kind=error.kind();if !witnesses.iter().any(|(_,_,prior)|*prior==kind){witnesses.push((a,b,kind));}let value=oracle::reference(T::one(),T::one(),$op,policy).unwrap();(T::one(),T::one(),value)}
            };left.push(a);right.push(b);expected.push(value);
        }
        let mut output=vec![T::sentinel();N+2];
        macro_rules! run {($a:expr,$b:expr,$out:expr)=>{match policy {
            PcuFloatUnderflowPolicy::IeeeAfterRounding=>source::$module::portable_grid::<T,N>($a,$b,$out),
            PcuFloatUnderflowPolicy::RejectSubnormalResult=>source::$module::portable_tight::<T,N>($a,$b,$out),
            PcuFloatUnderflowPolicy::AllowGradualUnderflow=>source::$module::portable_allow::<T,N>($a,$b,$out),
        }};}
        run!(&left,&right,&mut output).unwrap();oracle::verify(&expected,&output);
        if policy==PcuFloatUnderflowPolicy::IeeeAfterRounding {
            source::$module::portable::<T,N>(&left,&right,&mut output).unwrap();oracle::verify(&expected,&output);
            source::$module::portable_strict::<T,N>(&left,&right,&mut output).unwrap();oracle::verify(&expected,&output);
        }
        // Every fault class encountered in the exact full encoding corpus is a separate
        // terminal Reject test, with lower logical lane5 competing with lane41.
        for &(a,b,kind) in &witnesses {
            let mut bad_left=vec![T::one();N];let mut bad_right=bad_left.clone();bad_left[5]=a;bad_right[5]=b;bad_left[41]=a;bad_right[41]=b;
            let before=output.clone();fault(run!(&bad_left,&bad_right,&mut output),5,kind);assert_eq!(output,before);
        }
        // Requested-header explicit preparation independently retains eligibility and actual
        // typed operand/load facts; the native peer is supplied by portable_low_precision.
        let bindings=source::$module::portable_grid_bindings::<T>();
        let builder=match policy {
            PcuFloatUnderflowPolicy::IeeeAfterRounding=>source::$module::portable_grid_ir::<T,N>(&bindings),
            PcuFloatUnderflowPolicy::RejectSubnormalResult=>source::$module::portable_tight_ir::<T,N>(&bindings),
            PcuFloatUnderflowPolicy::AllowGradualUnderflow=>source::$module::portable_allow_ir::<T,N>(&bindings),
        }.unwrap();
        let mut prepared = builder.with_ir(|ir| {
                let desc=fusion_pcu::describe_portable_v1_map(ir).unwrap();
                assert_eq!(desc.scalar,T::TYPE);assert_eq!(desc.underflow,policy);
                assert_eq!(desc.logical_extent,u32::try_from(N).unwrap());
                backend.prepare_host_kernel(ir).unwrap()
            });prepared.call(&mut[
            PcuHostArgument::read(PcuBindingRef::new(0,0),&left),PcuHostArgument::read(PcuBindingRef::new(0,1),&right),PcuHostArgument::read_write(PcuBindingRef::new(0,2),&mut output)]).unwrap();oracle::verify(&expected,&output);
    }
    // Genuine portable broadcast and escaped owners, discard, fresh retry and tails.
    // Actual prepared SSA operand permutations, including an unused nonfinite load:
    // no executor may assume load order equals arithmetic operand order.
    let bindings=source::$module::portable_bindings::<T>();let builder=source::$module::portable_ir::<T,SMALL>(&bindings).unwrap();let base=builder.ir();
    for repeated in [false,true] {
        let mut ops=base.ops.to_vec();for op in &mut ops {if let fusion_pcu::PcuDispatchOp::Data(fusion_pcu::PcuDispatchDataOp::CheckedFloatBinary{lhs,rhs,..})=op {if repeated {*rhs=*lhs;}else{std::mem::swap(lhs,rhs);}}}
        let changed=fusion_pcu::PcuDispatchKernelIr{ops:&ops,..base};let desc=fusion_pcu::describe_portable_v1_map(&changed).unwrap();assert_eq!(desc.operands,if repeated{[0,0]}else{[1,0]});
        let a=T::from(T::ONE+(1<<T::FRACTION));let b=if repeated{T::from(T::MAX+1)}else{T::from(T::ONE+(1<<T::FRACTION)+(1<<(T::FRACTION-1)))};
        let left=vec![a;SMALL];let right=vec![b;SMALL];let value=if repeated{oracle::expected(a,a,$op,PcuFloatUnderflowPolicy::IeeeAfterRounding).0}else{oracle::expected(b,a,$op,PcuFloatUnderflowPolicy::IeeeAfterRounding).0};let mut output=vec![T::sentinel();SMALL+2];
        let mut prepared=backend.prepare_host_kernel(&changed).unwrap();prepared.call(&mut[PcuHostArgument::read(PcuBindingRef::new(0,0),&left),PcuHostArgument::read(PcuBindingRef::new(0,1),&right),PcuHostArgument::read_write(PcuBindingRef::new(0,2),&mut output)]).unwrap();oracle::verify(&vec![value;SMALL],&output);
    }
    let left=vec![T::one();SMALL];let right=left.clone();let mut output=vec![T::sentinel();SMALL+2];
    source::$module::portable_broadcast::<T,SMALL>(&left,&T::one(),&mut output).unwrap();
    let expected=vec![oracle::expected(T::one(),T::one(),$op,PcuFloatUnderflowPolicy::IeeeAfterRounding).0;SMALL];oracle::verify(&expected,&output);
    let input=source::identity(left.as_slice()).unwrap();let good=source::identity(right.as_slice()).unwrap();let mut bad=right;bad[5]=T::from(T::MAX+1);bad[41]=T::from(T::MAX+1);let bad=source::identity(bad.as_slice()).unwrap();
    let mut resident=source::identity(output.as_slice()).unwrap();fault(source::$module::portable::<T,SMALL>(&input,&bad,&mut resident),5,PcuExecutionFaultKind::InvalidFloatingOperand);
    assert!(matches!(resident.read_into(&mut output),Err(PcuExecutionError::Argument(global::PcuArgumentError::ResidentValueDiscarded))));assert!(source::identity::<T>(&resident).is_err());assert!(source::$module::portable::<T,SMALL>(&input,&good,&mut resident).is_err());
    resident=source::identity(output.as_slice()).unwrap();source::$module::portable::<T,SMALL>(&input,&good,&mut resident).unwrap();resident.read_into(&mut output).unwrap();oracle::verify(&expected,&output);
    source::$module::portable::<T,SMALL>(&left,&good,&mut resident).unwrap();resident.read_into(&mut output).unwrap();oracle::verify(&expected,&output);
    let mut short=source::identity(&vec![T::sentinel();SMALL-1]).unwrap();assert!(source::$module::portable::<T,SMALL>(&input,&good,&mut short).is_err());let mut prior=vec![T::zero();SMALL-1];short.read_into(&mut prior).unwrap();assert!(prior.iter().all(|v|*v==T::sentinel()));
    // Separate request permissions remain frozen and never weaken the scalar profile.
    global::configure(global::PcuExecutionPolicy{backend:global::PcuBackendChoice::Cuda,device:Some(0),block_size:256,numerical_mode:fusion_pcu::PcuNumericalMode::Strict,
        numerical_options:fusion_pcu::PcuNumericalOptions{compound_arithmetic:fusion_pcu::PcuCompoundArithmeticPolicy::BackendDefined,precision:fusion_pcu::PcuPrecisionPolicy::BackendOptimized,..Default::default()},..Default::default()}).unwrap();
    source::$module::portable::<T,SMALL>(&left,&left,&mut output).unwrap();oracle::verify(&expected,&output);selected();
 }};}
    operation!(add, 0);
    operation!(sub, 1);
    operation!(mul, 2);
    operation!(div, 3);
    // Global inheritance for an otherwise ordinary function and explicit unsupported negatives.
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Cuda,
        numerical_options: fusion_pcu::PcuNumericalOptions {
            reproducibility: fusion_pcu::PcuReproducibility::PortableV1,
            ..Default::default()
        },
        ..Default::default()
    })
    .unwrap();
    let a = [T::one(); 65];
    let mut out = [T::sentinel(); 67];
    source::add::direct::<T, 65>(&a, &a, &mut out).unwrap();
    assert!(source::unsupported_compound::<T, 65>(&a, &a, &mut out).is_err());
    assert!(source::add::portable::<f32, 65>(&[1.0; 65], &[1.0; 65], &mut [99.0; 67]).is_err());
    assert!(source::mul::clamp::<T, 65>(&a, &a, &mut out).is_err());
    selected();
}
macro_rules! fixtures {($($name:ident,$ty:ty;)+)=>{$(
 #[test] #[ignore="requires an authorized GPU; requested-header correctness, run serially"] fn $name(){selected();profile::<$ty>();}
)+};}
fixtures!(half,PcuF16Bits;brain,PcuBf16Bits;e4,PcuF8E4M3FnBits;e5,PcuF8E5M2Bits;);
