//! Independent widened signed/unsigned arithmetic verifies complete eight-bit pairs and transactions.
use super::*;
fn isa() -> PcuCpuImplementation {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        assert!(std::is_x86_feature_detected!("sse2"));
        PcuCpuImplementation::Sse2
    }
    #[cfg(target_arch = "aarch64")]
    {
        assert!(std::arch::is_aarch64_feature_detected!("neon"));
        PcuCpuImplementation::Neon
    }
}
fn byte(value: i16) -> u8 {
    u8::try_from(value.rem_euclid(256)).unwrap()
}
fn complete<T: PcuCheckedInteger>(signed: bool) {
    let (min, max) = if signed { (-128, 127) } else { (0, 255) };
    for op in [
        PcuDispatchIntegerBinaryOp::Add,
        PcuDispatchIntegerBinaryOp::Sub,
    ] {
        for range in [PcuRangePolicy::Reject, PcuRangePolicy::Clamp] {
            let execute = prepare::<T>(isa(), op, range, false, false).unwrap();
            for x in 0u16..256 {
                for start in (0u16..256).step_by(16) {
                    let a = [u8::try_from(x).unwrap(); 19];
                    let mut b = [0u8; 19];
                    for (i, slot) in b[..16].iter_mut().enumerate() {
                        *slot = u8::try_from(start + u16::try_from(i).unwrap()).unwrap();
                    }
                    let mut output = [117u8; 22];
                    let mut expected = [117u8; 22];
                    let mut first = None;
                    for i in 0..19 {
                        let value = |v: u8| {
                            if signed && v >= 128 {
                                i16::from(v) - 256
                            } else {
                                i16::from(v)
                            }
                        };
                        let lhs = value(a[i]);
                        let rhs = value(b[i]);
                        let r = if op == PcuDispatchIntegerBinaryOp::Add {
                            lhs + rhs
                        } else {
                            lhs - rhs
                        };
                        if r < min || r > max {
                            first.get_or_insert((
                                i,
                                if r < min {
                                    fusion_pcu::PcuExecutionFaultKind::ArithmeticUnderflow
                                } else {
                                    fusion_pcu::PcuExecutionFaultKind::ArithmeticOverflow
                                },
                            ));
                        }
                        expected[i] = byte(r.clamp(min, max));
                    }
                    let result = execute(&a, &b, &mut output, 19);
                    if let Some((index, kind)) = first {
                        assert_eq!(
                            result,
                            Err(PcuCpuCheckedIntegerError::Fault(PcuExecutionFault {
                                kind,
                                invocation_id: u64::try_from(index).unwrap(),
                                recovered: range == PcuRangePolicy::Clamp
                            }))
                        );
                        if range == PcuRangePolicy::Reject {
                            assert_eq!(output, [117; 22]);
                        } else {
                            assert_eq!(output, expected);
                        }
                    } else {
                        assert_eq!(result, Ok(()));
                        assert_eq!(output, expected);
                    }
                }
            }
        }
    }
}
#[test]
fn unsigned_complete_encoding_pairs() {
    complete::<u8>(false);
}
#[test]
fn signed_complete_encoding_pairs() {
    complete::<i8>(true);
}

macro_rules! width {
 ($test:ident,$ty:ty)=>{
 #[test]
 fn $test(){
  let min=i128::from(<$ty>::MIN);let max=i128::from(<$ty>::MAX);
  let mut seed=0x541c_43a7_b628_931d_u64;
  for extent in[1usize,2,3,7,15,16,17,31,32,33]{
   let mut a=[<$ty>::MIN;33];let mut b=[<$ty>::MAX;33];
   for i in 0..33{seed=seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);let bits=seed.to_le_bytes();a[i]=<$ty>::from_le_bytes(core::array::from_fn(|j|bits[j]));seed=seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);let bits=seed.to_le_bytes();b[i]=<$ty>::from_le_bytes(core::array::from_fn(|j|bits[j]));}
   a[0]=<$ty>::MAX;b[0]=1;a[1]=<$ty>::MIN;b[1]=1;a[2]=0;b[2]=1;
   for op in[PcuDispatchIntegerBinaryOp::Add,PcuDispatchIntegerBinaryOp::Sub]{for range in[PcuRangePolicy::Reject,PcuRangePolicy::Clamp]{for(lb,rb)in[(false,false),(true,false),(false,true),(true,true)]{
    let left:std::vec::Vec<_>=a[..if lb{1}else{extent}].iter().flat_map(|v|v.to_le_bytes()).collect();let right:std::vec::Vec<_>=b[..if rb{1}else{extent}].iter().flat_map(|v|v.to_le_bytes()).collect();
    let bytes=extent*core::mem::size_of::<$ty>();let mut output=std::vec![117u8;bytes+7];let mut expected=output.clone();let mut first=None;
    for i in 0..extent{let lhs=i128::from(a[if lb{0}else{i}]);let rhs=i128::from(b[if rb{0}else{i}]);let result=if op==PcuDispatchIntegerBinaryOp::Add{lhs+rhs}else{lhs-rhs};if result<min||result>max{first.get_or_insert((i,if result<min{fusion_pcu::PcuExecutionFaultKind::ArithmeticUnderflow}else{fusion_pcu::PcuExecutionFaultKind::ArithmeticOverflow}));}expected[i*core::mem::size_of::<$ty>()..(i+1)*core::mem::size_of::<$ty>()].copy_from_slice(&<$ty>::try_from(result.clamp(min,max)).unwrap().to_le_bytes());}
    let result=prepare::<$ty>(isa(),op,range,lb,rb).unwrap()(&left,&right,&mut output,extent);
    if let Some((index,kind))=first{assert_eq!(result,Err(PcuCpuCheckedIntegerError::Fault(PcuExecutionFault{kind,invocation_id:u64::try_from(index).unwrap(),recovered:range==PcuRangePolicy::Clamp})));if range==PcuRangePolicy::Reject{assert!(output.iter().all(|v|*v==117));}else{assert_eq!(output,expected);}}
    else{assert_eq!(result,Ok(()));assert_eq!(output,expected);}
   }}}
  }
 }
 };
}
width!(unsigned16_full_bits_and_boundaries, u16);
width!(signed16_full_bits_and_boundaries, i16);
width!(unsigned32_full_bits_and_boundaries, u32);
width!(signed32_full_bits_and_boundaries, i32);
width!(unsigned64_full_bits_and_boundaries, u64);
width!(signed64_full_bits_and_boundaries, i64);
