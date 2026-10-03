//! Independent exact carrier oracle, all raw narrow encodings and changed high limbs.
#[rustfmt::skip]
use pcu_facade::{PcuScalar,PcuU256,PcuI256,PcuU512,PcuI512,PcuF16Bits,PcuBf16Bits,PcuF8E4M3FnBits,PcuF8E5M2Bits,PcuF128Bits,PcuF256Bits,PcuHostArgument,PcuBindingRef};
use fusion_pcu_metal::MetalSession;
#[path = "sample/sample.rs"]
mod sample;
#[path = "source/source.rs"]
mod source;
use sample::Sample;
fn bytes<T: PcuScalar>(values: &[T]) -> Vec<u8> {
    PcuHostArgument::read(PcuBindingRef::new(0, 0), values)
        .bytes()
        .to_vec()
}
fn qualify<T: Sample, const N: usize>() {
    let _guard = crate::source_policy_guard();
    let session = MetalSession::open(0).unwrap();
    let mut direct = source::direct_prepare::<T, N, _>(&session).unwrap();
    let mut grid = source::grid_prepare::<T, N, _>(&session).unwrap();
    let mut broadcast = source::broadcast_prepare::<T, N, _>(&session).unwrap();
    let mut grid_broadcast = source::grid_broadcast_prepare::<T, N, _>(&session).unwrap();
    let sentinel = T::pattern(531);
    let mut output = vec![sentinel; N + 3];
    for phase in 0..3 {
        let input = (0..N)
            .map(|i| T::pattern(i + phase * 277))
            .collect::<Vec<_>>();
        direct(&input, &mut output).unwrap();
        assert_eq!(bytes(&output[..N]), bytes(&input));
        assert_eq!(bytes(&output[N..]), bytes(&[sentinel; 3]));
        grid(&input, &mut output).unwrap();
        assert_eq!(bytes(&output[..N]), bytes(&input));
        assert_eq!(bytes(&output[N..]), bytes(&[sentinel; 3]));
        let seed = T::pattern(phase * 91 + 257);
        broadcast(&seed, &mut output).unwrap();
        assert_eq!(bytes(&output[..N]), bytes(&vec![seed; N]));
        assert_eq!(bytes(&output[N..]), bytes(&[sentinel; 3]));
        grid_broadcast(&seed, &mut output).unwrap();
        assert_eq!(bytes(&output[..N]), bytes(&vec![seed; N]));
        assert_eq!(bytes(&output[N..]), bytes(&[sentinel; 3]));
        let before = output.clone();
        assert!(direct(&input[..1], &mut output).is_err());
        assert_eq!(bytes(&output), bytes(&before));
    }
}
macro_rules! cases{($($name:ident,$ty:ty,$count:expr);+)=>{$(#[test]#[ignore="Requires actual Metal22 carrier direct/grid/broadcast byte proof."]fn $name(){qualify::<$ty,$count>();})+};}
cases!(u8_carrier,u8,256;i8_carrier,i8,256;u16_carrier,u16,65536;i16_carrier,i16,65536;u32_carrier,u32,257;i32_carrier,i32,257;u64_carrier,u64,257;i64_carrier,i64,257;u128_carrier,u128,257;i128_carrier,i128,257;u256_carrier,PcuU256,257;i256_carrier,PcuI256,257;u512_carrier,PcuU512,257;i512_carrier,PcuI512,257;half_carrier,PcuF16Bits,65536;bf_carrier,PcuBf16Bits,65536;e4_carrier,PcuF8E4M3FnBits,256;e5_carrier,PcuF8E5M2Bits,256;f32_carrier,f32,257;f64_carrier,f64,257;f128_carrier,PcuF128Bits,257;f256_carrier,PcuF256Bits,257);

#[allow(clippy::too_many_lines)] // Exact same-session raw/device/logical/mixed and preflight lifecycle for each22 carrier.
fn resident<T: Sample>() {
    use pcu_facade::{
        PcuTensor, PcuMemoryPoolId, global, PcuNumericalMode, PcuArgumentError, PcuExecutionError,
    };
    let backend = super::low_precision::open_backend();
    let pool = PcuMemoryPoolId(521);
    let sentinel = T::pattern(531);
    let input = std::array::from_fn::<_, 68, _>(|i| T::pattern(i + 257));
    let raw_input = backend.upload_buffer(pool, &input).unwrap();
    let mut raw_output = backend.upload_buffer(pool, &[sentinel; 68]).unwrap();
    let mut raw = source::direct_prepare_device::<T, 65, _>(&backend).unwrap();
    raw(&raw_input, &mut raw_output).unwrap();
    let mut host = [sentinel; 68];
    backend
        .download_buffer(pool, &raw_output, &mut host)
        .unwrap();
    assert_eq!(bytes(&host[..65]), bytes(&input[..65]));
    assert_eq!(bytes(&host[65..]), bytes(&[sentinel; 3]));
    let input_owner = PcuTensor::from_device_buffer(backend.clone(), raw_input, &[68]).unwrap();
    let output = backend.upload_buffer(pool, &[sentinel; 68]).unwrap();
    let mut output_owner = PcuTensor::from_device_buffer(backend.clone(), output, &[68]).unwrap();
    let seed = T::pattern(947);
    let seed_buffer = backend
        .upload_buffer(pool, core::slice::from_ref(&seed))
        .unwrap();
    let seed_owner = PcuTensor::from_device_buffer(backend, seed_buffer, &[]).unwrap();
    let foreign_backend = super::low_precision::open_backend();
    let buffer = foreign_backend
        .upload_buffer(pool, &[sentinel; 68])
        .unwrap();
    let mut foreign = PcuTensor::from_device_buffer(foreign_backend, buffer, &[68]).unwrap();
    for mode in [PcuNumericalMode::Boundary, PcuNumericalMode::Strict] {
        global::configure(global::PcuExecutionPolicy {
            backend: global::PcuBackendChoice::Metal,
            numerical_mode: mode,
            ..Default::default()
        })
        .unwrap();
        source::direct::<T, 65>(&input_owner, &mut output_owner).unwrap();
        output_owner.read_into(&mut host).unwrap();
        assert_eq!(bytes(&host[..65]), bytes(&input[..65]));
        assert_eq!(bytes(&host[65..]), bytes(&[sentinel; 3]));
        source::grid::<T, 65>(&input_owner, &mut host).unwrap();
        assert_eq!(bytes(&host[..65]), bytes(&input[..65]));
        source::direct::<T, 65>(&input, &mut output_owner).unwrap();
        source::broadcast::<T, 65>(&seed_owner, &mut output_owner).unwrap();
        output_owner.read_into(&mut host).unwrap();
        assert_eq!(bytes(&host[..65]), bytes(&[seed; 65]));
        assert_eq!(bytes(&host[65..]), bytes(&[sentinel; 3]));
        source::grid_broadcast::<T, 65>(&seed_owner, &mut host).unwrap();
        assert_eq!(bytes(&host[..65]), bytes(&[seed; 65]));
        assert!(matches!(
            source::direct::<T, 65>(&input_owner, &mut foreign),
            Err(PcuExecutionError::Argument(
                PcuArgumentError::SessionMismatch
            ))
        ));
        foreign.read_into(&mut host).unwrap();
        assert_eq!(bytes(&host), bytes(&[sentinel; 68]));
        assert!(source::direct::<T, 65>(&input[..1], &mut output_owner).is_err());
        output_owner.read_into(&mut host).unwrap();
        assert_eq!(bytes(&host[..65]), bytes(&[seed; 65]));
        source::direct::<T, 65>(&input, &mut output_owner).unwrap();
        input_owner.read_into(&mut host).unwrap();
        assert_eq!(bytes(&host), bytes(&input));
        global::clear_thread_cache().unwrap();
        output_owner.read_into(&mut host).unwrap();
        assert_eq!(bytes(&host[..65]), bytes(&input[..65]));
        assert_eq!(bytes(&host[65..]), bytes(&[sentinel; 3]));
    }
    global::use_defaults().unwrap();
}
#[test]
#[ignore = "Requires actualMetal22 byte-addressed rawdevice,logicalowner,mixed andscalarresidentbroadcast proof."]
fn twenty_two_resident_and_mixed_carriers() {
    let _guard = crate::source_policy_guard();
    macro_rules! widths{($($ty:ty),+)=>{$(resident::<$ty>();)+};}
    widths!(
        u8,
        i8,
        u16,
        i16,
        u32,
        i32,
        u64,
        i64,
        u128,
        i128,
        PcuU256,
        PcuI256,
        PcuU512,
        PcuI512,
        PcuF16Bits,
        PcuBf16Bits,
        PcuF8E4M3FnBits,
        PcuF8E5M2Bits,
        f32,
        f64,
        PcuF128Bits,
        PcuF256Bits
    );
}
