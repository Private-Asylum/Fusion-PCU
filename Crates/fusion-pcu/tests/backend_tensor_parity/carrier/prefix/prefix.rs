//! Ordinary annotated calls consume a logical prefix without shrinking an escaped owner.
use super::*;

#[pcu(invocations = 7)]
fn copy<T: PcuScalar>(input: &[T], output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = input[id];
}

#[pcu(invocations = 1)]
fn grid<T: PcuScalar>(input: &[T], output: &mut [T]) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < 7 {
        output[id] = input[id];
        id += stride;
    }
}

#[pcu(invocations = 7)]
fn broadcast<T: PcuScalar>(input: &[T], output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = input[0];
}

fn check_prefix<T: Sample, const N: usize>(actual: &[T; N], prefix: &[T; 7], tail: T) {
    for (value, expected) in actual[..7].iter().zip(prefix) {
        assert_eq!(value.encode_le().as_ref(), expected.encode_le().as_ref());
    }
    for value in &actual[7..] {
        assert_eq!(value.encode_le().as_ref(), tail.encode_le().as_ref());
    }
}

fn format<T: Sample>() {
    let sentinel = T::sample(91);
    for phase in [0_u8, 23, 71] {
        let full: [T; 11] =
            core::array::from_fn(|id| T::sample(phase.wrapping_add(u8::try_from(id).unwrap())));
        let expected: [T; 7] = full[..7].try_into().unwrap();
        let input = retain(&full).unwrap();
        let mut stack = [sentinel; 9];
        copy(&input, &mut stack).unwrap();
        check_prefix(&stack, &expected, sentinel);
        stack.fill(sentinel);
        grid(&input, &mut stack).unwrap();
        check_prefix(&stack, &expected, sentinel);
        stack.fill(sentinel);
        broadcast(&input, &mut stack).unwrap();
        check_prefix(&stack, &[full[0]; 7], sentinel);

        let mut output = retain(&[sentinel; 11]).unwrap();
        copy(&input, &mut output).unwrap();
        let mut host = [sentinel; 13];
        output.read_into(&mut host).unwrap();
        check_prefix(&host, &expected, sentinel);
        assert_eq!(output.shape(), &[11]);
        // Host values use the minimum read span; switching to host input must
        // not reuse the resident input's complete physical-shape specialization.
        copy(&expected, &mut output).unwrap();
        host.fill(sentinel);
        output.read_into(&mut host).unwrap();
        check_prefix(&host, &expected, sentinel);
        let mut original = [sentinel; 13];
        input.read_into(&mut original).unwrap();
        for (actual, expected) in original[..11].iter().zip(full) {
            assert_eq!(actual.encode_le().as_ref(), expected.encode_le().as_ref());
        }
        for value in &original[11..] {
            assert_eq!(value.encode_le().as_ref(), sentinel.encode_le().as_ref());
        }
        let mut short = [sentinel; 6];
        assert!(copy(&input, &mut short).is_err());
        for value in short {
            assert_eq!(value.encode_le().as_ref(), sentinel.encode_le().as_ref());
        }
        global::clear_thread_cache().unwrap();
        copy(&input, &mut stack).unwrap();
        check_prefix(&stack, &expected, sentinel);
        drop(input);
        host.fill(sentinel);
        output.read_into(&mut host).unwrap();
        check_prefix(&host, &expected, sentinel);
    }
}

/// Separate full-capacity residency gate: every source entry is genuine #[pcu].
pub fn verify(backend: global::PcuBackendChoice) {
    let _guard = POLICY_LOCK.lock().unwrap();
    global::configure(global::PcuExecutionPolicy {
        backend,
        ..Default::default()
    })
    .unwrap();
    format::<u8>();
    format::<i8>();
    format::<u16>();
    format::<i16>();
    format::<u32>();
    format::<i32>();
    format::<u64>();
    format::<i64>();
    format::<u128>();
    format::<i128>();
    format::<PcuU256>();
    format::<PcuI256>();
    format::<PcuU512>();
    format::<PcuI512>();
    format::<PcuF16Bits>();
    format::<PcuBf16Bits>();
    format::<PcuF8E4M3FnBits>();
    format::<PcuF8E5M2Bits>();
    format::<f32>();
    format::<f64>();
    format::<PcuF128Bits>();
    format::<PcuF256Bits>();
    global::clear_thread_cache().unwrap();
    global::use_defaults().unwrap();
}
