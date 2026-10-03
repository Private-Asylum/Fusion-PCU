//! Native opaque source routing and exact retained ownership.
#[rustfmt::skip]
use fusion_pcu::{
    global,
    global::{PcuArgumentError, PcuBackendChoice, PcuExecutionPolicy},
    PcuExecutionError,
};
#[path = "mlx/binary/binary.rs"]
mod binary;
#[path = "mlx/integer/integer.rs"]
mod integer;
mod source;
static POLICY_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[test]
#[cfg_attr(
    not(all(target_os = "macos", target_arch = "aarch64")),
    ignore = "requires pinned native Apple silicon MLX"
)]
fn ordinary_mlx_source_staging_affinity_consumption_and_escape() {
    let _guard = POLICY_LOCK.lock().unwrap();
    global::configure(PcuExecutionPolicy {
        backend: PcuBackendChoice::Mlx,
        ..PcuExecutionPolicy::default()
    })
    .unwrap();
    let left = [[1.0_f32, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let right = [[7.0_f32, 8.0], [9.0, 10.0], [11.0, 12.0]];
    let identity = [[1.0_f32, 0.0], [0.0, 1.0]];
    let result = source::matrix(&left, &right).unwrap();
    assert_eq!(result.shape(), [2, 2]);
    let mut actual = [99.0_f32; 5];
    result.read_into(&mut actual).unwrap();
    assert_eq!(
        actual.map(f32::to_bits),
        [58.0_f32, 64.0, 139.0, 154.0, 99.0].map(f32::to_bits)
    );
    let escaped = source::matrix::<2, 2, 2>(&result, &identity).unwrap();
    global::clear_thread_cache().unwrap();
    drop(result);
    escaped.read_into(&mut actual).unwrap();
    assert_eq!(
        actual.map(f32::to_bits),
        [58.0_f32, 64.0, 139.0, 154.0, 99.0].map(f32::to_bits)
    );
    let consumed = source::consume_matrix(escaped, &identity).unwrap();
    consumed.read_into(&mut actual).unwrap();
    assert_eq!(
        actual.map(f32::to_bits),
        [58.0_f32, 64.0, 139.0, 154.0, 99.0].map(f32::to_bits)
    );
    // A different cold shape shares the live realm/device session. Shape or
    // call-site differences are not proof of foreign physical ownership.
    let compatible = source::matrix(&identity, &identity).unwrap();
    let composed = source::matrix::<2, 2, 2>(&consumed, &compatible).unwrap();
    composed.read_into(&mut actual).unwrap();
    assert_eq!(
        actual.map(f32::to_bits),
        [58.0_f32, 64.0, 139.0, 154.0, 99.0].map(f32::to_bits)
    );
    assert!(source::checked_matrix(&left, &right).is_err());
    assert!(source::portable_matrix(&left, &right).is_err());
    #[cfg(feature = "metal")]
    {
        global::configure(PcuExecutionPolicy {
            backend: PcuBackendChoice::Metal,
            ..PcuExecutionPolicy::default()
        })
        .unwrap();
        assert!(matches!(
            source::matrix::<2, 2, 2>(&consumed, &identity),
            Err(PcuExecutionError::ResidentPolicyConflict)
        ));
        global::configure(PcuExecutionPolicy {
            backend: PcuBackendChoice::Mlx,
            ..PcuExecutionPolicy::default()
        })
        .unwrap();
    }
    let target = fusion_pcu::PcuBindingRef::new(0, 0);
    let borrow = <fusion_pcu::PcuTensor<f32> as global::PcuReadStorage<
        f32,
        global::FixedMatrixShape<2, 2>,
    >>::as_pcu_call_argument(&consumed, target);
    assert!(matches!(
        borrow,
        Err(PcuArgumentError::UnsupportedResidentBorrow)
    ));
    let mut short = [77.0_f32; 3];
    assert!(consumed.read_into(&mut short).is_err());
    assert_eq!(short.map(f32::to_bits), [77.0_f32; 3].map(f32::to_bits));
    global::configure(PcuExecutionPolicy {
        backend: PcuBackendChoice::Mlx,
        range_policy: fusion_pcu::PcuRangePolicy::Clamp,
        ..PcuExecutionPolicy::default()
    })
    .unwrap();
    assert!(matches!(
        source::matrix(&left, &right),
        Err(PcuExecutionError::UnsupportedRangePolicy)
    ));
    global::use_defaults().unwrap();
    #[cfg(not(any(feature = "rocm", feature = "cuda", feature = "vulkan")))]
    {
        automatic_roles(&left, &right, &identity, &mut actual);
    }
    global::clear_thread_cache().unwrap();
}

#[test]
#[cfg_attr(
    not(all(target_os = "macos", target_arch = "aarch64")),
    ignore = "requires native Apple silicon cached MLX storage views"
)]
fn ordinary_mlx_bit_views_compose_and_retain_escaped_owners() {
    let _guard = POLICY_LOCK.lock().unwrap();
    global::configure(PcuExecutionPolicy {
        backend: PcuBackendChoice::Mlx,
        ..Default::default()
    })
    .unwrap();
    let right = [[7.0_f32, 8.0], [9.0, 10.0], [11.0, 12.0]];
    let identity = [[1.0_f32, 0.0], [0.0, 1.0]];
    // Literal exact products keep the oracle independent of the native BLAS
    // accumulation/contraction policy; all inputs and products are integers.
    for (shift, expected_values) in [
        (0.0_f32, [58.0_f32, 64.0, 139.0, 154.0, 99.0]),
        (1.0, [65.0, 72.0, 146.0, 162.0, 99.0]),
        (-1.0, [51.0, 56.0, 132.0, 146.0, 99.0]),
    ] {
        let left = [[1.0 + shift, 2.0, 3.0], [4.0 + shift, 5.0, 6.0]];
        let encoded_left = source::identity_matrix(&left).unwrap();
        let encoded_right = source::identity_matrix(&right).unwrap();
        let product = source::matrix::<2, 3, 2>(&encoded_left, &encoded_right).unwrap();
        // Each boundary is a real source function. The borrowed owners retain
        // same-session device storage; conversion is a cached descriptor view.
        let encoded_product = source::identity_matrix::<2, 2>(&product).unwrap();
        let native_again = source::matrix::<2, 2, 2>(&encoded_product, &identity).unwrap();
        global::clear_thread_cache().unwrap();
        drop(encoded_left);
        drop(encoded_right);
        drop(product);
        drop(encoded_product);
        let escaped = source::consume_identity(native_again).unwrap();
        let borrowed = source::identity_matrix::<2, 2>(&escaped).unwrap();
        let expected = expected_values.map(f32::to_bits);
        let mut stack = [99.0_f32; 5];
        borrowed.read_into(&mut stack).unwrap();
        assert_eq!(stack.map(f32::to_bits), expected);
        drop(borrowed);
        escaped.read_into(&mut stack).unwrap();
        assert_eq!(stack.map(f32::to_bits), expected);
        let mut short = [77.0_f32; 3];
        assert!(escaped.read_into(&mut short).is_err());
        assert_eq!(short.map(f32::to_bits), [77.0_f32; 3].map(f32::to_bits));
        assert!(source::checked_matrix::<2, 2, 2>(&escaped, &identity).is_err());
        assert!(source::portable_matrix::<2, 2, 2>(&escaped, &identity).is_err());
    }
    global::clear_thread_cache().unwrap();
    global::use_defaults().unwrap();
}

#[cfg(not(any(feature = "rocm", feature = "cuda", feature = "vulkan")))]
fn automatic_roles(
    left: &[[f32; 3]; 2],
    right: &[[f32; 2]; 3],
    identity: &[[f32; 2]; 2],
    actual: &mut [f32; 5],
) {
    let automatic = source::matrix(left, right).unwrap();
    automatic.read_into(actual).unwrap();
    assert_eq!(
        actual.map(f32::to_bits),
        [58.0_f32, 64.0, 139.0, 154.0, 99.0].map(f32::to_bits)
    );
    // The same genuine source site switches roles; its cold offer boundary must switch too.
    let borrowed = source::matrix::<2, 2, 2>(&automatic, identity).unwrap();
    let both = source::matrix::<2, 2, 2>(&automatic, &borrowed).unwrap();
    both.read_into(actual).unwrap();
    assert_eq!(
        actual.map(f32::to_bits),
        [12260.0_f32, 13568.0, 29468.0, 32612.0, 99.0].map(f32::to_bits)
    );
    let right_borrowed = source::matrix::<2, 2, 2>(identity, &borrowed).unwrap();
    global::clear_thread_cache().unwrap();
    drop(automatic);
    drop(borrowed);
    let retained = source::matrix::<2, 2, 2>(&right_borrowed, identity).unwrap();
    retained.read_into(actual).unwrap();
    assert_eq!(
        actual.map(f32::to_bits),
        [58.0_f32, 64.0, 139.0, 154.0, 99.0].map(f32::to_bits)
    );
    assert!(source::checked_matrix(left, right).is_err());
    assert!(source::portable_matrix(left, right).is_err());
}
