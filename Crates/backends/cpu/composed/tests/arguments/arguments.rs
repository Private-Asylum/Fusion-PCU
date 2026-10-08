//! Error precedence and call-local routing must agree with the public validator.
use super::*;
use crate::host::{validate_arguments, validate_arguments_with_indices};
use crate::PcuCpuHostArgumentError as ArgumentError;

#[test]
fn resolver_preserves_validation_order_and_unused_declarations() {
    let first = PcuBindingRef::new(0, 0);
    let unused = PcuBindingRef::new(0, 1);
    let foreign = PcuBindingRef::new(0, 9);
    let schema = [
        (first, 4, PcuBindingAccess::ReadOnly),
        (unused, 0, PcuBindingAccess::ReadOnly),
    ];
    let check = |arguments: &[PcuHostArgument<'_>], expected| {
        assert_eq!(
            validate_arguments(arguments, &schema, PcuScalarType::F32, 4),
            Err(expected)
        );
        let mut resolved = [usize::MAX; 2];
        assert_eq!(
            validate_arguments_with_indices(
                arguments,
                &schema,
                PcuScalarType::F32,
                4,
                |declaration, position| resolved[declaration] = position
            ),
            Err(expected)
        );
    };
    check(
        &[PcuHostArgument::read(first, &[] as &[u32])],
        ArgumentError::Count {
            expected: 2,
            actual: 1,
        },
    );
    check(
        &[
            PcuHostArgument::read(first, &[] as &[u32]),
            PcuHostArgument::read(first, &[] as &[f32]),
        ],
        ArgumentError::DuplicateBinding(first),
    );
    check(
        &[
            PcuHostArgument::read(unused, &[] as &[u32]),
            PcuHostArgument::read(foreign, &[] as &[f32]),
        ],
        ArgumentError::MissingBinding(first),
    );
    check(
        &[
            PcuHostArgument::read(unused, &[] as &[u32]),
            PcuHostArgument::read(first, &[] as &[u32]),
        ],
        ArgumentError::TypeMismatch {
            binding: first,
            expected: PcuScalarType::F32,
            actual: PcuScalarType::U32,
        },
    );
    let mut empty = [] as [f32; 0];
    check(
        &[
            PcuHostArgument::read(unused, &[] as &[u32]),
            PcuHostArgument::read_write(first, &mut empty),
        ],
        ArgumentError::AccessMismatch {
            binding: first,
            expected: PcuBindingAccess::ReadOnly,
            actual: PcuBindingAccess::ReadWrite,
        },
    );
    check(
        &[
            PcuHostArgument::read(unused, &[] as &[u32]),
            PcuHostArgument::read(first, &[1_f32]),
        ],
        ArgumentError::BufferTooSmall {
            binding: first,
            required_bytes: 16,
            actual_bytes: 4,
        },
    );
    check(
        &[
            PcuHostArgument::read(unused, &[] as &[u32]),
            PcuHostArgument::read(first, &[1_f32; 4]),
        ],
        ArgumentError::TypeMismatch {
            binding: unused,
            expected: PcuScalarType::F32,
            actual: PcuScalarType::U32,
        },
    );
}

#[test]
fn resolver_returns_current_argument_positions() {
    let first = PcuBindingRef::new(0, 0);
    let unused = PcuBindingRef::new(0, 1);
    let schema = [
        (first, 4, PcuBindingAccess::ReadOnly),
        (unused, 0, PcuBindingAccess::ReadOnly),
    ];
    let arguments = [
        PcuHostArgument::read(unused, &[] as &[f32]),
        PcuHostArgument::read(first, &[1_f32; 4]),
    ];
    let mut resolved = [usize::MAX; 2];
    validate_arguments_with_indices(
        &arguments,
        &schema,
        PcuScalarType::F32,
        4,
        |declaration, position| resolved[declaration] = position,
    )
    .unwrap();
    assert_eq!(resolved, [1, 0]);
}

#[test]
fn invalid_unused_argument_leaves_scratch_and_outputs_untouched() {
    let bindings = unread_bindings::<f32>();
    unread_ir::<f32>(&bindings).unwrap().with_ir(|kernel| {
        let mut plan = PcuCpuCheckedComposedMap::<f32>::new()
            .prepare_host_kernel(kernel)
            .unwrap();
        plan.scratch.fill(0xa5);
        let scratch = plan.scratch.clone();
        let mut output = [99_f32; 6];
        assert_eq!(
            plan.call(&mut [
                PcuHostArgument::read_write(bindings[2].reference(), &mut output),
                PcuHostArgument::read(bindings[1].reference(), &[1_f32; 4]),
                PcuHostArgument::read(bindings[0].reference(), &[] as &[u32]),
            ]),
            Err(PcuCpuComposedMapError::InvalidArguments(
                ArgumentError::TypeMismatch {
                    binding: bindings[0].reference(),
                    expected: PcuScalarType::F32,
                    actual: PcuScalarType::U32,
                }
            ))
        );
        assert_eq!(plan.scratch, scratch);
        assert_eq!(output.map(f32::to_bits), [99_f32.to_bits(); 6]);
        plan.call(&mut [
            PcuHostArgument::read(bindings[0].reference(), &[] as &[f32]),
            PcuHostArgument::read_write(bindings[2].reference(), &mut output),
            PcuHostArgument::read(bindings[1].reference(), &[2_f32; 4]),
        ])
        .unwrap();
        assert_eq!(
            output[..4]
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            [8_f32.to_bits(); 4]
        );
        assert_eq!(
            output[4..]
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            [99_f32.to_bits(); 2]
        );
    });
}

#[test]
fn extent_overflow_precedes_buffer_size_and_later_declarations() {
    let first = PcuBindingRef::new(0, 0);
    let later = PcuBindingRef::new(0, 1);
    let schema = [
        (first, usize::MAX, PcuBindingAccess::ReadOnly),
        (later, 0, PcuBindingAccess::ReadWrite),
    ];
    let arguments = [
        PcuHostArgument::read(later, &[] as &[u32]),
        PcuHostArgument::read(first, &[] as &[f32]),
    ];
    let expected = Err(ArgumentError::ExtentOverflow(first));
    assert_eq!(
        validate_arguments(&arguments, &schema, PcuScalarType::F32, 4),
        expected
    );
    let mut resolved = [usize::MAX; 2];
    assert_eq!(
        validate_arguments_with_indices(
            &arguments,
            &schema,
            PcuScalarType::F32,
            4,
            |declaration, position| resolved[declaration] = position
        ),
        expected
    );
    assert_eq!(resolved, [usize::MAX; 2]);
}
