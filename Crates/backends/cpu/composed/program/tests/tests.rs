use super::*;
use alloc::vec;

#[derive(Clone)]
struct Draft {
    scalar: PcuScalarType,
    size: usize,
    extent: usize,
    schema: Vec<(PcuBindingRef, usize, PcuBindingAccess)>,
    resources: Vec<Resource>,
    initializers: Vec<Step>,
    steps: Vec<Step>,
    scratch: usize,
}
impl Draft {
    fn valid() -> Self {
        let input = PcuBindingRef::new(0, 0);
        let output = PcuBindingRef::new(0, 1);
        Self {
            scalar: PcuScalarType::U64,
            size: 8,
            extent: 2,
            schema: vec![
                (input, 2, PcuBindingAccess::ReadOnly),
                (output, 2, PcuBindingAccess::ReadWrite),
            ],
            resources: vec![
                Resource {
                    binding: input,
                    declaration: 0,
                    read_bytes: 16,
                    write_bytes: 0,
                    shadow: None,
                },
                Resource {
                    binding: output,
                    declaration: 1,
                    read_bytes: 0,
                    write_bytes: 16,
                    shadow: Some(0),
                },
            ],
            initializers: vec![],
            steps: vec![
                Step::Load {
                    result: 0,
                    resource: 0,
                    zero: false,
                },
                Step::Store {
                    resource: 1,
                    value: 0,
                },
            ],
            scratch: 16,
        }
    }
    fn seal(self) -> Result<ValidatedProgram, Error> {
        ValidatedProgram::new(
            self.scalar,
            self.size,
            self.extent,
            self.schema,
            self.resources,
            self.initializers,
            self.steps,
            self.scratch,
        )
    }
}
fn rejected(change: impl FnOnce(&mut Draft)) {
    let mut draft = Draft::valid();
    change(&mut draft);
    assert!(matches!(
        draft.seal(),
        Err(Error::InvalidProgram | Error::ExtentOverflow)
    ));
}

#[test]
fn register_bounds_dominance_and_unique_partition_definitions() {
    rejected(|d| {
        d.steps[0] = Step::Load {
            result: STEPS,
            resource: 0,
            zero: false,
        }
    });
    rejected(|d| {
        d.steps[1] = Step::Store {
            resource: 1,
            value: STEPS,
        }
    });
    rejected(|d| d.steps.swap(0, 1));
    rejected(|d| {
        d.initializers.push(Step::IntegerConstant {
            result: 0,
            bytes: [0; 16],
        });
    });
    rejected(|d| {
        d.steps.insert(
            1,
            Step::Load {
                result: 0,
                resource: 0,
                zero: true,
            },
        );
    });
    rejected(|d| {
        d.initializers.push(Step::Store {
            resource: 1,
            value: 0,
        });
    });
    rejected(|d| {
        d.initializers.push(Step::Load {
            result: 1,
            resource: 0,
            zero: false,
        });
    });
    rejected(|d| {
        d.steps.push(Step::IntegerConstant {
            result: 1,
            bytes: [0; 16],
        });
    });
    rejected(|d| d.steps.extend(vec![d.steps[1]; STEPS]));
    let mut draft = Draft::valid();
    draft.steps[0] = Step::Load {
        result: STEPS - 1,
        resource: 0,
        zero: false,
    };
    draft.steps[1] = Step::Store {
        resource: 1,
        value: STEPS - 1,
    };
    assert!(draft.seal().is_ok());
}

#[test]
fn resource_bounds_permissions_and_publication_coverage() {
    rejected(|d| {
        d.steps[0] = Step::Load {
            result: 0,
            resource: 2,
            zero: false,
        }
    });
    rejected(|d| {
        d.steps[1] = Step::Store {
            resource: BINDINGS,
            value: 0,
        }
    });
    rejected(|d| d.resources[0].declaration = 2);
    rejected(|d| d.resources[0].binding = PcuBindingRef::new(1, 9));
    rejected(|d| d.schema[1].0 = d.schema[0].0);
    rejected(|d| d.schema[1].2 = PcuBindingAccess::ReadOnly);
    rejected(|d| d.resources[1].shadow = None);
    rejected(|d| d.resources[0].shadow = Some(0));
    rejected(|d| d.resources[0].read_bytes = 8);
    rejected(|d| d.resources[0].read_bytes = 15);
    rejected(|d| d.schema[0].1 = 1);
    rejected(|d| d.resources[1].write_bytes = 8);
    rejected(|d| {
        d.resources[1].write_bytes = 24;
        d.schema[1].1 = 3;
        d.scratch = 24;
    });
    rejected(|d| {
        d.steps.pop();
    });
    rejected(|d| {
        d.initializers.push(Step::Load {
            result: 1,
            resource: 1,
            zero: true,
        });
        d.resources[1].read_bytes = 16;
    });
    let mut draft = Draft::valid();
    for index in 2..BINDINGS {
        let binding = PcuBindingRef::new(0, u32::try_from(index).unwrap());
        draft.schema.push((binding, 2, PcuBindingAccess::ReadWrite));
        draft.resources.push(Resource {
            binding,
            declaration: index,
            read_bytes: 0,
            write_bytes: 16,
            shadow: Some((index - 1) * 16),
        });
        draft.steps.push(Step::Store {
            resource: index,
            value: 0,
        });
    }
    draft.scratch = 48;
    assert!(draft.clone().seal().is_ok());
    draft.resources[3].shadow = Some(8);
    assert!(matches!(draft.seal(), Err(Error::InvalidProgram)));
}

#[test]
fn extent_overflow_shadow_bounds_and_carrier_identity() {
    rejected(|d| d.extent = usize::MAX);
    rejected(|d| d.schema[0].1 = usize::MAX);
    rejected(|d| d.resources[1].shadow = Some(usize::MAX));
    rejected(|d| d.scratch = 15);
    rejected(|d| d.size = 4);
    rejected(|d| d.scalar = PcuScalarType::Bool);
    rejected(|d| {
        d.initializers.push(Step::Constant {
            result: 1,
            bytes: [0; 8],
        });
    });
    let program = Draft::valid().seal().unwrap();
    assert!(program.validate_carrier::<u64>(16).is_ok());
    assert!(matches!(
        program.validate_carrier::<i64>(16),
        Err(Error::InvalidProgram)
    ));
    assert!(matches!(
        program.validate_carrier::<u64>(15),
        Err(Error::InvalidProgram)
    ));
    let cloned = program.clone();
    assert!(cloned.validate_carrier::<u64>(16).is_ok());
    assert_ne!(program.steps().as_ptr(), cloned.steps().as_ptr());
}

#[test]
fn zero_extent_unreachable_store_without_shadow_and_nonfinite_constant() {
    let mut draft = Draft::valid();
    draft.extent = 0;
    draft.resources[1].write_bytes = 0;
    draft.resources[1].shadow = None;
    draft.scratch = 0;
    assert!(draft.seal().is_ok());
    let mut draft = Draft::valid();
    draft.scalar = PcuScalarType::F32;
    draft.size = 4;
    draft.resources[0].read_bytes = 8;
    draft.resources[1].write_bytes = 8;
    draft.scratch = 8;
    let mut bytes = [0; 8];
    bytes[..4].copy_from_slice(&f32::NAN.to_ne_bytes());
    draft.initializers.push(Step::Constant { result: 1, bytes });
    // The boundary validates representation only, preserving runtime fault timing.
    assert!(draft.seal().is_ok());
}

#[test]
fn safe_private_executor_cannot_bypass_argument_or_carrier_preflight() {
    #[rustfmt::skip]
    use fusion_pcu::{
        PcuHostArgument,
        PcuImplementationRequirements,
    };
    let mut plan = super::super::PcuCpuPreparedComposedMap {
        requirements: PcuImplementationRequirements::DEFAULT,
        local_id: 0,
        program: Draft::valid().seal().unwrap(),
        scratch: vec![0xa5; 16],
    };
    let input = [1_u64];
    let mut output = [77_u64; 2];
    let execute = plan.program.executor();
    assert!(matches!(
        execute(
            &mut plan,
            &mut [
                PcuHostArgument::read(PcuBindingRef::new(0, 0), &input),
                PcuHostArgument::read_write(PcuBindingRef::new(0, 1), &mut output),
            ]
        ),
        Err(Error::InvalidArguments(_))
    ));
    assert_eq!(output, [77; 2]);
    assert_eq!(plan.scratch, vec![0xa5; 16]);
    let input = [1_u64; 2];
    let wrong_carrier = execution::select(PcuScalarType::F64).unwrap();
    assert!(matches!(
        wrong_carrier(
            &mut plan,
            &mut [
                PcuHostArgument::read(PcuBindingRef::new(0, 0), &input),
                PcuHostArgument::read_write(PcuBindingRef::new(0, 1), &mut output),
            ]
        ),
        Err(Error::InvalidProgram)
    ));
    assert_eq!(output, [77; 2]);
}

#[test]
fn integer_block_admission_requires_readonly_loads_and_non_tiny_extent() {
    assert!(
        Draft::valid()
            .seal()
            .unwrap()
            .integer_instructions()
            .is_none()
    );
    let mut draft = Draft::valid();
    draft.extent = 64;
    for entry in &mut draft.schema {
        entry.1 = 64;
    }
    draft.resources[0].read_bytes = 512;
    draft.resources[1].write_bytes = 512;
    draft.scratch = 512;
    let program = draft.clone().seal().unwrap();
    assert!(program.integer_instructions().is_some());
    for zero in [false, true] {
        let mut mutable = draft.clone();
        mutable.resources[1].read_bytes = 512;
        mutable.steps[0] = Step::Load {
            result: 0,
            resource: 1,
            zero,
        };
        // Both mutable per-lane loads and cross-lane broadcasts stay on the
        // validated scalar path. Merely sealing them is not an independence proof.
        let program = mutable.seal().unwrap();
        assert!(program.integer_instructions().is_none());
    }
    let mut maximum = draft;
    maximum.steps[0] = Step::Load {
        result: STEPS - 1,
        resource: 0,
        zero: false,
    };
    maximum.steps[1] = Step::Store {
        resource: 1,
        value: STEPS - 1,
    };
    assert!(maximum.seal().unwrap().integer_instructions().is_some());
}
