use super::*;
use crate::global::PcuExecutionPolicy;

const fn descriptor() -> crate::PcuDeviceDescriptor<'static> {
    let reference = crate::PcuObjectRef {
        provider: crate::PcuProviderId(5),
        generation: 7,
        kind: crate::PcuObjectKind::Device,
        id: 2,
    };
    crate::PcuDeviceDescriptor {
        reference,
        target: crate::PcuObjectRef {
            kind: crate::PcuObjectKind::Target,
            ..reference
        },
        name: "reported device",
        class: crate::PcuDeviceClass::Gpu,
        vendor: None,
        architecture: None,
        generation: None,
        location: None,
    }
}

fn score(candidate: &PcuInvocationCandidate<'_>) -> i128 {
    assert_eq!(candidate.total_memory_bytes, None);
    assert_eq!(candidate.facts.compute_unit_count, Some(17));
    assert_eq!(candidate.kernel.entry.logical_shape, [32, 1, 1]);
    assert_eq!(
        candidate.policy_defaults.range_policy,
        crate::PcuRangePolicy::Clamp
    );
    i128::from(candidate.kernel.entry.logical_shape[0])
}

#[test]
fn operation_scorer_receives_actual_work_facts_and_unknown_capacity() {
    let builder = crate::model::PcuDispatchKernelBuilder::<4>::new(1, "entry", [32, 1, 1]);
    let kernel = builder.ir();
    let policy = PcuExecutionPolicy {
        score_invocation: Some(score),
        range_policy: crate::PcuRangePolicy::Clamp,
        ..Default::default()
    };
    let mut fact_queries = 0;
    let result = score_candidate(policy, descriptor(), None, Some(&kernel), || {
        fact_queries += 1;
        Ok::<_, ()>(crate::PcuDeviceFacts {
            compute_unit_count: Some(17),
            ..Default::default()
        })
    });
    assert_eq!(result, Ok(32));
    assert_eq!(fact_queries, 1);
}

#[test]
fn default_and_non_invocation_ranking_do_not_query_physical_facts() {
    let policy = PcuExecutionPolicy::default();
    assert_eq!(
        score_candidate::<()>(policy, descriptor(), Some(42), None, || panic!(
            "cold facts"
        )),
        Ok(42)
    );
    assert_eq!(
        score_candidate::<()>(
            PcuExecutionPolicy {
                score_invocation: Some(score),
                ..policy
            },
            descriptor(),
            None,
            None,
            || panic!("no invocation to score"),
        ),
        Ok(0)
    );
}

#[test]
fn fact_query_failure_is_retained_instead_of_inventing_hardware() {
    let builder = crate::model::PcuDispatchKernelBuilder::<4>::new(1, "entry", [32, 1, 1]);
    let kernel = builder.ir();
    assert_eq!(
        score_candidate(
            PcuExecutionPolicy {
                score_invocation: Some(score),
                ..Default::default()
            },
            descriptor(),
            None,
            Some(&kernel),
            || Err("provider fact failure"),
        ),
        Err("provider fact failure")
    );
}
