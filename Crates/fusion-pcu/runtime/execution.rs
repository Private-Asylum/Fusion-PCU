//! Allocation-free validation of dependencies between prepared work units.
//!
//! This is a scheduling contract, not a scheduler. A provider may submit independent nodes
//! concurrently, but every conflicting use of the same logical resource must be ordered by a
//! dependency path. Resource IDs must already account for physical aliases.

use crate::PcuMemoryAccess;

/// Identity of a resource within one execution graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PcuExecutionResourceId(pub u32);

/// One use of a logical resource by a work unit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PcuExecutionResourceUse {
    pub resource: PcuExecutionResourceId,
    pub access: PcuMemoryAccess,
}

/// One prepared work unit. Dependencies name earlier nodes by their slice index.
///
/// The node's actual operation, execution context, and completion remain owned by its dialect
/// and backend. This descriptor only states the ordering and resource-access law.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PcuExecutionNode<'a> {
    pub dependencies: &'a [usize],
    pub resources: &'a [PcuExecutionResourceUse],
}

/// Upgrade one existing ordering edge into a terminal-success gate.
///
/// The successor must not begin until the predecessor has completed successfully. This is
/// stronger than a plain ordering edge, which may be satisfied by ordered enqueue on a queue.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PcuExecutionSuccessGate {
    pub predecessor: usize,
    pub successor: usize,
}

/// Terminal state recorded for one node by [`PcuExecutionFaultGateState`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PcuExecutionNodeState {
    /// The node has not been submitted.
    #[default]
    Pending,
    /// The node has been submitted and has not reached a terminal state.
    Running,
    /// The node completed successfully.
    Succeeded,
    /// The node completed with an error.
    Failed,
    /// The node was cancelled before successful completion.
    Cancelled,
}

/// Result of checking whether a node may be submitted under its success gates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PcuExecutionAdmission {
    /// Every gated predecessor has succeeded; the node was changed to `Running`.
    Started,
    /// At least one gated predecessor has not completed yet.
    Waiting,
    /// A gated predecessor failed or was cancelled; the node must not be submitted.
    Blocked,
    /// The node is already running or terminal.
    AlreadyStarted,
}

/// Caller-owned, allocation-free runtime enforcement for declared success gates.
///
/// The backend must call [`try_start`](Self::try_start) immediately before submitting each
/// node, and [`finish`](Self::finish) only after device completion is known. The caller owns
/// the state slice and all operation/resource storage; it must retain resources for every
/// `Running` node until `finish` records success, failure, or cancellation. A blocked node is
/// never submitted; use [`cancel_pending`](Self::cancel_pending) to mark it cancelled.
pub struct PcuExecutionFaultGateState<'a> {
    gates: &'a [PcuExecutionSuccessGate],
    states: &'a mut [PcuExecutionNodeState],
}

impl<'a> PcuExecutionFaultGateState<'a> {
    /// Creates a gate tracker over caller-owned state, initially marking every node pending.
    ///
    /// Callers must first validate the gates against the graph with
    /// [`validate_execution_submission`] or [`validate_execution_fault_gates`].
    pub fn new(
        gates: &'a [PcuExecutionSuccessGate],
        states: &'a mut [PcuExecutionNodeState],
    ) -> Self {
        states.fill(PcuExecutionNodeState::Pending);
        Self { gates, states }
    }

    /// Attempts to admit a node for submission and records it as running on success.
    ///
    /// Callers must serialize this operation with completion updates for the same graph.
    /// Ordinary ordering dependencies still need their backend's usual completion/enqueue
    /// handling; this method enforces terminal success only for declared success gates.
    pub fn try_start(&mut self, node: usize) -> Option<PcuExecutionAdmission> {
        let state = self.states.get(node)?;
        if *state != PcuExecutionNodeState::Pending {
            return Some(PcuExecutionAdmission::AlreadyStarted);
        }
        let mut waiting = false;
        for gate in self.gates.iter().filter(|gate| gate.successor == node) {
            match self.states.get(gate.predecessor).copied()? {
                PcuExecutionNodeState::Succeeded => {}
                PcuExecutionNodeState::Failed | PcuExecutionNodeState::Cancelled => {
                    return Some(PcuExecutionAdmission::Blocked);
                }
                PcuExecutionNodeState::Pending | PcuExecutionNodeState::Running => waiting = true,
            }
        }
        if waiting {
            Some(PcuExecutionAdmission::Waiting)
        } else {
            self.states[node] = PcuExecutionNodeState::Running;
            Some(PcuExecutionAdmission::Started)
        }
    }

    /// Records a running node's confirmed terminal outcome; returns false for invalid transitions.
    ///
    /// `Cancelled` means device/backend work is confirmed quiescent, not merely that a
    /// cancellation request was issued. Resources must remain retained until that point.
    pub fn finish(&mut self, node: usize, outcome: PcuExecutionNodeState) -> bool {
        if !matches!(
            outcome,
            PcuExecutionNodeState::Succeeded
                | PcuExecutionNodeState::Failed
                | PcuExecutionNodeState::Cancelled
        ) || self.states.get(node) != Some(&PcuExecutionNodeState::Running)
        {
            return false;
        }
        self.states[node] = outcome;
        true
    }

    /// Marks a node that was never submitted as cancelled, such as a gate-blocked successor.
    pub fn cancel_pending(&mut self, node: usize) -> bool {
        if self.states.get(node) != Some(&PcuExecutionNodeState::Pending) {
            return false;
        }
        self.states[node] = PcuExecutionNodeState::Cancelled;
        true
    }
}

/// Structural or access-ordering failure in a proposed execution graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PcuExecutionGraphError {
    InsufficientScratch {
        required: usize,
        provided: usize,
    },
    DependencyNotEarlier {
        node: usize,
        dependency: usize,
    },
    DuplicateDependency {
        node: usize,
        dependency: usize,
    },
    ResourceOutOfRange {
        node: usize,
        resource: PcuExecutionResourceId,
    },
    DuplicateResourceUse {
        node: usize,
        resource: PcuExecutionResourceId,
    },
    UnorderedConflict {
        earlier: usize,
        later: usize,
        resource: PcuExecutionResourceId,
    },
    FaultFlagCountMismatch {
        required: usize,
        provided: usize,
    },
    SuccessGateWithoutDependency(PcuExecutionSuccessGate),
    DuplicateSuccessGate(PcuExecutionSuccessGate),
    FaultOutputNotSuccessGated {
        producer: usize,
        consumer: usize,
        resource: PcuExecutionResourceId,
    },
}

/// Checks a topologically ordered work list without allocating.
///
/// `reachable` needs at least one bool per node and is overwritten. Two reads may execute
/// concurrently; any pair containing a write to the same resource needs a dependency path.
/// Callers must give physically aliasing resources the same logical ID, or separately prove that
/// their ranges cannot overlap. Successful validation establishes ordering only: it does not
/// enqueue work, establish device completion, or authorize storage release.
///
/// # Errors
///
/// Returns the first malformed edge, resource use, or unordered access conflict.
pub fn validate_execution_graph(
    nodes: &[PcuExecutionNode<'_>],
    resource_count: u32,
    reachable: &mut [bool],
) -> Result<(), PcuExecutionGraphError> {
    if reachable.len() < nodes.len() {
        return Err(PcuExecutionGraphError::InsufficientScratch {
            required: nodes.len(),
            provided: reachable.len(),
        });
    }

    for (current, node) in nodes.iter().enumerate() {
        for (index, &dependency) in node.dependencies.iter().enumerate() {
            if dependency >= current {
                return Err(PcuExecutionGraphError::DependencyNotEarlier {
                    node: current,
                    dependency,
                });
            }
            if node.dependencies[..index].contains(&dependency) {
                return Err(PcuExecutionGraphError::DuplicateDependency {
                    node: current,
                    dependency,
                });
            }
        }
        for (index, use_) in node.resources.iter().enumerate() {
            if use_.resource.0 >= resource_count {
                return Err(PcuExecutionGraphError::ResourceOutOfRange {
                    node: current,
                    resource: use_.resource,
                });
            }
            if node.resources[..index]
                .iter()
                .any(|previous| previous.resource == use_.resource)
            {
                return Err(PcuExecutionGraphError::DuplicateResourceUse {
                    node: current,
                    resource: use_.resource,
                });
            }
        }

        reachable[..current].fill(false);
        for &dependency in node.dependencies {
            reachable[dependency] = true;
        }
        // Edges always point backward, so this reverse pass computes the transitive closure
        // of the current node without a heap or recursion.
        for predecessor in (0..current).rev() {
            if reachable[predecessor] {
                for &ancestor in nodes[predecessor].dependencies {
                    reachable[ancestor] = true;
                }
            }
        }
        for (earlier, prior) in nodes[..current].iter().enumerate() {
            if reachable[earlier] {
                continue;
            }
            for use_ in node.resources {
                if prior.resources.iter().any(|previous| {
                    previous.resource == use_.resource
                        && (previous.access != PcuMemoryAccess::ReadOnly
                            || use_.access != PcuMemoryAccess::ReadOnly)
                }) {
                    return Err(PcuExecutionGraphError::UnorderedConflict {
                        earlier,
                        later: current,
                        resource: use_.resource,
                    });
                }
            }
        }
    }
    Ok(())
}

/// Checks that uses of a possibly faulting node's writes wait for its terminal success.
///
/// Call [`validate_execution_graph`] first. `may_fault` has one entry per node and `scratch`
/// needs the same length. A success gate must also be a direct ordering dependency. Once an
/// edge waits for a producer's success, that guarantee propagates through later dependency
/// paths. A plain ordered enqueue is insufficient: a checked kernel can finish with an error
/// after its successor has already consumed an undefined output.
///
/// The backend must enforce every declared success gate at submission time. This validator
/// proves the declaration is sufficient for resource users; it does not observe completions or
/// turn an unchecked backend queue into a fault-aware scheduler.
///
/// # Errors
///
/// Returns a malformed gate or the first resource use not protected from a producer fault.
pub fn validate_execution_fault_gates(
    nodes: &[PcuExecutionNode<'_>],
    may_fault: &[bool],
    success_gates: &[PcuExecutionSuccessGate],
    scratch: &mut [bool],
) -> Result<(), PcuExecutionGraphError> {
    if may_fault.len() != nodes.len() {
        return Err(PcuExecutionGraphError::FaultFlagCountMismatch {
            required: nodes.len(),
            provided: may_fault.len(),
        });
    }
    if scratch.len() < nodes.len() {
        return Err(PcuExecutionGraphError::InsufficientScratch {
            required: nodes.len(),
            provided: scratch.len(),
        });
    }
    for (current, node) in nodes.iter().enumerate() {
        for &dependency in node.dependencies {
            if dependency >= current {
                return Err(PcuExecutionGraphError::DependencyNotEarlier {
                    node: current,
                    dependency,
                });
            }
        }
    }
    for (index, &gate) in success_gates.iter().enumerate() {
        if gate.successor >= nodes.len()
            || gate.predecessor >= gate.successor
            || !nodes[gate.successor]
                .dependencies
                .contains(&gate.predecessor)
        {
            return Err(PcuExecutionGraphError::SuccessGateWithoutDependency(gate));
        }
        if success_gates[..index].contains(&gate) {
            return Err(PcuExecutionGraphError::DuplicateSuccessGate(gate));
        }
    }
    for (producer, node) in nodes.iter().enumerate() {
        if !may_fault[producer]
            || !node
                .resources
                .iter()
                .any(|use_| use_.access != PcuMemoryAccess::ReadOnly)
        {
            continue;
        }
        scratch[..nodes.len()].fill(false);
        for consumer in producer + 1..nodes.len() {
            scratch[consumer] = nodes[consumer].dependencies.iter().any(|&dependency| {
                (dependency == producer
                    && success_gates.contains(&PcuExecutionSuccessGate {
                        predecessor: producer,
                        successor: consumer,
                    }))
                    || scratch[dependency]
            });
            for written in node
                .resources
                .iter()
                .filter(|use_| use_.access != PcuMemoryAccess::ReadOnly)
            {
                if nodes[consumer]
                    .resources
                    .iter()
                    .any(|use_| use_.resource == written.resource)
                    && !scratch[consumer]
                {
                    return Err(PcuExecutionGraphError::FaultOutputNotSuccessGated {
                        producer,
                        consumer,
                        resource: written.resource,
                    });
                }
            }
        }
    }
    Ok(())
}

/// Validates both resource ordering and fault-gate coverage for one execution graph.
///
/// This is the preferred validation entry point before constructing
/// [`PcuExecutionFaultGateState`]. The same scratch slice serves both checks.
///
/// # Errors
///
/// Returns the first resource-ordering, fault-gate, or scratch-capacity error.
pub fn validate_execution_submission(
    nodes: &[PcuExecutionNode<'_>],
    resource_count: u32,
    may_fault: &[bool],
    success_gates: &[PcuExecutionSuccessGate],
    scratch: &mut [bool],
) -> Result<(), PcuExecutionGraphError> {
    validate_execution_graph(nodes, resource_count, scratch)?;
    validate_execution_fault_gates(nodes, may_fault, success_gates, scratch)
}

#[cfg(test)]
mod tests {
    use super::*;

    const READ: PcuExecutionResourceUse = PcuExecutionResourceUse {
        resource: PcuExecutionResourceId(0),
        access: PcuMemoryAccess::ReadOnly,
    };
    const WRITE: PcuExecutionResourceUse = PcuExecutionResourceUse {
        resource: PcuExecutionResourceId(0),
        access: PcuMemoryAccess::WriteOnly,
    };

    #[test]
    fn transitive_dependencies_order_writes_without_direct_edges() {
        let nodes = [
            PcuExecutionNode {
                dependencies: &[],
                resources: &[WRITE],
            },
            PcuExecutionNode {
                dependencies: &[0],
                resources: &[],
            },
            PcuExecutionNode {
                dependencies: &[1],
                resources: &[READ],
            },
        ];
        assert_eq!(validate_execution_graph(&nodes, 1, &mut [false; 3]), Ok(()));
    }

    #[test]
    fn independent_readers_are_legal_but_unordered_write_is_rejected() {
        let nodes = [
            PcuExecutionNode {
                dependencies: &[],
                resources: &[READ],
            },
            PcuExecutionNode {
                dependencies: &[],
                resources: &[READ],
            },
            PcuExecutionNode {
                dependencies: &[0],
                resources: &[WRITE],
            },
        ];
        assert_eq!(
            validate_execution_graph(&nodes[..2], 1, &mut [false; 2]),
            Ok(())
        );
        assert_eq!(
            validate_execution_graph(&nodes, 1, &mut [false; 3]),
            Err(PcuExecutionGraphError::UnorderedConflict {
                earlier: 1,
                later: 2,
                resource: PcuExecutionResourceId(0),
            })
        );
    }

    #[test]
    fn malformed_edges_and_uses_fail_before_scheduling() {
        let forward = [PcuExecutionNode {
            dependencies: &[0],
            resources: &[],
        }];
        assert_eq!(
            validate_execution_graph(&forward, 1, &mut [false; 1]),
            Err(PcuExecutionGraphError::DependencyNotEarlier {
                node: 0,
                dependency: 0
            })
        );
        let duplicate = [
            PcuExecutionNode {
                dependencies: &[],
                resources: &[],
            },
            PcuExecutionNode {
                dependencies: &[0, 0],
                resources: &[],
            },
        ];
        assert_eq!(
            validate_execution_graph(&duplicate, 1, &mut [false; 2]),
            Err(PcuExecutionGraphError::DuplicateDependency {
                node: 1,
                dependency: 0
            })
        );
        let bad_resource = [PcuExecutionNode {
            dependencies: &[],
            resources: &[READ],
        }];
        assert_eq!(
            validate_execution_graph(&bad_resource, 0, &mut [false; 1]),
            Err(PcuExecutionGraphError::ResourceOutOfRange {
                node: 0,
                resource: PcuExecutionResourceId(0),
            })
        );
        assert_eq!(
            validate_execution_graph(&bad_resource, 1, &mut []),
            Err(PcuExecutionGraphError::InsufficientScratch {
                required: 1,
                provided: 0,
            })
        );
        let duplicate_resource = [PcuExecutionNode {
            dependencies: &[],
            resources: &[READ, WRITE],
        }];
        assert_eq!(
            validate_execution_graph(&duplicate_resource, 1, &mut [false; 1]),
            Err(PcuExecutionGraphError::DuplicateResourceUse {
                node: 0,
                resource: PcuExecutionResourceId(0),
            })
        );
    }

    #[test]
    fn faulting_writer_needs_terminal_success_before_output_reuse() {
        let nodes = [
            PcuExecutionNode {
                dependencies: &[],
                resources: &[WRITE],
            },
            PcuExecutionNode {
                dependencies: &[0],
                resources: &[],
            },
            PcuExecutionNode {
                dependencies: &[1],
                resources: &[READ],
            },
        ];
        let flags = [true, false, false];
        assert_eq!(validate_execution_graph(&nodes, 1, &mut [false; 3]), Ok(()));
        assert_eq!(
            validate_execution_fault_gates(&nodes, &flags, &[], &mut [false; 3]),
            Err(PcuExecutionGraphError::FaultOutputNotSuccessGated {
                producer: 0,
                consumer: 2,
                resource: PcuExecutionResourceId(0),
            })
        );
        let gated = [PcuExecutionSuccessGate {
            predecessor: 0,
            successor: 1,
        }];
        assert_eq!(
            validate_execution_fault_gates(&nodes, &flags, &gated, &mut [false; 3]),
            Ok(())
        );
    }

    #[test]
    fn success_gate_must_be_a_real_edge() {
        let nodes = [
            PcuExecutionNode {
                dependencies: &[],
                resources: &[WRITE],
            },
            PcuExecutionNode {
                dependencies: &[],
                resources: &[READ],
            },
        ];
        let gate = PcuExecutionSuccessGate {
            predecessor: 0,
            successor: 1,
        };
        assert_eq!(
            validate_execution_fault_gates(&nodes, &[true, false], &[gate], &mut [false; 2]),
            Err(PcuExecutionGraphError::SuccessGateWithoutDependency(gate))
        );
    }

    #[test]
    fn waiting_for_an_unprotected_intermediate_does_not_hide_producer_fault() {
        let nodes = [
            PcuExecutionNode {
                dependencies: &[],
                resources: &[WRITE],
            },
            PcuExecutionNode {
                dependencies: &[0],
                resources: &[],
            },
            PcuExecutionNode {
                dependencies: &[1],
                resources: &[READ],
            },
        ];
        let wrong_gate = PcuExecutionSuccessGate {
            predecessor: 1,
            successor: 2,
        };
        assert_eq!(
            validate_execution_fault_gates(
                &nodes,
                &[true, false, false],
                &[wrong_gate],
                &mut [false; 3],
            ),
            Err(PcuExecutionGraphError::FaultOutputNotSuccessGated {
                producer: 0,
                consumer: 2,
                resource: PcuExecutionResourceId(0),
            })
        );
    }

    #[test]
    fn runtime_gate_state_waits_for_success_and_blocks_failure_or_cancellation() {
        let gates = [PcuExecutionSuccessGate {
            predecessor: 0,
            successor: 1,
        }];
        let mut states = [PcuExecutionNodeState::Pending; 2];
        let mut tracker = PcuExecutionFaultGateState::new(&gates, &mut states);
        assert_eq!(tracker.try_start(1), Some(PcuExecutionAdmission::Waiting));
        assert_eq!(tracker.try_start(0), Some(PcuExecutionAdmission::Started));
        assert_eq!(
            tracker.try_start(0),
            Some(PcuExecutionAdmission::AlreadyStarted)
        );
        assert!(tracker.finish(0, PcuExecutionNodeState::Failed));
        assert_eq!(tracker.try_start(1), Some(PcuExecutionAdmission::Blocked));
        assert!(tracker.cancel_pending(1));
        assert!(!tracker.finish(1, PcuExecutionNodeState::Succeeded));
        assert_eq!(tracker.try_start(3), None);

        let mut states = [PcuExecutionNodeState::Pending; 2];
        let mut tracker = PcuExecutionFaultGateState::new(&gates, &mut states);
        assert_eq!(tracker.try_start(0), Some(PcuExecutionAdmission::Started));
        assert!(tracker.finish(0, PcuExecutionNodeState::Succeeded));
        assert_eq!(tracker.try_start(1), Some(PcuExecutionAdmission::Started));
        assert!(tracker.finish(1, PcuExecutionNodeState::Cancelled));
        assert_eq!(
            tracker.try_start(1),
            Some(PcuExecutionAdmission::AlreadyStarted)
        );
    }

    #[test]
    fn combined_submission_validation_checks_resource_order_and_gates() {
        let nodes = [
            PcuExecutionNode {
                dependencies: &[],
                resources: &[WRITE],
            },
            PcuExecutionNode {
                dependencies: &[0],
                resources: &[READ],
            },
        ];
        let gate = PcuExecutionSuccessGate {
            predecessor: 0,
            successor: 1,
        };
        assert_eq!(
            validate_execution_submission(&nodes, 1, &[true, false], &[gate], &mut [false; 2]),
            Ok(())
        );
        assert_eq!(
            validate_execution_submission(&nodes, 1, &[true, false], &[], &mut [false; 2]),
            Err(PcuExecutionGraphError::FaultOutputNotSuccessGated {
                producer: 0,
                consumer: 1,
                resource: PcuExecutionResourceId(0),
            })
        );
    }
}
