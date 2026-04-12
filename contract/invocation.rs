//! Driver-facing PCU invocation shape and typed binding vocabulary.
//!
//! This stays intentionally below orchestration policy. Backend choice, fallback preference, and
//! prepared-dispatch state belong to implementation/composition crates; the core contract layer
//! only describes what one PCU profile invocation looks like in the abstract.

use core::num::NonZeroU32;

use super::{
    PcuBindingRef,
    PcuKernel,
    PcuParameter,
    PcuParameterBinding,
    PcuParameterSlot,
    PcuParameterValue,
};

/// Invocation geometry for one profile dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PcuInvocationShape {
    threads: NonZeroU32,
}

impl PcuInvocationShape {
    /// Creates one checked invocation shape.
    #[must_use]
    pub const fn threads(threads: NonZeroU32) -> Self {
        Self { threads }
    }

    /// Returns the requested logical thread count.
    #[must_use]
    pub const fn thread_count(self) -> NonZeroU32 {
        self.threads
    }
}

/// One abstract invocation descriptor without backend-selection policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PcuInvocation<'a> {
    pub kernel: &'a PcuKernel<'a>,
    pub shape: PcuInvocationShape,
}

/// Caller-provided runtime-parameter bindings for one prepared PCU program unit.
#[derive(Debug, Clone, Copy)]
pub struct PcuInvocationParameters<'a> {
    pub bindings: &'a [PcuParameterBinding],
}

impl PcuInvocationParameters<'_> {
    /// Returns one empty runtime-parameter table.
    #[must_use]
    pub const fn empty() -> Self {
        Self { bindings: &[] }
    }

    /// Returns whether no runtime parameters are supplied.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.bindings.is_empty()
    }

    /// Returns one submit-time binding for the requested slot when present.
    #[must_use]
    pub fn binding(self, slot: PcuParameterSlot) -> Option<PcuParameterBinding> {
        self.bindings
            .iter()
            .copied()
            .find(|binding| binding.slot == slot)
    }

    /// Returns one submit-time runtime value for the requested slot when present.
    #[must_use]
    pub fn value(self, slot: PcuParameterSlot) -> Option<PcuParameterValue> {
        self.binding(slot).map(|binding| binding.value)
    }

    /// Returns whether these submit-time runtime parameters satisfy one declared parameter list
    /// exactly.
    #[must_use]
    pub fn validate_against(self, parameters: &[PcuParameter<'_>]) -> bool {
        for (index, parameter) in parameters.iter().enumerate() {
            if parameters[..index]
                .iter()
                .any(|existing| existing.slot == parameter.slot)
            {
                return false;
            }
            let Some(value) = self.value(parameter.slot) else {
                return false;
            };
            if !value.matches_type(parameter.value_type) {
                return false;
            }
        }

        for (index, binding) in self.bindings.iter().enumerate() {
            if self.bindings[..index]
                .iter()
                .any(|existing| existing.slot == binding.slot)
            {
                return false;
            }
            if !parameters
                .iter()
                .any(|parameter| parameter.slot == binding.slot)
            {
                return false;
            }
        }

        true
    }
}

/// Runtime-visible target for one invocation-time binding payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PcuInvocationTarget<'a> {
    Binding(PcuBindingRef),
    Port(&'a str),
}

/// Typed invocation-time data payload bound to one target.
#[derive(Debug)]
pub enum PcuInvocationBuffer<'a> {
    BytesIn(&'a [u8]),
    BytesOut(&'a mut [u8]),
    BytesInOut {
        input: &'a [u8],
        output: &'a mut [u8],
    },
    HalfWordsIn(&'a [u16]),
    HalfWordsOut(&'a mut [u16]),
    HalfWordsInOut {
        input: &'a [u16],
        output: &'a mut [u16],
    },
    WordsIn(&'a [u32]),
    WordsOut(&'a mut [u32]),
    WordsInOut {
        input: &'a [u32],
        output: &'a mut [u32],
    },
}

/// One invocation-time binding payload attached to one runtime target.
#[derive(Debug)]
pub struct PcuInvocationBinding<'a> {
    pub target: PcuInvocationTarget<'a>,
    pub buffer: PcuInvocationBuffer<'a>,
}

/// Runtime binding table for one prepared PCU program unit.
#[derive(Debug, Clone, Copy)]
pub struct PcuInvocationBindings<'a> {
    pub bindings: &'a [PcuInvocationBinding<'a>],
}

impl<'a> PcuInvocationBindings<'a> {
    /// Returns one empty invocation binding table.
    #[must_use]
    pub const fn empty() -> Self {
        Self { bindings: &[] }
    }

    /// Returns whether no runtime binding payloads are supplied.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.bindings.is_empty()
    }
}
