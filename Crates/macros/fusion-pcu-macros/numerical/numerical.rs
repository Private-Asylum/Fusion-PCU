//! Independent function-local numerical option parsing and cold capture emission.

use proc_macro2::TokenStream;
use quote::quote;
#[rustfmt::skip]
use syn::{
    Error,
    Ident,
    Path,
};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[allow(clippy::redundant_pub_crate)] // Parsing/emission crosses sibling modules, not the crate API.
pub(super) struct NumericalFlags {
    native_compound: Option<bool>,
    backend_precision: Option<bool>,
    deterministic: Option<bool>,
}

#[allow(clippy::redundant_pub_crate)] // Parent and owned companion share this private parser seam.
impl NumericalFlags {
    /// Returns false for flags belonging to another independent policy family.
    pub(super) fn parse_flag(&mut self, flag: &Ident) -> Result<bool, Error> {
        let (setting, enabled, family) = match flag.to_string().as_str() {
            "native_compound" => (&mut self.native_compound, true, "compound arithmetic"),
            "checked_compound" => (&mut self.native_compound, false, "compound arithmetic"),
            "backend_precision" => (&mut self.backend_precision, true, "precision"),
            "preserve_precision" => (&mut self.backend_precision, false, "precision"),
            "deterministic" => (&mut self.deterministic, true, "reproducibility"),
            "non_deterministic" => (&mut self.deterministic, false, "reproducibility"),
            _ => return Ok(false),
        };
        if let Some(previous) = setting {
            let kind = if *previous == enabled {
                "duplicate"
            } else {
                "conflicting"
            };
            return Err(Error::new(
                flag.span(),
                format!("{kind} `pcu` {family} flags"),
            ));
        }
        *setting = Some(enabled);
        Ok(true)
    }

    pub(super) const fn is_empty(self) -> bool {
        self.native_compound.is_none()
            && self.backend_precision.is_none()
            && self.deterministic.is_none()
    }

    pub(super) fn overrides(self, pcu: &Path) -> TokenStream {
        let compound = match self.native_compound {
            Some(true) => {
                quote! { ::core::option::Option::Some(#pcu::PcuCompoundArithmeticPolicy::BackendDefined) }
            }
            Some(false) => {
                quote! { ::core::option::Option::Some(#pcu::PcuCompoundArithmeticPolicy::Checked) }
            }
            None => quote! { ::core::option::Option::None },
        };
        let precision = match self.backend_precision {
            Some(true) => {
                quote! { ::core::option::Option::Some(#pcu::PcuPrecisionPolicy::BackendOptimized) }
            }
            Some(false) => {
                quote! { ::core::option::Option::Some(#pcu::PcuPrecisionPolicy::Preserve) }
            }
            None => quote! { ::core::option::Option::None },
        };
        let reproducibility = match self.deterministic {
            Some(true) => {
                quote! { ::core::option::Option::Some(#pcu::PcuReproducibility::PortableV1) }
            }
            Some(false) => {
                quote! { ::core::option::Option::Some(#pcu::PcuReproducibility::Unspecified) }
            }
            None => quote! { ::core::option::Option::None },
        };
        quote! {
            #pcu::PcuNumericalOverrides {
                compound_arithmetic: #compound,
                precision: #precision,
                reproducibility: #reproducibility,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::NumericalFlags;
    #[rustfmt::skip]
    use crate::{
        PcuDispatchArgs,
        PcuScalarHelperArgs,
        expand_pcu_dispatch,
    };

    #[test]
    fn independent_axes_parse_without_implying_strictness() {
        for strict in ["strict", "non_strict"] {
            for compound in ["native_compound", "checked_compound"] {
                for precision in ["backend_precision", "preserve_precision"] {
                    for reproduction in ["deterministic", "non_deterministic"] {
                        let flags = format!(
                            "flag({strict}),flag({compound}),flag({precision}),flag({reproduction})"
                        );
                        let owned = syn::parse_str::<PcuScalarHelperArgs>(&flags).unwrap();
                        let dispatch =
                            syn::parse_str::<PcuDispatchArgs>(&format!("invocations=1,{flags}"))
                                .unwrap();
                        assert_eq!(owned.numerical_options, dispatch.numerical_options);
                        assert_eq!(owned.numerical_mode, Some(strict == "strict"));
                        assert_eq!(
                            owned.numerical_options.native_compound,
                            Some(compound == "native_compound")
                        );
                        assert_eq!(
                            owned.numerical_options.backend_precision,
                            Some(precision == "backend_precision")
                        );
                        assert_eq!(
                            owned.numerical_options.deterministic,
                            Some(reproduction == "deterministic")
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn each_axis_rejects_duplicate_and_conflicting_flags() {
        for (first, opposite) in [
            ("native_compound", "checked_compound"),
            ("backend_precision", "preserve_precision"),
            ("deterministic", "non_deterministic"),
        ] {
            for (lhs, rhs) in [
                (first, first),
                (opposite, opposite),
                (first, opposite),
                (opposite, first),
            ] {
                let flags = format!("flag({lhs}),flag({rhs})");
                assert!(syn::parse_str::<PcuScalarHelperArgs>(&flags).is_err());
                assert!(
                    syn::parse_str::<PcuDispatchArgs>(&format!("invocations=1,{flags}")).is_err()
                );
            }
        }
    }

    #[test]
    fn empty_flags_emit_inheritance_and_selected_flags_emit_only_their_axis() {
        let path = syn::parse_quote!(::fusion_pcu);
        let inherited = NumericalFlags::default().overrides(&path).to_string();
        assert_eq!(inherited.matches("None").count(), 3);
        let selected = syn::parse_str::<PcuScalarHelperArgs>("flag(deterministic)").unwrap();
        let emitted = selected.numerical_options.overrides(&path).to_string();
        assert!(emitted.contains("PortableV1"));
        assert_eq!(emitted.matches("None").count(), 2);
    }

    #[test]
    fn invocation_flags_reject_unimplemented_requirements_before_lowering() {
        let source = syn::parse_quote! {
            fn negate(input: &[f32], output: &mut [f32]) {
                let id = pcu::context::global_invocation_id();
                output[id] = -input[id];
            }
        };
        for flag in ["native_compound", "backend_precision", "deterministic"] {
            let args =
                syn::parse_str::<PcuDispatchArgs>(&format!("invocations=4,flag({flag})")).unwrap();
            let error = expand_pcu_dispatch(args, &source).unwrap_err().to_string();
            assert!(error.contains("owned tensor"), "{error}");
        }
    }
}
