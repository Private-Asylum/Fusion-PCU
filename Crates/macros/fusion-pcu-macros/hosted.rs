//! Direct typed entries for the global hosted dispatcher.

use proc_macro2::TokenStream;
#[rustfmt::skip]
use quote::{
    format_ident,
    quote,
};
#[rustfmt::skip]
use syn::{
    GenericParam,
    Type,
};
use super::prepared::Input;

#[allow(clippy::too_many_lines)]
pub fn generate(input: &Input<'_>) -> TokenStream {
    let pcu = input.pcu;
    let function = input.function;
    let ident = input.function_ident;
    let vis = input.visibility;
    let attributes = &function.attrs;
    // Generated argument adapters must mention typed, deliberately unread parameters.
    // The author's underscore convention remains valid; only generated wrappers need this allow.
    let unread_underscore_allow = input
        .arguments
        .iter()
        .any(|arg| !arg.access.used() && arg.ident.to_string().starts_with('_'))
        .then(|| quote! { #[allow(clippy::used_underscore_binding)] });
    let generics = &function.sig.generics;
    let where_clause = &generics.where_clause;
    let fresh_ident = |base: &str| {
        let mut ident = format_ident!("{base}");
        while input.arguments.iter().any(|arg| arg.ident == ident)
            || generics.params.iter().any(|param| match param {
                GenericParam::Type(param) => param.ident == ident,
                GenericParam::Const(param) => param.ident == ident,
                GenericParam::Lifetime(param) => param.lifetime.ident == ident,
            })
        {
            ident = format_ident!("_{ident}");
        }
        ident
    };
    let site_ident = fresh_ident("__PCU_SITE");
    let args_ident = fresh_ident("__pcu_arguments");
    let context_ident = fresh_ident("__pcu_context");
    let bindings_ident = fresh_ident("__pcu_bindings");
    let builder_ident = fresh_ident("__pcu_builder");
    let policy_ident = format_ident!("__pcu_float_underflow_policy");
    let range_policy_ident = format_ident!("__pcu_float_range_policy");
    let generic_args = input.generic_arguments;
    let bindings_call = &input.bindings_call;
    let policy_builder_ident = &input.policy_builder_ident;
    let policy_builder_call = if generic_args.is_empty() {
        quote! { #policy_builder_ident(&#bindings_ident, #policy_ident, #range_policy_ident, #context_ident.numerical_requirements()) }
    } else {
        quote! { #policy_builder_ident::<#(#generic_args),*>(&#bindings_ident, #policy_ident, #range_policy_ident, #context_ident.numerical_requirements()) }
    };
    let underflow_policy = input.explicit_policy.as_ref().map_or_else(
        || quote! { #context_ident.float_underflow_policy() },
        |policy| quote! { #policy },
    );
    let range_policy = input.explicit_range_policy.as_ref().map_or_else(
        || quote! { #context_ident.range_policy() },
        |policy| quote! { #policy },
    );
    let range_profile_guard = if input.supports_range_policy {
        quote! {}
    } else {
        quote! {
            if #context_ident.range_policy() == #pcu::PcuRangePolicy::Clamp {
                return ::core::result::Result::Err(
                    #pcu::global::PcuExecutionError::UnsupportedRangePolicy
                );
            }
        }
    };
    let marker = fresh_ident("__PcuSpecialization");
    let marker_type = if generic_args.is_empty() {
        quote! { #marker }
    } else {
        quote! { #marker<#(#generic_args),*> }
    };

    let mut direct_inputs = function.sig.inputs.clone();
    let mut conversions = Vec::new();
    for (input_arg, arg) in direct_inputs.iter_mut().zip(input.arguments) {
        let syn::FnArg::Typed(typed) = input_arg else {
            unreachable!("receivers are rejected")
        };
        let syn::Pat::Ident(pattern) = typed.pat.as_ref() else {
            unreachable!("binding patterns are validated")
        };
        let name = &pattern.ident;
        let scalar = &arg.scalar;
        let shape = source_shape(pcu, &arg.ty, arg.scalar_reference);
        let trait_name = if arg.access.writable() {
            quote! { PcuWriteStorage }
        } else {
            quote! { PcuReadStorage }
        };
        let Type::Reference(reference) = typed.ty.as_mut() else {
            unreachable!("validated PCU arguments are references")
        };
        *reference.elem = syn::parse_quote!((impl #pcu::#trait_name<#scalar, #shape> + ?Sized));
        let slot = arg.binding;
        let target = quote! { #pcu::PcuBindingRef::new(0, #slot) };
        if arg.access.used() {
            conversions.push(quote! {
                #pcu::#trait_name::as_pcu_call_argument(#name, #target)
                    .map_err(#pcu::global::argument_error)?
            });
        } else {
            // Validated lowering emitted no access. Preserve the source declaration,
            // without probing an irrelevant owner's storage or session.
            let metadata = if arg.access.writable() {
                quote! { unused_read_write }
            } else {
                quote! { unused_read }
            };
            conversions.push(quote! {{
                let _ = #name;
                #pcu::PcuCallArgument::#metadata::<#scalar>(#target)
            }});
        }
    }
    let count = input.arguments.len();
    let generic_params = generics
        .params
        .iter()
        .filter(|param| !matches!(param, GenericParam::Lifetime(_)))
        .collect::<syn::punctuated::Punctuated<_, syn::Token![,]>>();
    let types = generic_params
        .iter()
        .filter_map(|param| match param {
            GenericParam::Type(p) => Some(&p.ident),
            _ => None,
        })
        .collect::<Vec<_>>();
    let marker_definition = if generic_params.is_empty() {
        quote! { struct #marker; }
    } else {
        quote! { struct #marker<#generic_params>(::core::marker::PhantomData<fn() -> (#(#types,)*)>); }
    };

    quote! {
        #(#attributes)*
        #[allow(dead_code)]
        #unread_underscore_allow
        #vis fn #ident #generics (#direct_inputs) -> ::core::result::Result<(), #pcu::global::PcuExecutionError> #where_clause {
            #marker_definition
            static #site_ident: #pcu::global::PcuHostCallSite = #pcu::global::PcuHostCallSite::new();
            let #args_ident: [#pcu::PcuCallArgument<'_>; #count] = [#(#conversions),*];
            #pcu::global::call_arguments(
                &#site_ident,
                ::core::any::TypeId::of::<#marker_type>(),
                #args_ident,
                |#context_ident| {
                    #range_profile_guard
                    let #bindings_ident = #bindings_call;
                    let #policy_ident = #underflow_policy;
                    let #range_policy_ident = #range_policy;
                    let #builder_ident = #policy_builder_call.map_err(#pcu::global::build_error)?;
                    #builder_ident.with_ir(|ir| #context_ident.prepare(ir))
                },
            )
        }
    }
}

fn source_shape(pcu: &syn::Path, ty: &Type, scalar_reference: bool) -> TokenStream {
    if scalar_reference {
        return quote! { #pcu::ScalarShape };
    }
    let Type::Reference(reference) = ty else {
        unreachable!("validated reference argument")
    };
    match reference.elem.as_ref() {
        Type::Slice(_) => quote! { #pcu::SliceShape },
        Type::Array(array) => {
            if let Type::Array(inner) = array.elem.as_ref() {
                let rows = &array.len;
                let columns = &inner.len;
                quote! { #pcu::FixedMatrixShape<#rows, #columns> }
            } else {
                let length = &array.len;
                quote! { #pcu::FixedArrayShape<#length> }
            }
        }
        _ => unreachable!("validated PCU argument shape"),
    }
}

#[cfg(test)]
#[path = "hosted/tests/tests.rs"]
mod tests;
