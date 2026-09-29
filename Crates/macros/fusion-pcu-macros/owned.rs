//! Source lowering for owned resident tensor compositions.
//!
//! This profile treats a deliberately small subset of a function body as a graph description.
//! The original body is never run as a host fallback: only the generated capture companion is
//! called by the cold graph factory.

#[rustfmt::skip]
use quote::{
    format_ident,
    quote,
};
#[rustfmt::skip]
use syn::{
    Error,
    Expr,
    ExprCall,
    FnArg,
    ItemFn,
    Local,
    Pat,
    Path,
    ReturnType,
    Type,
};

#[path = "owned/consumption.rs"]
mod consumption;

/// Identifies an explicit resident-owner return profile before scalar-helper lowering.
pub fn declares_owned_tensor_return(output: &ReturnType) -> bool {
    let ReturnType::Type(_, ty) = output else {
        return false;
    };
    let Type::Path(result) = ty.as_ref() else {
        return false;
    };
    let Some(segment) = result.path.segments.last() else {
        return false;
    };
    if segment.ident != "Result" {
        return false;
    }
    let syn::PathArguments::AngleBracketed(arguments) = &segment.arguments else {
        return false;
    };
    matches!(arguments.args.first(), Some(syn::GenericArgument::Type(Type::Path(owner)))
        if owner.path.segments.last().is_some_and(|segment| segment.ident == "PcuTensor"))
}

/// Expand a homogeneous scalar composition over slices, fixed arrays, or fixed matrices.
pub fn expand_owned_return(
    function: &ItemFn,
    crate_path: &Path,
) -> Result<proc_macro2::TokenStream, Error> {
    let (parameters, const_params) = validate_owned_function(function)?;
    let input_idents = parameters
        .iter()
        .map(|parameter| parameter.ident.clone())
        .collect::<Vec<_>>();
    let input_modes = parameters
        .iter()
        .map(|parameter| parameter.mode)
        .collect::<Vec<_>>();
    let const_args = const_params
        .iter()
        .map(|parameter| parameter.ident.clone())
        .collect::<Vec<_>>();
    let type_args = function
        .sig
        .generics
        .type_params()
        .map(|parameter| parameter.ident.clone())
        .collect::<Vec<_>>();
    let companion_crate_path = super::rebase_companion_path(crate_path.clone());
    let mut source_names = source_binding_names(function);
    source_names.extend(input_idents.iter().cloned());
    source_names.extend(const_args.iter().cloned());
    source_names.extend(type_args.iter().cloned());
    let capture_ident = fresh_ident(
        "__pcu_capture",
        &input_idents[0],
        &source_names.iter().collect::<Vec<_>>(),
    );
    let parsed = parse_program(
        function,
        &input_idents,
        &input_modes,
        &const_args,
        &type_args,
        &capture_ident,
        &companion_crate_path,
    )?;
    let mut generated_reserved = source_names.iter().collect::<Vec<_>>();
    generated_reserved.push(&capture_ident);
    let wrapper_marker = fresh_ident(
        "__PcuOwnedCaptureMarker",
        &input_idents[0],
        &generated_reserved,
    );
    generated_reserved.push(&wrapper_marker);
    let site_ident = fresh_ident(
        "__PCU_OWNED_TENSOR_SITE",
        &input_idents[0],
        &generated_reserved,
    );
    generated_reserved.push(&site_ident);
    let source_ident = fresh_ident("__pcu_owned_source", &input_idents[0], &generated_reserved);
    generated_reserved.push(&source_ident);
    let entry_capture_ident =
        fresh_ident("__pcu_entry_capture", &input_idents[0], &generated_reserved);
    let capture_marker = fresh_ident(
        "__PcuCaptureRecursionMarker",
        &input_idents[0],
        &generated_reserved,
    );
    let emission = OwnedEmission {
        function,
        crate_path: &companion_crate_path,
        parameters: &parameters,
        capture_ident: &capture_ident,
        wrapper_marker: &wrapper_marker,
        capture_marker: &capture_marker,
        site_ident: &site_ident,
        source_ident: &source_ident,
        entry_capture_ident: &entry_capture_ident,
        program: &parsed,
    };
    Ok(emit_owned_items(&emission))
}

struct OwnedEmission<'a> {
    function: &'a ItemFn,
    crate_path: &'a Path,
    parameters: &'a [SourceParameter],
    capture_ident: &'a syn::Ident,
    wrapper_marker: &'a syn::Ident,
    capture_marker: &'a syn::Ident,
    site_ident: &'a syn::Ident,
    source_ident: &'a syn::Ident,
    entry_capture_ident: &'a syn::Ident,
    program: &'a CapturedProgram,
}

// This builds one wrapper and its matching cold companion from the same names and signatures;
// splitting the quote would duplicate the tightly coupled specialization and visibility data.
#[allow(clippy::cognitive_complexity, clippy::too_many_lines)]
fn emit_owned_items(emission: &OwnedEmission<'_>) -> proc_macro2::TokenStream {
    let OwnedEmission {
        function,
        crate_path: companion_crate_path,
        parameters,
        capture_ident,
        wrapper_marker,
        capture_marker,
        site_ident,
        source_ident,
        entry_capture_ident,
        program: parsed,
    } = emission;
    let input_count = parameters.len();
    // Signature validation guarantees the concrete scalar and matching input profile.
    let scalar = owned_return_scalar(&function.sig.output).expect("validated owned scalar");
    let consumed_parameter = parameters
        .iter()
        .find(|parameter| parameter.mode == SourceMode::ConsumedResident);
    let sole_consumed_parameter = (parameters.len() == 1)
        .then_some(consumed_parameter)
        .flatten();
    let all_consumed_parameters = parameters.len() >= 2
        && parameters
            .iter()
            .all(|parameter| parameter.mode == SourceMode::ConsumedResident);
    let mixed_consumed_parameters =
        parameters.len() > 2 && consumed_parameter.is_some() && !all_consumed_parameters;
    let pair_consumed_parameter = if parameters.len() == 2
        && parameters
            .iter()
            .filter(|parameter| parameter.mode == SourceMode::ConsumedResident)
            .count()
            == 1
    {
        consumed_parameter.and_then(|donor| {
            let other = parameters
                .iter()
                .find(|parameter| parameter.ident != donor.ident)?;
            matches!(
                other.mode,
                SourceMode::Borrowed | SourceMode::BorrowedResident
            )
            .then(|| {
                let donor_index = parameters
                    .iter()
                    .position(|parameter| parameter.ident == donor.ident)
                    .expect("donor belongs to parameter list");
                (donor, other, donor_index)
            })
        })
    } else {
        None
    };
    let call_entry = format_ident!(
        "{}",
        if sole_consumed_parameter.is_some() {
            "call_consumed_tensor_capture"
        } else if all_consumed_parameters {
            "call_consumed_owners_tensor_capture"
        } else if pair_consumed_parameter.is_some() {
            "call_consumed_pair_tensor_capture"
        } else if mixed_consumed_parameters {
            "call_mixed_consumed_tensor_capture"
        } else {
            "call_owned_tensor_capture"
        }
    );
    let input_arguments = parameters.iter().map(|parameter| {
        let ident = &parameter.ident;
        match parameter.mode {
            SourceMode::Borrowed => {
                let shape_bound = parameter.shape.trait_argument(companion_crate_path);
                quote! {
                    #ident: &(impl #companion_crate_path::global::PcuTensorSource<#scalar, #shape_bound> + ?Sized)
                }
            }
            SourceMode::BorrowedResident => quote! {
                #ident: &#companion_crate_path::global::PcuTensor<#scalar>
            },
            SourceMode::ConsumedResident => quote! {
                #ident: #companion_crate_path::global::PcuTensor<#scalar>
            },
        }
    });
    let source_setup = if sole_consumed_parameter.is_some() || all_consumed_parameters {
        quote! {}
    } else if let Some((_, other, _)) = pair_consumed_parameter {
        let ident = &other.ident;
        let shape = other.shape.trait_argument(companion_crate_path);
        quote! {
            let #source_ident = #companion_crate_path::global::PcuTensorSource::<#scalar, #shape>::as_tensor_source(#ident)
                .map_err(#companion_crate_path::global::argument_error)?;
        }
    } else if mixed_consumed_parameters {
        let sources = parameters.iter().map(|parameter| {
            let ident = &parameter.ident;
            if parameter.mode == SourceMode::ConsumedResident {
                quote! {
                    #companion_crate_path::global::PcuTensorCallInput::Consumed(#ident)
                }
            } else {
                let shape = parameter.shape.trait_argument(companion_crate_path);
                quote! {
                    #companion_crate_path::global::PcuTensorCallInput::Borrowed(
                        #companion_crate_path::global::PcuTensorSource::<#scalar, #shape>::as_tensor_source(#ident)
                            .map_err(#companion_crate_path::global::argument_error)?
                    )
                }
            }
        });
        quote! { let #source_ident = [#(#sources),*]; }
    } else {
        let sources = parameters.iter().map(|parameter| {
            let ident = &parameter.ident;
            let shape = parameter.shape.trait_argument(companion_crate_path);
            let source = if parameter.mode == SourceMode::ConsumedResident {
                quote! { &#ident }
            } else {
                quote! { #ident }
            };
            quote! {
                #companion_crate_path::global::PcuTensorSource::<#scalar, #shape>::as_tensor_source(#source)
                    .map_err(#companion_crate_path::global::argument_error)?
            }
        });
        quote! { let #source_ident = [#(#sources),*]; }
    };
    let entry_arguments = if all_consumed_parameters {
        let owners = parameters.iter().map(|parameter| &parameter.ident);
        quote! { [#(#owners),*] }
    } else if let Some((parameter, _, donor_index)) = pair_consumed_parameter {
        let donor = &parameter.ident;
        quote! { #donor, #source_ident, #donor_index }
    } else {
        sole_consumed_parameter.map_or_else(
            || quote! { #source_ident },
            |parameter| {
                let input = &parameter.ident;
                quote! { #input }
            },
        )
    };
    let input_type = quote! { #companion_crate_path::global::PcuTensorGraphValue<#scalar> };
    let capture_parameters = parameters.iter().map(|parameter| {
        let ident = &parameter.ident;
        match parameter.mode {
            SourceMode::Borrowed | SourceMode::BorrowedResident => {
                quote! { #ident: #input_type }
            }
            SourceMode::ConsumedResident => quote! {
                #ident: #companion_crate_path::global::PcuTensorGraphOwner<#scalar>
            },
        }
    });
    let capture_values = (0..input_count)
        .map(syn::Index::from)
        .zip(parameters.iter())
        .map(|(index, parameter)| match parameter.mode {
            SourceMode::Borrowed | SourceMode::BorrowedResident => {
                quote! { __pcu_inputs[#index] }
            }
            SourceMode::ConsumedResident => quote! {
                #companion_crate_path::global::PcuTensorGraphOwner::from_graph_value(__pcu_inputs[#index])
            },
        });
    let function_ident = &function.sig.ident;
    let function_visibility = &function.vis;
    let companion_cfg = function
        .attrs
        .iter()
        .flat_map(super::project_companion_cfg)
        .collect::<Vec<_>>();
    let wrapper_attrs = function.attrs.iter().filter(|attribute| {
        attribute
            .path()
            .segments
            .last()
            .is_none_or(|segment| segment.ident != "pcu")
    });
    let output = &function.sig.output;
    let generic_params = &function.sig.generics.params;
    let where_clause = &function.sig.generics.where_clause;
    let generic_definitions = generic_params.iter().collect::<Vec<_>>();
    let generic_arguments = generic_params
        .iter()
        .map(|parameter| match parameter {
            syn::GenericParam::Type(parameter) => {
                let ident = &parameter.ident;
                quote! { #ident }
            }
            syn::GenericParam::Const(parameter) => {
                let ident = &parameter.ident;
                quote! { #ident }
            }
            syn::GenericParam::Lifetime(_) => unreachable!("validated signature"),
        })
        .collect::<Vec<_>>();
    let generic_declaration = if generic_params.is_empty() {
        quote! {}
    } else {
        quote! { <#(#generic_definitions),*> }
    };
    let marker_uses = generic_params
        .iter()
        .map(|parameter| match parameter {
            syn::GenericParam::Type(parameter) => {
                let ident = &parameter.ident;
                quote! { #ident }
            }
            syn::GenericParam::Const(parameter) => {
                let ident = &parameter.ident;
                quote! { [(); #ident] }
            }
            syn::GenericParam::Lifetime(_) => unreachable!("validated signature"),
        })
        .collect::<Vec<_>>();
    let marker_phantom = if generic_params.is_empty() {
        quote! {}
    } else {
        quote! { (::core::marker::PhantomData<fn() -> (#(#marker_uses),*)>,) }
    };
    let generic_marker_definition = if generic_params.is_empty() {
        quote! { struct #wrapper_marker; }
    } else {
        quote! { struct #wrapper_marker <#(#generic_definitions),*> #marker_phantom #where_clause; }
    };
    let marker_type = if generic_arguments.is_empty() {
        quote! { #wrapper_marker }
    } else {
        quote! { #wrapper_marker <#(#generic_arguments),*> }
    };
    let capture_marker_definition = if generic_params.is_empty() {
        quote! { struct #capture_marker; }
    } else {
        quote! { struct #capture_marker <#(#generic_definitions),*> #marker_phantom #where_clause; }
    };
    let capture_marker_type = if generic_arguments.is_empty() {
        quote! { #capture_marker }
    } else {
        quote! { #capture_marker <#(#generic_arguments),*> }
    };
    let capture_generic_args = if generic_arguments.is_empty() {
        quote! {}
    } else {
        quote! { ::<#(#generic_arguments),*> }
    };
    let input_shape_checks = parameters.iter().map(|parameter| {
        let ident = &parameter.ident;
        let value = match parameter.mode {
            SourceMode::Borrowed | SourceMode::BorrowedResident => quote! { #ident },
            SourceMode::ConsumedResident => quote! { #ident.borrowed_graph_value() },
        };
        match &parameter.shape {
            SourceShape::Slice => quote! { #capture_ident.require_rank(#value, 1)?; },
            SourceShape::DynamicResident => quote! {},
            shape => {
                let descriptor = shape
                    .capture_descriptor(companion_crate_path)
                    .expect("fixed source shapes have capture descriptors");
                quote! { #capture_ident.require_shape(#value, #descriptor)?; }
            }
        }
    });
    let capture_body = &parsed.statements;
    let output_tokens = &parsed.output;
    quote! {
        #(#wrapper_attrs)*
        // The wrapper must inspect every source argument to preserve its metadata, including
        // authored `_unused` parameters that the captured graph later prunes.
        #[allow(dead_code, clippy::used_underscore_binding)]
        #function_visibility fn #function_ident #generic_declaration (
            #(#input_arguments),*
        ) #output #where_clause {
            #generic_marker_definition
            static #site_ident: #companion_crate_path::global::PcuHostCallSite =
                #companion_crate_path::global::PcuHostCallSite::new();
            #source_setup
            #companion_crate_path::global::#call_entry(
                &#site_ident,
                ::core::any::TypeId::of::<#marker_type>(),
                #entry_arguments,
                #function_ident::__pcu_capture_entry #capture_generic_args,
            )
        }

        #[doc(hidden)]
        #(#companion_cfg)*
        #function_visibility mod #function_ident {
            #[allow(unused_imports)]
            use super::*;

            #capture_marker_definition

            pub fn __pcu_capture_entry #generic_declaration (
                #entry_capture_ident: &mut #companion_crate_path::global::PcuTensorGraphCapture,
                __pcu_inputs: [#input_type; #input_count],
            ) -> ::core::result::Result<#input_type, #companion_crate_path::PcuExecutionError> #where_clause {
                __pcu_capture #capture_generic_args (#entry_capture_ident, #(#capture_values),*)
            }

            // This cannot widen access beyond the companion module's original function visibility.
            // The module carries the source function's visibility; this child must be public
            // so a restricted module's permitted ancestors can reach the companion as well.
            // Authored underscore-prefixed formals are still checked here for their shape metadata.
            #[allow(clippy::used_underscore_binding)]
            pub fn __pcu_capture #generic_declaration (
                #capture_ident: &mut #companion_crate_path::global::PcuTensorGraphCapture,
                #(#capture_parameters),*
            ) -> ::core::result::Result<#input_type, #companion_crate_path::PcuExecutionError> #where_clause {
                #capture_ident.enter(::core::any::TypeId::of::<#capture_marker_type>())?;
                let __pcu_capture_result = (|| -> ::core::result::Result<
                    #input_type,
                    #companion_crate_path::PcuExecutionError,
                > {
                    #(#input_shape_checks)*
                    #(#capture_body)*
                    ::core::result::Result::Ok(#output_tokens)
                })();
                #capture_ident.leave();
                __pcu_capture_result
            }
        }
    }
}

#[derive(Clone)]
struct SourceParameter {
    ident: syn::Ident,
    shape: SourceShape,
    mode: SourceMode,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum SourceMode {
    Borrowed,
    BorrowedResident,
    ConsumedResident,
}

#[derive(Clone)]
enum SourceShape {
    Slice,
    DynamicResident,
    FixedArray(Expr),
    FixedMatrix { rows: Expr, columns: Expr },
}

impl SourceShape {
    fn trait_argument(&self, crate_path: &Path) -> proc_macro2::TokenStream {
        match self {
            Self::Slice => quote! { #crate_path::global::SliceShape },
            Self::DynamicResident => quote! { #crate_path::global::DynamicResidentShape },
            Self::FixedArray(length) => quote! { #crate_path::global::FixedArrayShape<#length> },
            Self::FixedMatrix { rows, columns } => {
                quote! { #crate_path::global::FixedMatrixShape<#rows, #columns> }
            }
        }
    }

    fn capture_descriptor(&self, crate_path: &Path) -> Option<proc_macro2::TokenStream> {
        match self {
            Self::Slice | Self::DynamicResident => None,
            Self::FixedArray(length) => {
                Some(quote! { #crate_path::global::PcuSourceShape::FixedArray { length: #length } })
            }
            Self::FixedMatrix { rows, columns } => Some(
                quote! { #crate_path::global::PcuSourceShape::FixedMatrix { rows: #rows, columns: #columns } },
            ),
        }
    }
}

fn validate_owned_function(
    function: &ItemFn,
) -> Result<(Vec<SourceParameter>, Vec<syn::ConstParam>), Error> {
    let const_params = validate_owned_signature(function)?;
    let const_names = const_params
        .iter()
        .map(|parameter| parameter.ident.clone())
        .collect::<Vec<_>>();
    if function.sig.inputs.is_empty() || function.sig.inputs.len() > 32 {
        return Err(Error::new_spanned(
            &function.sig.inputs,
            "owned tensor compositions require between 1 and 32 inputs with one homogeneous PCU scalar type",
        ));
    }
    let scalar = owned_return_scalar(&function.sig.output).expect("validated owned scalar");
    let parameters = validate_owned_parameters(function, &const_names, &scalar)?;
    if parameters
        .iter()
        .any(|parameter| parameter.mode == SourceMode::ConsumedResident)
    {
        let owners = parameters
            .iter()
            .filter(|parameter| parameter.mode == SourceMode::ConsumedResident)
            .map(|parameter| parameter.ident.clone())
            .collect::<Vec<_>>();
        consumption::validate_body(function, &owners)?;
    }
    Ok((parameters, const_params))
}

fn validate_owned_signature(function: &ItemFn) -> Result<Vec<syn::ConstParam>, Error> {
    if function.sig.asyncness.is_some()
        || function.sig.constness.is_some()
        || function.sig.unsafety.is_some()
        || function.sig.abi.is_some()
        || function.sig.generics.params.iter().any(|parameter| {
            !matches!(
                parameter,
                syn::GenericParam::Const(_) | syn::GenericParam::Type(_)
            )
        })
        || function.sig.variadic.is_some()
    {
        return Err(Error::new_spanned(
            &function.sig,
            "owned tensor compositions allow one scalar type parameter and const usize dimensions, and must be plain safe Rust functions",
        ));
    }
    let const_params = function
        .sig
        .generics
        .params
        .iter()
        .filter_map(|parameter| {
            let syn::GenericParam::Const(parameter) = parameter else {
                return None;
            };
            Some(parameter.clone())
        })
        .collect::<Vec<_>>();
    for parameter in &const_params {
        if !matches!(&parameter.ty, Type::Path(path) if path.path.is_ident("usize"))
            || parameter.default.is_some()
        {
            return Err(Error::new_spanned(
                parameter,
                "owned tensor dimensions must be non-defaulted `const usize` parameters",
            ));
        }
    }
    let scalar = owned_return_scalar(&function.sig.output);
    let type_params = function.sig.generics.type_params().collect::<Vec<_>>();
    let valid_scalar = match type_params.as_slice() {
        [] => scalar.is_some(),
        [parameter] => {
            scalar.as_ref() == Some(&parameter.ident)
                && parameter.default.is_none()
                && (parameter.bounds.iter().all(|bound| matches!(bound,
                    syn::TypeParamBound::Trait(bound)
                        if bound.path.segments.last().is_some_and(|segment| segment.ident == "PcuScalar")))
                    || parameter.bounds.is_empty())
                && (parameter.bounds.iter().any(|bound| matches!(bound,
                    syn::TypeParamBound::Trait(bound)
                        if bound.path.segments.last().is_some_and(|segment| segment.ident == "PcuScalar")))
                    || where_clause_has_scalar_bound(&function.sig.generics, &parameter.ident))
        }
        _ => false,
    };
    if !valid_scalar || !valid_scalar_where_clause(&function.sig.generics) {
        return Err(Error::new_spanned(
            &function.sig.output,
            "owned tensor composition must return a homogeneous scalar owner or a homogeneous `T: PcuScalar` owner",
        ));
    }
    Ok(const_params)
}

fn valid_scalar_where_clause(generics: &syn::Generics) -> bool {
    let Some(clause) = &generics.where_clause else {
        return true;
    };
    let Some(parameter) = generics.type_params().next() else {
        return false;
    };
    clause.predicates.iter().all(|predicate| {
        let syn::WherePredicate::Type(predicate) = predicate else {
            return false;
        };
        matches!(&predicate.bounded_ty, Type::Path(path) if path.path.is_ident(&parameter.ident))
            && predicate.bounds.iter().all(|bound| matches!(bound,
                syn::TypeParamBound::Trait(bound)
                    if bound.path.segments.last().is_some_and(|segment| segment.ident == "PcuScalar")))
    })
}

fn where_clause_has_scalar_bound(generics: &syn::Generics, ident: &syn::Ident) -> bool {
    generics.where_clause.as_ref().is_some_and(|clause| {
        clause.predicates.iter().any(|predicate| {
            matches!(predicate,
                syn::WherePredicate::Type(predicate)
                    if matches!(&predicate.bounded_ty, Type::Path(path) if path.path.is_ident(ident))
                        && predicate.bounds.iter().any(|bound| matches!(bound,
                            syn::TypeParamBound::Trait(bound)
                                if bound.path.segments.last().is_some_and(|segment| segment.ident == "PcuScalar"))))
        })
    })
}

fn validate_owned_parameters(
    function: &ItemFn,
    const_names: &[syn::Ident],
    scalar: &syn::Ident,
) -> Result<Vec<SourceParameter>, Error> {
    let mut parameters = Vec::with_capacity(function.sig.inputs.len());
    for argument in &function.sig.inputs {
        let FnArg::Typed(argument) = argument else {
            return Err(Error::new_spanned(
                argument,
                "owned tensor composition does not accept a self receiver",
            ));
        };
        if !argument.attrs.is_empty() {
            return Err(Error::new_spanned(
                argument,
                "owned tensor composition parameters do not accept attributes",
            ));
        }
        let Pat::Ident(pattern) = argument.pat.as_ref() else {
            return Err(Error::new_spanned(
                &argument.pat,
                "owned tensor composition requires a named input parameter",
            ));
        };
        if pattern.by_ref.is_some() || pattern.mutability.is_some() || pattern.subpat.is_some() {
            return Err(Error::new_spanned(
                &argument.pat,
                "owned tensor composition requires an unmodified named input parameter",
            ));
        }
        let (shape, mode) = if is_consumed_resident_type(argument.ty.as_ref(), scalar) {
            (SourceShape::DynamicResident, SourceMode::ConsumedResident)
        } else if is_borrowed_resident_type(argument.ty.as_ref(), scalar) {
            (SourceShape::DynamicResident, SourceMode::BorrowedResident)
        } else {
            (
                parse_source_shape(argument.ty.as_ref(), const_names, scalar).ok_or_else(|| {
                    Error::new_spanned(
                        argument,
                        "owned tensor inputs must be borrowed slices, fixed arrays, matrices, or one by-value PcuTensor matching the return scalar",
                    )
                })?,
                SourceMode::Borrowed,
            )
        };
        if parameters
            .iter()
            .any(|parameter: &SourceParameter| parameter.ident == pattern.ident)
        {
            return Err(Error::new_spanned(
                &argument.pat,
                "owned tensor composition input names must be unique",
            ));
        }
        parameters.push(SourceParameter {
            ident: pattern.ident.clone(),
            shape,
            mode,
        });
    }
    Ok(parameters)
}

fn is_consumed_resident_type(ty: &Type, scalar: &syn::Ident) -> bool {
    let Type::Path(path) = ty else {
        return false;
    };
    if path.qself.is_some() || path.path.leading_colon.is_some() {
        return false;
    }
    let Some(segment) = path.path.segments.last() else {
        return false;
    };
    if segment.ident != "PcuTensor" {
        return false;
    }
    let syn::PathArguments::AngleBracketed(arguments) = &segment.arguments else {
        return false;
    };
    arguments.args.len() == 1
        && matches!(arguments.args.first(), Some(syn::GenericArgument::Type(ty)) if is_scalar_type(ty, scalar))
}

fn is_borrowed_resident_type(ty: &Type, scalar: &syn::Ident) -> bool {
    let Type::Reference(reference) = ty else {
        return false;
    };
    reference.mutability.is_none()
        && reference.lifetime.is_none()
        && is_consumed_resident_type(reference.elem.as_ref(), scalar)
}

fn is_pcu_builtin(path: &Path, builtin: &str) -> bool {
    path.segments.len() == 2
        && path.segments[0].ident == "pcu"
        && matches!(path.segments[0].arguments, syn::PathArguments::None)
        && path.segments[1].ident == builtin
        && matches!(path.segments[1].arguments, syn::PathArguments::None)
}

fn unwrap_transparent(mut expression: &Expr) -> Result<&Expr, Error> {
    for _ in 0..64 {
        match expression {
            Expr::Paren(paren) if paren.attrs.is_empty() => expression = &paren.expr,
            Expr::Group(group) if group.attrs.is_empty() => expression = &group.expr,
            Expr::Paren(paren) => return Err(composition_body_error(paren)),
            Expr::Group(group) => return Err(composition_body_error(group)),
            _ => return Ok(expression),
        }
    }
    Err(composition_body_error(expression))
}

fn terminal_is_reference(mut expression: &Expr) -> bool {
    for _ in 0..64 {
        match expression {
            Expr::Paren(paren) if paren.attrs.is_empty() => expression = &paren.expr,
            Expr::Group(group) if group.attrs.is_empty() => expression = &group.expr,
            Expr::Reference(_) => return true,
            _ => return false,
        }
    }
    false
}

fn parse_source_shape(
    ty: &Type,
    const_names: &[syn::Ident],
    scalar: &syn::Ident,
) -> Option<SourceShape> {
    let Type::Reference(reference) = ty else {
        return None;
    };
    if reference.mutability.is_some() || reference.lifetime.is_some() {
        return None;
    }
    match reference.elem.as_ref() {
        Type::Slice(slice) if is_scalar_type(&slice.elem, scalar) => Some(SourceShape::Slice),
        Type::Array(array) if is_scalar_type(&array.elem, scalar) => Some(SourceShape::FixedArray(
            valid_const_expr(&array.len, const_names)?,
        )),
        Type::Array(outer) => {
            let Type::Array(inner) = outer.elem.as_ref() else {
                return None;
            };
            if !is_scalar_type(&inner.elem, scalar) {
                return None;
            }
            Some(SourceShape::FixedMatrix {
                rows: valid_const_expr(&outer.len, const_names)?,
                columns: valid_const_expr(&inner.len, const_names)?,
            })
        }
        _ => None,
    }
}

fn is_scalar_type(ty: &Type, scalar: &syn::Ident) -> bool {
    matches!(ty, Type::Path(path) if path.qself.is_none() && path.path.is_ident(scalar))
}

fn valid_const_expr(expression: &Expr, const_names: &[syn::Ident]) -> Option<Expr> {
    match expression {
        Expr::Lit(literal) if matches!(literal.lit, syn::Lit::Int(_)) => Some(expression.clone()),
        Expr::Path(path)
            if path.qself.is_none()
                && path
                    .path
                    .get_ident()
                    .is_some_and(|ident| const_names.contains(ident)) =>
        {
            Some(expression.clone())
        }
        _ => None,
    }
}

struct CapturedProgram {
    statements: Vec<proc_macro2::TokenStream>,
    output: proc_macro2::TokenStream,
}

#[derive(Clone)]
enum Value {
    Input(usize),
    BorrowedInput(usize),
    OwnerAlias { ident: syn::Ident, owner_id: usize },
    BorrowAlias { ident: syn::Ident, owner_id: usize },
}

enum Operation {
    Identity,
    Relu,
    Add,
    Sub,
    Mul,
    Matmul,
    Helper(Path),
}

fn parse_program(
    function: &ItemFn,
    inputs: &[syn::Ident],
    input_modes: &[SourceMode],
    const_args: &[syn::Ident],
    type_args: &[syn::Ident],
    capture: &syn::Ident,
    crate_path: &Path,
) -> Result<CapturedProgram, Error> {
    if function.block.stmts.is_empty() {
        return Err(composition_body_error(&function.block));
    }
    let (statements, terminal) = function
        .block
        .stmts
        .split_at(function.block.stmts.len() - 1);
    let mut state = ProgramState::new(
        function,
        inputs,
        input_modes,
        const_args,
        type_args,
        capture,
        crate_path,
    );
    for statement in statements {
        let syn::Stmt::Local(local) = statement else {
            return Err(composition_body_error(statement));
        };
        state.bind_local(local)?;
    }
    let [syn::Stmt::Expr(expression, None)] = terminal else {
        return Err(composition_body_error(&function.block));
    };
    let expression = unwrap_result_return(expression)?;
    if terminal_is_reference(expression) {
        return Err(Error::new_spanned(
            expression,
            "an owned tensor result cannot return a reference",
        ));
    }
    let output = state.lower(expression)?;
    match &output {
        Value::Input(index) if input_modes[*index] != SourceMode::ConsumedResident => {
            return Err(Error::new_spanned(
                expression,
                "an owned tensor result cannot return a borrowed input",
            ));
        }
        Value::BorrowedInput(_) | Value::BorrowAlias { .. } => {
            return Err(Error::new_spanned(
                expression,
                "an owned tensor result cannot return a borrowed value",
            ));
        }
        Value::Input(_) | Value::OwnerAlias { .. } => {}
    }
    let output = value_tokens(&output, inputs, input_modes, false);
    Ok(CapturedProgram {
        statements: state.emitted,
        output,
    })
}

struct ProgramState<'a> {
    inputs: &'a [syn::Ident],
    input_modes: &'a [SourceMode],
    const_args: &'a [syn::Ident],
    type_args: &'a [syn::Ident],
    capture: &'a syn::Ident,
    crate_path: &'a Path,
    bindings: Vec<(syn::Ident, Value)>,
    emitted: Vec<proc_macro2::TokenStream>,
    generated: Vec<syn::Ident>,
    next_owner_id: usize,
}

impl<'a> ProgramState<'a> {
    fn new(
        function: &ItemFn,
        inputs: &'a [syn::Ident],
        input_modes: &'a [SourceMode],
        const_args: &'a [syn::Ident],
        type_args: &'a [syn::Ident],
        capture: &'a syn::Ident,
        crate_path: &'a Path,
    ) -> Self {
        let mut generated = source_binding_names(function);
        generated.extend_from_slice(inputs);
        generated.extend_from_slice(const_args);
        generated.extend_from_slice(type_args);
        generated.push(capture.clone());
        Self {
            inputs,
            input_modes,
            const_args,
            type_args,
            capture,
            crate_path,
            bindings: Vec::new(),
            emitted: Vec::new(),
            generated,
            next_owner_id: inputs.len(),
        }
    }

    fn bind_local(&mut self, local: &Local) -> Result<(), Error> {
        let (name, expression) = local_binding(local)?;
        if self.const_args.iter().any(|constant| constant == name) {
            return Err(Error::new_spanned(
                &local.pat,
                "owned tensor composition locals cannot shadow const generic parameters",
            ));
        }
        if let Some(source) =
            owner_move_source(expression, self.inputs, self.input_modes, &self.bindings)?
        {
            let local_ident = self.fresh_local("__pcu_owner_alias");
            self.emitted.push(quote! { let #local_ident = #source; });
            let owner_id = self.next_owner_id;
            self.next_owner_id += 1;
            self.bindings.push((
                name.clone(),
                Value::OwnerAlias {
                    ident: local_ident,
                    owner_id,
                },
            ));
            return Ok(());
        }
        if let Some((source, owner_id)) =
            owner_borrow_source(expression, self.inputs, self.input_modes, &self.bindings)?
        {
            let local_ident = self.fresh_local("__pcu_owner_view");
            self.emitted.push(quote! { let #local_ident = &#source; });
            self.bindings.push((
                name.clone(),
                Value::BorrowAlias {
                    ident: local_ident,
                    owner_id,
                },
            ));
            return Ok(());
        }
        let value = self.lower(expression)?;
        self.bindings.push((name.clone(), value));
        Ok(())
    }

    fn lower(&mut self, expression: &Expr) -> Result<Value, Error> {
        LoweringState {
            inputs: self.inputs,
            input_modes: self.input_modes,
            const_args: self.const_args,
            type_args: self.type_args,
            bindings: &self.bindings,
            capture: self.capture,
            crate_path: self.crate_path,
            generated: &mut self.generated,
            emitted: &mut self.emitted,
            next_owner_id: &mut self.next_owner_id,
        }
        .lower(expression, 0)
    }

    fn fresh_local(&mut self, base: &str) -> syn::Ident {
        let reserved = self.generated.iter().collect::<Vec<_>>();
        let local = fresh_ident(base, &self.inputs[0], &reserved);
        self.generated.push(local.clone());
        local
    }
}

fn owner_move_source(
    expression: &Expr,
    inputs: &[syn::Ident],
    input_modes: &[SourceMode],
    bindings: &[(syn::Ident, Value)],
) -> Result<Option<syn::Ident>, Error> {
    let Expr::Path(path) = unwrap_transparent(expression)? else {
        return Ok(None);
    };
    let Some(identifier) = expression_path_ident(path)? else {
        return Ok(None);
    };
    if let Some((_, value)) = bindings.iter().rev().find(|(name, _)| name == identifier) {
        return Ok(match value {
            Value::OwnerAlias { ident, .. } => Some(ident.clone()),
            _ => None,
        });
    }
    if let Some(index) = inputs.iter().position(|input| input == identifier) {
        return Ok(
            (input_modes[index] == SourceMode::ConsumedResident).then(|| inputs[index].clone())
        );
    }
    Ok(None)
}

fn owner_borrow_source(
    expression: &Expr,
    inputs: &[syn::Ident],
    input_modes: &[SourceMode],
    bindings: &[(syn::Ident, Value)],
) -> Result<Option<(syn::Ident, usize)>, Error> {
    let Expr::Reference(reference) = unwrap_transparent(expression)? else {
        return Ok(None);
    };
    if reference.mutability.is_some() || !reference.attrs.is_empty() {
        return Ok(None);
    }
    let Expr::Path(path) = unwrap_transparent(&reference.expr)? else {
        return Ok(None);
    };
    let Some(identifier) = expression_path_ident(path)? else {
        return Ok(None);
    };
    if let Some((_, value)) = bindings.iter().rev().find(|(name, _)| name == identifier) {
        return Ok(match value {
            Value::OwnerAlias { ident, owner_id } | Value::BorrowAlias { ident, owner_id } => {
                Some((ident.clone(), *owner_id))
            }
            _ => None,
        });
    }
    if let Some(index) = inputs.iter().position(|input| input == identifier)
        && input_modes[index] == SourceMode::ConsumedResident
    {
        return Ok(Some((inputs[index].clone(), index)));
    }
    Ok(None)
}

fn local_binding(local: &Local) -> Result<(&syn::Ident, &Expr), Error> {
    if !local.attrs.is_empty() {
        return Err(Error::new_spanned(
            local,
            "owned tensor composition local bindings do not accept attributes",
        ));
    }
    let Pat::Ident(pattern) = &local.pat else {
        return Err(Error::new_spanned(
            &local.pat,
            "owned tensor composition let bindings must use a plain identifier",
        ));
    };
    if pattern.mutability.is_some() || pattern.by_ref.is_some() || pattern.subpat.is_some() {
        return Err(Error::new_spanned(
            &local.pat,
            "owned tensor composition let bindings must be immutable plain identifiers",
        ));
    }
    if let Some(initializer) = &local.init {
        if initializer.diverge.is_some() {
            return Err(Error::new_spanned(
                &initializer.expr,
                "owned tensor composition does not support let-else control flow",
            ));
        }
        Ok((&pattern.ident, &initializer.expr))
    } else {
        Err(Error::new_spanned(
            local,
            "owned tensor composition let bindings require an operation initializer",
        ))
    }
}

struct LoweringState<'a> {
    inputs: &'a [syn::Ident],
    input_modes: &'a [SourceMode],
    const_args: &'a [syn::Ident],
    type_args: &'a [syn::Ident],
    bindings: &'a [(syn::Ident, Value)],
    capture: &'a syn::Ident,
    crate_path: &'a Path,
    generated: &'a mut Vec<syn::Ident>,
    emitted: &'a mut Vec<proc_macro2::TokenStream>,
    next_owner_id: &'a mut usize,
}

impl LoweringState<'_> {
    fn lower(&mut self, expression: &Expr, depth: usize) -> Result<Value, Error> {
        const MAX_EXPRESSION_DEPTH: usize = 64;
        if depth > MAX_EXPRESSION_DEPTH {
            return Err(Error::new_spanned(
                expression,
                "owned tensor expression exceeds the maximum nesting depth of 64",
            ));
        }
        match expression {
            Expr::Try(syntax) => {
                if !syntax.attrs.is_empty() || !try_operand_is_operation(&syntax.expr) {
                    return Err(composition_body_error(expression));
                }
                self.lower(&syntax.expr, depth + 1)
            }
            Expr::Paren(paren) => {
                if !paren.attrs.is_empty() {
                    return Err(composition_body_error(expression));
                }
                self.lower(&paren.expr, depth + 1)
            }
            Expr::Group(group) => {
                if !group.attrs.is_empty() {
                    return Err(composition_body_error(expression));
                }
                self.lower(&group.expr, depth + 1)
            }
            Expr::Binary(binary) => self.lower_binary(binary, depth),
            Expr::Call(call) => self.lower_call(call, depth),
            Expr::Reference(reference) => {
                if reference.mutability.is_some() || !reference.attrs.is_empty() {
                    return Err(composition_body_error(expression));
                }
                let referenced = unwrap_transparent(&reference.expr)?;
                if matches!(referenced, Expr::Path(_))
                    && let Some(value) = expression_value(referenced, self.inputs, self.bindings)?
                {
                    match value {
                        Value::Input(index)
                            if self.input_modes[index] == SourceMode::ConsumedResident =>
                        {
                            return Ok(Value::BorrowedInput(index));
                        }
                        Value::OwnerAlias { ident, owner_id }
                        | Value::BorrowAlias { ident, owner_id } => {
                            return Ok(Value::BorrowAlias { ident, owner_id });
                        }
                        _ => {}
                    }
                }
                let value = self.lower(&reference.expr, depth + 1)?;
                Ok(match value {
                    Value::OwnerAlias { ident, owner_id }
                    | Value::BorrowAlias { ident, owner_id } => {
                        Value::BorrowAlias { ident, owner_id }
                    }
                    other => other,
                })
            }
            Expr::Path(_) => expression_value(expression, self.inputs, self.bindings)?
                .ok_or_else(|| composition_body_error(expression)),
            _ => Err(composition_body_error(expression)),
        }
    }
}

// A source value is not a Result. Transparent grouping may wrap one fallible operation, but
// nested `?` and borrowed values must not silently acquire Try semantics after body replacement.
fn try_operand_is_operation(mut expression: &Expr) -> bool {
    for _ in 0..64 {
        match expression {
            Expr::Paren(paren) => expression = &paren.expr,
            Expr::Group(group) => expression = &group.expr,
            Expr::Call(_) | Expr::Binary(_) => return true,
            _ => return false,
        }
    }
    false
}

impl LoweringState<'_> {
    fn lower_binary(&mut self, binary: &syn::ExprBinary, depth: usize) -> Result<Value, Error> {
        if !binary.attrs.is_empty() {
            return Err(composition_body_error(binary));
        }
        let operation = match &binary.op {
            syn::BinOp::Add(_) => Operation::Add,
            syn::BinOp::Sub(_) => Operation::Sub,
            syn::BinOp::Mul(_) => Operation::Mul,
            _ => return Err(composition_body_error(binary)),
        };
        let lhs = self.lower(&binary.left, depth + 1)?;
        let rhs = self.lower(&binary.right, depth + 1)?;
        Ok(self.emit(operation, &[lhs, rhs]))
    }

    fn lower_call(&mut self, call: &ExprCall, depth: usize) -> Result<Value, Error> {
        if !call.attrs.is_empty() {
            return Err(composition_body_error(call));
        }
        let Expr::Path(path) = call.func.as_ref() else {
            return Err(composition_body_error(&call.func));
        };
        if !path.attrs.is_empty()
            || path.qself.is_some()
            || path.path.leading_colon.is_some()
            || path
                .path
                .segments
                .iter()
                .take(path.path.segments.len().saturating_sub(1))
                .any(|segment| !matches!(segment.arguments, syn::PathArguments::None))
        {
            return Err(composition_body_error(&call.func));
        }
        let operation = operation_for_path(
            &path.path,
            &call.func,
            self.inputs,
            self.const_args,
            self.type_args,
            self.bindings,
        )?;
        let expected = match &operation {
            Operation::Identity | Operation::Relu => Some(1),
            Operation::Add | Operation::Sub | Operation::Mul | Operation::Matmul => Some(2),
            Operation::Helper(_) => None,
        };
        if expected.is_some_and(|expected| call.args.len() != expected)
            || (call.args.is_empty() && matches!(&operation, Operation::Helper(_)))
        {
            return Err(composition_body_error(&call.args));
        }
        let mut values = Vec::with_capacity(call.args.len());
        for argument in &call.args {
            values.push(self.lower(argument, depth + 1)?);
        }
        Ok(self.emit(operation, &values))
    }

    fn emit(&mut self, operation: Operation, sources: &[Value]) -> Value {
        let reserved = self.generated.iter().collect::<Vec<_>>();
        let temporary = fresh_ident("__pcu_graph_value", &self.inputs[0], &reserved);
        self.generated.push(temporary.clone());
        let helper_arguments = matches!(&operation, Operation::Helper(_));
        let values = sources
            .iter()
            .map(|value| value_tokens(value, self.inputs, self.input_modes, helper_arguments));
        let capture = self.capture;
        let call = match operation {
            Operation::Identity => quote! { #capture.identity(#(#values),*)? },
            Operation::Relu => quote! { #capture.relu(#(#values),*)? },
            Operation::Add => quote! { #capture.add(#(#values),*)? },
            Operation::Sub => quote! { #capture.sub(#(#values),*)? },
            Operation::Mul => quote! { #capture.mul(#(#values),*)? },
            Operation::Matmul => quote! { #capture.matmul(#(#values),*)? },
            Operation::Helper(path) => quote! { #path(#capture, #(#values),*)? },
        };
        let crate_path = self.crate_path;
        self.emitted.push(quote! {
            let #temporary = #crate_path::global::PcuTensorGraphOwner::from_graph_value(#call);
        });
        let owner_id = *self.next_owner_id;
        *self.next_owner_id += 1;
        Value::OwnerAlias {
            ident: temporary,
            owner_id,
        }
    }
}

fn operation_for_path(
    path: &Path,
    span: &impl quote::ToTokens,
    inputs: &[syn::Ident],
    const_args: &[syn::Ident],
    type_args: &[syn::Ident],
    bindings: &[(syn::Ident, Value)],
) -> Result<Operation, Error> {
    if path
        .segments
        .first()
        .is_some_and(|segment| segment.ident == "pcu")
    {
        if path.segments.len() != 2 {
            return Err(composition_body_error(span));
        }
        if !matches!(path.segments[1].arguments, syn::PathArguments::None) {
            return Err(composition_body_error(span));
        }
        return match path.segments[1].ident.to_string().as_str() {
            "relu" => Ok(Operation::Relu),
            "identity" => Ok(Operation::Identity),
            "add" => Ok(Operation::Add),
            "sub" => Ok(Operation::Sub),
            "mul" => Ok(Operation::Mul),
            "matmul" => Ok(Operation::Matmul),
            _ => Err(composition_body_error(span)),
        };
    }
    if path.segments.len() == 1 {
        let callee = &path.segments[0].ident;
        if inputs.iter().any(|input| input == callee)
            || bindings.iter().any(|(name, _)| name == callee)
        {
            return Err(Error::new_spanned(
                span,
                "owned tensor helper call is shadowed by a local value",
            ));
        }
    }
    let mut source_path = path.clone();
    let helper_arguments = source_path
        .segments
        .last_mut()
        .map(|segment| core::mem::replace(&mut segment.arguments, syn::PathArguments::None))
        .unwrap_or_default();
    if !valid_helper_arguments(&helper_arguments, const_args, type_args) {
        return Err(composition_body_error(span));
    }
    let mut target = companion_path(&source_path);
    target
        .segments
        .last_mut()
        .expect("capture segment was appended")
        .arguments = helper_arguments;
    Ok(Operation::Helper(target))
}

fn valid_helper_arguments(
    arguments: &syn::PathArguments,
    const_args: &[syn::Ident],
    type_args: &[syn::Ident],
) -> bool {
    match arguments {
        syn::PathArguments::None => true,
        syn::PathArguments::AngleBracketed(arguments) => arguments.args.iter().all(|argument| {
            match argument {
                syn::GenericArgument::Const(expression) => {
                    valid_const_expr(expression, const_args).is_some()
                }
                // Rust parses a bare identifier in turbofish position as a type argument,
                // even when it resolves to a const parameter in the generated companion.
                syn::GenericArgument::Type(Type::Path(path)) => path
                    .path
                    .get_ident()
                    .is_some_and(|ident| const_args.contains(ident) || type_args.contains(ident)),
                _ => false,
            }
        }),
        syn::PathArguments::Parenthesized(_) => false,
    }
}

fn companion_path(path: &Path) -> Path {
    let mut path = path.clone();
    if path
        .segments
        .first()
        .is_some_and(|first| first.ident == "self")
    {
        path.segments
            .first_mut()
            .expect("first segment checked")
            .ident = format_ident!("super");
    } else if path
        .segments
        .first()
        .is_some_and(|first| first.ident == "super")
    {
        path.segments
            .insert(0, syn::PathSegment::from(format_ident!("super")));
    }
    path.segments
        .push(syn::PathSegment::from(format_ident!("__pcu_capture")));
    path
}

fn source_binding_names(function: &ItemFn) -> Vec<syn::Ident> {
    function
        .block
        .stmts
        .iter()
        .filter_map(|statement| {
            let syn::Stmt::Local(local) = statement else {
                return None;
            };
            let Pat::Ident(pattern) = &local.pat else {
                return None;
            };
            Some(pattern.ident.clone())
        })
        .collect()
}

fn unwrap_result_return(expression: &Expr) -> Result<&Expr, Error> {
    let Expr::Call(ExprCall {
        attrs, func, args, ..
    }) = expression
    else {
        return Ok(expression);
    };
    let Expr::Path(path) = func.as_ref() else {
        return Ok(expression);
    };
    if path.path.is_ident("Ok") {
        if !attrs.is_empty()
            || !path.attrs.is_empty()
            || path.qself.is_some()
            || path.path.leading_colon.is_some()
            || args.len() != 1
        {
            return Err(composition_body_error(expression));
        }
        return Ok(&args[0]);
    }
    Ok(expression)
}

fn expression_value(
    expression: &Expr,
    inputs: &[syn::Ident],
    bindings: &[(syn::Ident, Value)],
) -> Result<Option<Value>, Error> {
    let identifier = match expression {
        Expr::Path(path) => expression_path_ident(path)?,
        Expr::Reference(reference) => {
            if reference.mutability.is_some() || !reference.attrs.is_empty() {
                return Err(composition_body_error(expression));
            }
            let Expr::Path(path) = reference.expr.as_ref() else {
                return Err(composition_body_error(expression));
            };
            expression_path_ident(path)?
        }
        _ => None,
    };
    let Some(identifier) = identifier else {
        return Ok(None);
    };
    if let Some((_, value)) = bindings.iter().rev().find(|(name, _)| name == identifier) {
        return Ok(Some(value.clone()));
    }
    Ok(inputs
        .iter()
        .position(|input| input == identifier)
        .map(Value::Input))
}

fn expression_path_ident(path: &syn::ExprPath) -> Result<Option<&syn::Ident>, Error> {
    if !path.attrs.is_empty() || path.qself.is_some() || path.path.leading_colon.is_some() {
        return Err(composition_body_error(path));
    }
    Ok(path.path.get_ident())
}

fn value_tokens(
    value: &Value,
    inputs: &[syn::Ident],
    input_modes: &[SourceMode],
    helper_argument: bool,
) -> proc_macro2::TokenStream {
    match value {
        Value::Input(index) => {
            let input = &inputs[*index];
            if input_modes[*index] == SourceMode::ConsumedResident && !helper_argument {
                quote! { #input.into_graph_value() }
            } else {
                quote! { #input }
            }
        }
        Value::BorrowedInput(index) => {
            let input = &inputs[*index];
            quote! { #input.borrowed_graph_value() }
        }
        Value::OwnerAlias { ident: alias, .. } => {
            if helper_argument {
                quote! { #alias }
            } else {
                quote! { #alias.into_graph_value() }
            }
        }
        Value::BorrowAlias { ident: alias, .. } => quote! { #alias.borrowed_graph_value() },
    }
}

fn composition_body_error(span: &impl quote::ToTokens) -> Error {
    Error::new_spanned(
        span,
        "owned tensor body accepts immutable let bindings of pcu::relu|identity|add|sub|mul, binary `+`|`-`|`*`, or another #[pcu] helper, and a final value/Ok(value)",
    )
}

fn fresh_ident(base: &str, input: &syn::Ident, reserved: &[&syn::Ident]) -> syn::Ident {
    let mut candidate = format_ident!("{base}");
    while candidate == *input || reserved.iter().any(|identifier| **identifier == candidate) {
        candidate = format_ident!("_{candidate}");
    }
    candidate
}

fn owned_return_scalar(output: &ReturnType) -> Option<syn::Ident> {
    let ReturnType::Type(_, ty) = output else {
        return None;
    };
    let Type::Path(result) = ty.as_ref() else {
        return None;
    };
    let result_segment = result.path.segments.last()?;
    if result_segment.ident != "Result" {
        return None;
    }
    let syn::PathArguments::AngleBracketed(result_args) = &result_segment.arguments else {
        return None;
    };
    if result_args.args.len() != 2 {
        return None;
    }
    let Some(syn::GenericArgument::Type(Type::Path(tensor))) = result_args.args.first() else {
        return None;
    };
    let tensor_segment = tensor.path.segments.last()?;
    if tensor_segment.ident != "PcuTensor" {
        return None;
    }
    let syn::PathArguments::AngleBracketed(tensor_args) = &tensor_segment.arguments else {
        return None;
    };
    if tensor_args.args.len() != 1 {
        return None;
    }
    let Some(syn::GenericArgument::Type(Type::Path(scalar))) = tensor_args.args.first() else {
        return None;
    };
    if scalar.qself.is_some()
        || scalar.path.segments.len() != 1
        || !matches!(result_args.args.last(), Some(syn::GenericArgument::Type(Type::Path(error))) if error.path.segments.last().is_some_and(|segment| segment.ident == "PcuExecutionError"))
    {
        return None;
    }
    scalar.path.get_ident().cloned()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn expand(source: &ItemFn) -> String {
        expand_owned_return(source, &syn::parse_quote!(::pcu_alias))
            .expect("owned tensor composition should expand")
            .to_string()
    }

    #[test]
    fn f64_owned_profile_preserves_scalar_in_sources_and_capture() {
        let source: ItemFn = syn::parse_quote! {
            fn project<const R: usize, const K: usize, const C: usize>(
                left: &[[f64; K]; R], right: &[[f64; C]; K],
            ) -> Result<PcuTensor<f64>, PcuExecutionError> {
                Ok(pcu::relu(pcu::matmul(left, right)?)?)
            }
        };
        let tokens = expand(&source);
        assert!(tokens.contains("PcuTensorSource < f64"));
        assert!(tokens.contains("PcuTensorGraphValue < f64 >"));
        assert!(tokens.contains("call_owned_tensor_capture"));
        assert!(!tokens.contains("call_owned_tensor_capture_f64"));
        assert!(!tokens.contains("PcuTensorGraphValue < f32 >"));
    }

    #[test]
    fn owned_profile_rejects_mixed_scalar_inputs_and_unsupported_returns() {
        for source in [
            "fn bad(left: &[f32], right: &[f64]) -> Result<PcuTensor<f64>, PcuExecutionError> { Ok(left + right) }",
            "fn bad(input: &[f64]) -> Result<PcuTensor<f32>, PcuExecutionError> { Ok(input) }",
            "fn bad(input: &[u64]) -> Result<PcuTensor<[u64; 2]>, PcuExecutionError> { Ok(input) }",
        ] {
            let function: ItemFn = syn::parse_str(source).unwrap();
            assert!(expand_owned_return(&function, &syn::parse_quote!(::pcu_alias)).is_err());
        }
        let integer: ItemFn = syn::parse_quote! {
            fn copy(input: &[u32]) -> Result<PcuTensor<u32>, PcuExecutionError> {
                Ok(pcu::identity(input)?)
            }
        };
        assert!(expand_owned_return(&integer, &syn::parse_quote!(::pcu_alias)).is_ok());
    }

    #[test]
    fn consumed_owner_keeps_exact_by_value_signature_and_calls_consuming_entry() {
        let source: ItemFn = syn::parse_quote! {
            fn activate<T: PcuScalar>(input: PcuTensor<T>)
                -> Result<PcuTensor<T>, PcuExecutionError>
            {
                Ok(pcu::relu(&input)?)
            }
        };
        let tokens = expand(&source);
        assert!(tokens.contains("input : :: pcu_alias :: global :: PcuTensor < T >"));
        assert!(tokens.contains("call_consumed_tensor_capture"));
        assert!(tokens.contains("activate :: __pcu_capture_entry :: < T >"));
        assert!(tokens.contains("PcuTensorGraphOwner < T >"));
        assert!(tokens.contains("PcuTensorGraphOwner :: from_graph_value"));
        assert!(!tokens.contains("require_rank (input . borrowed_graph_value () , 1) ?"));
        assert!(!tokens.contains("as_tensor_source"));
    }

    #[test]
    fn one_consumed_owner_and_one_borrowed_resident_use_pair_entry() {
        let source: ItemFn = syn::parse_quote! {
            fn add<T: PcuScalar>(lhs: PcuTensor<T>, rhs: &PcuTensor<T>)
                -> Result<PcuTensor<T>, PcuExecutionError>
            {
                Ok(pcu::add(&lhs, rhs)?)
            }
        };
        let tokens = expand(&source);
        assert!(tokens.contains("call_consumed_pair_tensor_capture"));
        assert!(tokens.contains("lhs , __pcu_owned_source , 0"));
        assert!(tokens.contains("PcuTensorSource :: < T , :: pcu_alias :: global :: DynamicResidentShape > :: as_tensor_source (rhs)"));
        assert!(!tokens.contains("as_tensor_source (& lhs)"));
        assert!(tokens.contains("add :: __pcu_capture_entry :: < T >"));
    }

    #[test]
    fn two_consumed_owners_use_the_pair_transfer_entry() {
        let source: ItemFn = syn::parse_quote! {
            fn add<T: PcuScalar>(lhs: PcuTensor<T>, rhs: PcuTensor<T>)
                -> Result<PcuTensor<T>, PcuExecutionError>
            {
                Ok(pcu::add(&lhs, &rhs)?)
            }
        };
        let tokens = expand(&source);
        assert!(tokens.contains("call_consumed_owners_tensor_capture"));
        assert!(tokens.contains("[lhs , rhs]"));
        assert!(!tokens.contains("call_consumed_pair_tensor_capture"));
        assert!(!tokens.contains("as_tensor_source"));
    }

    #[test]
    fn multiple_consumed_owners_use_const_sized_transfer_entry() {
        let source: ItemFn = syn::parse_quote! {
            fn add<T: PcuScalar>(unused: PcuTensor<T>, lhs: PcuTensor<T>, rhs: PcuTensor<T>)
                -> Result<PcuTensor<T>, PcuExecutionError>
            {
                Ok(pcu::add(&lhs, &rhs)?)
            }
        };
        let tokens = expand(&source);
        assert!(tokens.contains("call_consumed_owners_tensor_capture"));
        assert!(tokens.contains("[unused , lhs , rhs]"));
        assert!(tokens.contains(
            "__pcu_inputs : [:: pcu_alias :: global :: PcuTensorGraphValue < T > ; 3usize]"
        ));
        assert!(!tokens.contains("as_tensor_source"));
    }

    #[test]
    fn mixed_three_formal_sources_use_stack_carriers() {
        let source: ItemFn = syn::parse_quote! {
            fn add<T: PcuScalar>(unused: &PcuTensor<T>, lhs: PcuTensor<T>, rhs: &PcuTensor<T>)
                -> Result<PcuTensor<T>, PcuExecutionError>
            {
                Ok(pcu::add(&lhs, rhs)?)
            }
        };
        let tokens = expand(&source);
        assert!(tokens.contains("call_mixed_consumed_tensor_capture"));
        assert!(tokens.contains("PcuTensorCallInput :: Consumed (lhs)"));
        assert_eq!(tokens.matches("PcuTensorCallInput :: Borrowed").count(), 2);
        assert!(!tokens.contains("call_consumed_pair_tensor_capture"));
    }

    #[test]
    fn unused_fixed_array_formals_keep_shape_validation_under_scoped_allow() {
        let source: ItemFn = syn::parse_quote! {
            fn add<T: PcuScalar>(
                _ram: &[T; 2],
                lhs: PcuTensor<T>,
                rhs: PcuTensor<T>,
            ) -> Result<PcuTensor<T>, PcuExecutionError> {
                Ok(pcu::add(&lhs, &rhs)?)
            }
        };
        let tokens = expand(&source);
        assert!(tokens.contains("call_mixed_consumed_tensor_capture"));
        assert!(tokens.contains("allow (clippy :: used_underscore_binding)"));
        assert!(tokens.contains("require_shape (_ram"));
    }

    #[test]
    fn consumed_owner_accepts_concrete_float_and_rejects_invalid_ownership() {
        for source in [
            "fn relu(input: PcuTensor<f32>) -> Result<PcuTensor<f32>, PcuExecutionError> { Ok(pcu::relu(input)?) }",
            "fn relu(input: PcuTensor<f64>) -> Result<PcuTensor<f64>, PcuExecutionError> { Ok(pcu::relu(input)?) }",
            "fn relu(input: PcuTensor<f32>) -> Result<PcuTensor<f32>, PcuExecutionError> { Ok((pcu::relu((input))?)) }",
        ] {
            let function: ItemFn = syn::parse_str(source).unwrap();
            let tokens = expand(&function);
            assert!(tokens.contains("call_consumed_tensor_capture"));
        }

        for source in [
            "fn bad(input: PcuTensor<f32>) -> Result<PcuTensor<f32>, PcuExecutionError> { Ok(pcu::add(input, input)?) }",
            "fn bad(input: PcuTensor<f64>) -> Result<PcuTensor<f32>, PcuExecutionError> { Ok(pcu::relu(input)?) }",
            "fn bad(input: &mut PcuTensor<f32>) -> Result<PcuTensor<f32>, PcuExecutionError> { Ok(pcu::relu(input)?) }",
        ] {
            let function: ItemFn = syn::parse_str(source).unwrap();
            assert!(expand_owned_return(&function, &syn::parse_quote!(::pcu_alias)).is_err());
        }
    }

    #[test]
    fn mixed_owner_and_host_inputs_use_pair_entry_with_borrowed_source() {
        let source: ItemFn = syn::parse_quote! {
            fn combine<T: PcuScalar>(left: PcuTensor<T>, right: &[T])
                -> Result<PcuTensor<T>, PcuExecutionError>
            {
                let rhs = pcu::identity(right)?;
                let lhs = pcu::relu(left)?;
                Ok(pcu::add(lhs, rhs)?)
            }
        };
        let tokens = expand(&source);
        assert!(tokens.contains("call_consumed_pair_tensor_capture"));
        assert!(!tokens.contains("as_tensor_source (& left)"));
        assert!(tokens.contains("PcuTensorSource :: < T , :: pcu_alias :: global :: SliceShape > :: as_tensor_source (right)"));
        assert!(tokens.contains("PcuTensorGraphOwner :: from_graph_value"));
        assert!(tokens.contains("combine :: __pcu_capture_entry"));
    }

    #[test]
    fn unused_consumed_formals_keep_metadata_without_underscore_lint_noise() {
        let source: ItemFn = syn::parse_quote! {
            fn ignore_second<T: PcuScalar>(used: PcuTensor<T>, _unused: PcuTensor<T>)
                -> Result<PcuTensor<T>, PcuExecutionError>
            {
                Ok(pcu::identity(used)?)
            }
        };
        let tokens = expand(&source);
        assert!(tokens.contains("clippy :: used_underscore_binding"));
        assert!(tokens.contains("_unused : :: pcu_alias :: global :: PcuTensor < T >"));
        assert!(tokens.contains("call_consumed_owners_tensor_capture"));
        assert!(!tokens.contains("as_tensor_source"));
    }

    #[test]
    fn multiple_consumed_owners_keep_independent_noncopy_tokens() {
        let source: ItemFn = syn::parse_quote! {
            fn combine<T: PcuScalar>(left: PcuTensor<T>, right: PcuTensor<T>)
                -> Result<PcuTensor<T>, PcuExecutionError>
            {
                Ok(pcu::add(pcu::relu(left)?, pcu::relu(right)?)?)
            }
        };
        let tokens = expand(&source);
        assert!(tokens.contains("call_consumed_owners_tensor_capture"));
        assert!(!tokens.contains("as_tensor_source"));
        assert!(
            tokens
                .matches("PcuTensorGraphOwner :: from_graph_value")
                .count()
                >= 2
        );
    }

    #[test]
    fn operation_rvalue_references_borrow_owner_results() {
        let source: ItemFn = syn::parse_quote! {
            fn combine<T: PcuScalar>(left: PcuTensor<T>, right: PcuTensor<T>)
                -> Result<PcuTensor<T>, PcuExecutionError>
            {
                Ok(pcu::add(&pcu::relu(left)?, &pcu::identity(right)?)?)
            }
        };
        let tokens = expand(&source);
        assert!(tokens.contains("PcuTensorGraphOwner :: from_graph_value"));
        assert!(tokens.contains("borrowed_graph_value"));

        for function in [
            syn::parse_quote! {
                fn bad(input: &[f32]) -> Result<PcuTensor<f32>, PcuExecutionError> {
                    Ok(&pcu::relu(input)?)
                }
            },
            syn::parse_quote! {
                fn bad(input: PcuTensor<f32>) -> Result<PcuTensor<f32>, PcuExecutionError> {
                    Ok(&pcu::relu(input)?)
                }
            },
        ] {
            assert!(expand_owned_return(&function, &syn::parse_quote!(::pcu_alias)).is_err());
        }
    }

    #[test]
    fn borrowed_sources_cannot_be_returned_as_owned_results() {
        for function in [
            syn::parse_quote! {
                fn bad(input: &[f32]) -> Result<PcuTensor<f32>, PcuExecutionError> {
                    Ok(input)
                }
            },
            syn::parse_quote! {
                fn bad(input: &PcuTensor<f32>) -> Result<PcuTensor<f32>, PcuExecutionError> {
                    Ok(input)
                }
            },
            syn::parse_quote! {
                fn bad(input: PcuTensor<f32>) -> Result<PcuTensor<f32>, PcuExecutionError> {
                    let view = &input;
                    Ok(view)
                }
            },
        ] {
            assert!(expand_owned_return(&function, &syn::parse_quote!(::pcu_alias)).is_err());
        }

        let consumed: ItemFn = syn::parse_quote! {
            fn valid(input: PcuTensor<f32>) -> Result<PcuTensor<f32>, PcuExecutionError> {
                Ok(input)
            }
        };
        assert!(expand_owned_return(&consumed, &syn::parse_quote!(::pcu_alias)).is_ok());
    }

    #[test]
    fn consumed_and_borrowed_resident_sources_use_the_pair_entry() {
        let source: ItemFn = syn::parse_quote! {
            fn combine<T: PcuScalar>(left: PcuTensor<T>, right: &PcuTensor<T>)
                -> Result<PcuTensor<T>, PcuExecutionError>
            {
                Ok(pcu::add(pcu::relu(left)?, pcu::relu(right)?)?)
            }
        };
        let tokens = expand(&source);
        assert!(tokens.contains("call_consumed_pair_tensor_capture"));
        assert!(tokens.contains("right : & :: pcu_alias :: global :: PcuTensor < T >"));
        assert!(tokens.contains("as_tensor_source (right)"));
        assert!(!tokens.contains("as_tensor_source (& left)"));
        assert!(tokens.contains("PcuTensorGraphOwner :: from_graph_value"));
    }

    #[test]
    fn borrowed_resident_references_keep_graph_value_companion_mode() {
        let source: ItemFn = syn::parse_quote! {
            fn borrow<T: PcuScalar>(input: &PcuTensor<T>)
                -> Result<PcuTensor<T>, PcuExecutionError>
            {
                Ok(pcu::identity(input)?)
            }
        };
        let tokens = expand(&source);
        assert!(tokens.contains("input : & :: pcu_alias :: global :: PcuTensor < T >"));
        assert!(tokens.contains("DynamicResidentShape"));
        assert!(tokens.contains(
            "__pcu_inputs : [:: pcu_alias :: global :: PcuTensorGraphValue < T > ; 1usize]"
        ));
        assert!(tokens.contains("input : :: pcu_alias :: global :: PcuTensorGraphValue < T >"));
        assert!(tokens.contains("PcuTensorGraphOwner :: from_graph_value"));
    }

    #[test]
    fn grouped_borrows_preserve_owner_until_its_later_move() {
        let source: ItemFn = syn::parse_quote! {
            fn borrow_then_consume<T: PcuScalar>(input: PcuTensor<T>)
                -> Result<PcuTensor<T>, PcuExecutionError>
            {
                let copied = helper((&(&input)))?;
                let consumed = pcu::relu((input))?;
                Ok(pcu::add(copied, consumed)?)
            }
        };
        let tokens = expand(&source);
        assert!(tokens.contains("helper :: __pcu_capture"));
        assert!(tokens.contains("input . borrowed_graph_value ()"));
        assert!(tokens.contains("into_graph_value"), "{tokens}");
    }

    #[test]
    fn borrowed_consuming_helper_can_be_followed_by_owner_move() {
        let source: ItemFn = syn::parse_quote! {
            fn borrow_then_consume<T: PcuScalar>(input: PcuTensor<T>)
                -> Result<PcuTensor<T>, PcuExecutionError>
            {
                let copied = helper(&input)?;
                let consumed = pcu::relu(input)?;
                Ok(pcu::add(copied, consumed)?)
            }
        };
        let tokens = expand(&source);
        assert!(tokens.contains("helper :: __pcu_capture"));
        assert!(tokens.contains("input . borrowed_graph_value ()"));
        assert!(tokens.contains("into_graph_value"), "{tokens}");
    }

    #[test]
    fn owner_and_borrow_aliases_emit_real_rust_moves_and_borrows() {
        let source: ItemFn = syn::parse_quote! {
            fn rename_then_consume<T: PcuScalar>(input: PcuTensor<T>)
                -> Result<PcuTensor<T>, PcuExecutionError>
            {
                let owner = input;
                let alias = owner;
                let view = &alias;
                let observed = pcu::identity(view)?;
                let consumed = pcu::relu(alias)?;
                Ok(pcu::add(observed, consumed)?)
            }
        };
        let tokens = expand(&source);
        assert!(tokens.contains("let __pcu_owner_alias = input"));
        assert!(tokens.contains("let ___pcu_owner_alias = __pcu_owner_alias"));
        assert!(tokens.contains("let __pcu_owner_view = & ___pcu_owner_alias"));
        assert!(tokens.contains("__pcu_owner_view . borrowed_graph_value ()"));
        assert!(tokens.contains("___pcu_owner_alias . into_graph_value ()"));
    }

    #[test]
    fn immutable_shadowing_uses_newest_binding_but_keeps_old_view_root() {
        let source: ItemFn = syn::parse_quote! {
            fn shadow<T: PcuScalar>(left: PcuTensor<T>, right: PcuTensor<T>)
                -> Result<PcuTensor<T>, PcuExecutionError>
            {
                let view = &left;
                let left = pcu::relu(right)?;
                let next = pcu::relu(left)?;
                let observed = pcu::identity(view)?;
                Ok(pcu::add(observed, next)?)
            }
        };
        let tokens = expand(&source);
        assert!(tokens.contains("let __pcu_owner_view = & left"));
        assert!(!tokens.contains("let left ="));
        assert!(tokens.contains("__pcu_owner_view . borrowed_graph_value ()"));
        assert!(tokens.contains("__pcu_graph_value . into_graph_value ()"));
    }

    #[test]
    fn borrowed_input_aliases_can_be_shadowed_by_operation_owners() {
        let source: ItemFn = syn::parse_quote! {
            fn shadow(input: &[f32]) -> Result<PcuTensor<f32>, PcuExecutionError> {
                let alias = input;
                let input = pcu::identity(alias)?;
                Ok(pcu::relu(input)?)
            }
        };
        let tokens = expand(&source);
        assert!(tokens.contains("identity"), "{tokens}");
        assert!(tokens.contains("let __pcu_graph_value ="));
        assert!(tokens.contains("__pcu_graph_value . into_graph_value ()"));
    }

    #[test]
    fn immutable_locals_cannot_shadow_const_generic_parameters() {
        let source: ItemFn = syn::parse_quote! {
            fn invalid<const N: usize>(input: &[f32])
                -> Result<PcuTensor<f32>, PcuExecutionError>
            {
                let N = pcu::identity(input)?;
                Ok(N)
            }
        };
        let error = expand_owned_return(&source, &syn::parse_quote!(::pcu_alias)).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("cannot shadow const generic parameters")
        );
    }

    #[test]
    fn shadowed_single_segment_helper_names_remain_noncallable() {
        let source: ItemFn = syn::parse_quote! {
            fn invalid(input: &[f32]) -> Result<PcuTensor<f32>, PcuExecutionError> {
                let helper = pcu::identity(input)?;
                Ok(helper(input)?)
            }
        };
        let error = expand_owned_return(&source, &syn::parse_quote!(::pcu_alias)).unwrap_err();
        assert!(error.to_string().contains("helper call is shadowed"));
    }

    #[test]
    fn generic_scalar_and_const_arguments_flow_through_capture_markers() {
        let source: ItemFn = syn::parse_quote! {
            fn identity<T: PcuScalar, const R: usize, const C: usize>(
                input: &[[T; C]; R],
            ) -> Result<PcuTensor<T>, PcuExecutionError> {
                Ok(pcu::identity(input)?)
            }
        };
        let tokens = expand(&source);
        assert!(tokens.contains(
            "PcuTensorSource < T , :: pcu_alias :: global :: FixedMatrixShape < R , C > >"
        ));
        assert!(tokens.contains("PcuTensorGraphValue < T >"));
        assert!(tokens.contains("call_owned_tensor_capture"));
        assert!(tokens.contains("TypeId :: of :: < __PcuOwnedCaptureMarker < T , R , C > >"));
        assert!(tokens.contains("__pcu_capture_entry :: < T , R , C >"));

        let where_bound: ItemFn = syn::parse_quote! {
            fn identity<T, const N: usize>(input: &[T; N])
                -> Result<PcuTensor<T>, PcuExecutionError>
            where
                T: PcuScalar,
            {
                Ok(pcu::identity(input)?)
            }
        };
        assert!(expand_owned_return(&where_bound, &syn::parse_quote!(::pcu_alias)).is_ok());
    }

    #[test]
    fn generic_scalar_helpers_preserve_explicit_type_and_const_turbofish() {
        let source: ItemFn = syn::parse_quote! {
            fn outer<T: PcuScalar, const N: usize>(input: &[T; N])
                -> Result<PcuTensor<T>, PcuExecutionError>
            {
                Ok(inner::<T, N>(input)?)
            }
        };
        let tokens = expand(&source);
        assert!(tokens.contains("inner :: __pcu_capture :: < T , N >"));
    }

    #[test]
    fn generic_type_and_const_names_are_reserved_from_capture_locals() {
        let source: ItemFn = syn::parse_quote! {
            fn collision<__PcuCaptureRecursionMarker: PcuScalar, const __PCU_OWNED_TENSOR_SITE: usize>(
                __pcu_owned_source: &[__PcuCaptureRecursionMarker; __PCU_OWNED_TENSOR_SITE],
            ) -> Result<PcuTensor<__PcuCaptureRecursionMarker>, PcuExecutionError> {
                let __pcu_capture = pcu::identity(__pcu_owned_source)?;
                Ok(__pcu_capture)
            }
        };
        let tokens = expand(&source);
        assert!(tokens.contains("struct ___PcuCaptureRecursionMarker"));
        assert!(tokens.contains("static ___PCU_OWNED_TENSOR_SITE"));
        assert!(tokens.contains("let ___pcu_owned_source"));
        assert!(tokens.contains("___pcu_capture : & mut"));
    }

    #[test]
    fn helper_calls_lower_to_companion_function_items_and_capture_calls() {
        let source: ItemFn = syn::parse_quote! {
            pub fn outer(input: &[f32]) -> Result<PcuTensor<f32>, PcuExecutionError> {
                let first = inner(input)?;
                Ok(math::finish(&first)?)
            }
        };
        let expanded = expand(&source);
        assert!(expanded.contains("outer :: __pcu_capture"));
        assert!(expanded.contains("inner :: __pcu_capture"));
        assert!(expanded.contains("math :: finish :: __pcu_capture"));
        assert!(expanded.contains("__pcu_capture . enter"));
        assert!(expanded.contains("__pcu_capture . leave"));
        assert!(!expanded.contains("call_owned_tensor_program"));
    }

    #[test]
    fn builtins_are_captured_and_unrecognized_pcu_calls_are_rejected() {
        let source: ItemFn = syn::parse_quote! {
            fn relu(input: &[f32]) -> Result<PcuTensor<f32>, PcuExecutionError> {
                let activated = pcu::relu(input)?;
                Ok(pcu::identity(activated)?)
            }
        };
        let expanded = expand(&source);
        assert!(expanded.contains("__pcu_capture . relu"));
        assert!(expanded.contains("__pcu_capture . identity"));

        let unknown: ItemFn = syn::parse_quote! {
            fn relu(input: &[f32]) -> Result<PcuTensor<f32>, PcuExecutionError> {
                pcu::softmax(input)
            }
        };
        assert!(expand_owned_return(&unknown, &syn::parse_quote!(::pcu_alias)).is_err());
    }

    #[test]
    fn multiple_inputs_and_add_are_emitted_as_graph_values() {
        let source: ItemFn = syn::parse_quote! {
            fn sum(lhs: &[f32], rhs: &[f32]) -> Result<PcuTensor<f32>, PcuExecutionError> {
                let both = pcu::add(lhs, &rhs)?;
                Ok(pcu::relu(both)?)
            }
        };
        let expanded = expand(&source);
        assert!(expanded.contains("PcuTensorSource :: < f32 , :: pcu_alias :: global :: SliceShape > :: as_tensor_source (lhs)"));
        assert!(expanded.contains("PcuTensorSource :: < f32 , :: pcu_alias :: global :: SliceShape > :: as_tensor_source (rhs)"));
        assert!(expanded.contains("PcuTensorGraphValue < f32 > ; 2usize"));
        assert!(expanded.contains("__pcu_capture . add (lhs , rhs)"));
        assert!(expanded.contains("__pcu_capture_entry"));

        let wrong_builtin_arity: ItemFn = syn::parse_quote! {
            fn sum(lhs: &[f32], rhs: &[f32]) -> Result<PcuTensor<f32>, PcuExecutionError> {
                pcu::add(lhs)
            }
        };
        assert!(
            expand_owned_return(&wrong_builtin_arity, &syn::parse_quote!(::pcu_alias)).is_err()
        );

        let duplicate_names: ItemFn = syn::parse_quote! {
            fn sum(lhs: &[f32], lhs: &[f32]) -> Result<PcuTensor<f32>, PcuExecutionError> {
                pcu::add(lhs, lhs)
            }
        };
        assert!(expand_owned_return(&duplicate_names, &syn::parse_quote!(::pcu_alias)).is_err());
    }

    #[test]
    fn fixed_matrix_inputs_emit_shape_bounds_guards_and_matmul() {
        let source: ItemFn = syn::parse_quote! {
            fn gemm<const M: usize, const K: usize, const N: usize>(
                lhs: &[[f32; K]; M],
                rhs: &[[f32; N]; K],
            ) -> Result<PcuTensor<f32>, PcuExecutionError> {
                Ok(pcu::matmul(lhs, rhs)?)
            }
        };
        let expanded = expand(&source);
        assert!(expanded.contains("FixedMatrixShape < M , K >"));
        assert!(expanded.contains("FixedMatrixShape < K , N >"));
        assert!(expanded.contains("PcuSourceShape :: FixedMatrix { rows : M , columns : K }"));
        assert!(expanded.contains("PcuSourceShape :: FixedMatrix { rows : K , columns : N }"));
        assert!(expanded.contains("__pcu_capture . matmul (lhs , rhs)"));
        assert!(expanded.contains("TypeId :: of :: < __PcuOwnedCaptureMarker < M , K , N > >"));

        let incompatible: ItemFn = syn::parse_quote! {
            fn bad<const M: usize, const K: usize>(lhs: &[[f32; K]; M])
                -> Result<PcuTensor<f32>, PcuExecutionError>
            {
                Ok(pcu::matmul(lhs, lhs)?)
            }
        };
        assert!(expand_owned_return(&incompatible, &syn::parse_quote!(::pcu_alias)).is_ok());
    }

    #[test]
    fn rank_one_and_fixed_array_sources_have_distinct_shape_carriers() {
        let slice: ItemFn = syn::parse_quote! {
            fn activate(input: &[f32]) -> Result<PcuTensor<f32>, PcuExecutionError> {
                Ok(pcu::relu(input)?)
            }
        };
        let slice_tokens = expand(&slice);
        assert!(slice_tokens.contains(":: pcu_alias :: global :: PcuTensorSource < f32 , :: pcu_alias :: global :: SliceShape >"));
        assert!(slice_tokens.contains("require_rank (input , 1)"));

        let array: ItemFn = syn::parse_quote! {
            fn activate<const N: usize>(input: &[f32; N])
                -> Result<PcuTensor<f32>, PcuExecutionError>
            {
                Ok(pcu::relu(input)?)
            }
        };
        let array_tokens = expand(&array);
        assert!(array_tokens.contains("FixedArrayShape < N >"));
        assert!(array_tokens.contains("PcuSourceShape :: FixedArray { length : N }"));
    }

    #[test]
    fn const_generic_helper_arguments_move_to_capture_companion() {
        let source: ItemFn = syn::parse_quote! {
            fn caller<const M: usize, const K: usize>(input: &[[f32; K]; M])
                -> Result<PcuTensor<f32>, PcuExecutionError>
            {
                Ok(inner::<M, K>(input)?)
            }
        };
        let expanded = expand(&source);
        assert!(expanded.contains("inner :: __pcu_capture :: < M , K >"));

        let unsupported: ItemFn = syn::parse_quote! {
            fn caller<const M: usize>(input: &[f32; M])
                -> Result<PcuTensor<f32>, PcuExecutionError>
            {
                Ok(inner::<f32>(input)?)
            }
        };
        assert!(expand_owned_return(&unsupported, &syn::parse_quote!(::pcu_alias)).is_err());
    }

    #[test]
    fn owned_profile_rejects_more_than_thirty_two_inputs() {
        let source: ItemFn = syn::parse_quote! {
            fn sum(
                a0: &[f32], a1: &[f32], a2: &[f32], a3: &[f32], a4: &[f32], a5: &[f32], a6: &[f32], a7: &[f32],
                a8: &[f32], a9: &[f32], a10: &[f32], a11: &[f32], a12: &[f32], a13: &[f32], a14: &[f32], a15: &[f32],
                a16: &[f32], a17: &[f32], a18: &[f32], a19: &[f32], a20: &[f32], a21: &[f32], a22: &[f32], a23: &[f32],
                a24: &[f32], a25: &[f32], a26: &[f32], a27: &[f32], a28: &[f32], a29: &[f32], a30: &[f32], a31: &[f32],
                a32: &[f32],
            ) -> Result<PcuTensor<f32>, PcuExecutionError> { Ok(a0) }
        };
        assert!(expand_owned_return(&source, &syn::parse_quote!(::pcu_alias)).is_err());
    }

    #[test]
    fn elementwise_operators_and_named_operations_share_capture_methods() {
        let source: ItemFn = syn::parse_quote! {
            fn arithmetic(lhs: &[f32], rhs: &[f32]) -> Result<PcuTensor<f32>, PcuExecutionError> {
                let sum = lhs + rhs;
                let difference = pcu::sub(lhs, rhs)?;
                let product = difference * rhs;
                Ok(pcu::mul(sum, product)?)
            }
        };
        let expanded = expand(&source);
        assert!(expanded.contains("__pcu_capture . add (lhs , rhs)"));
        assert!(expanded.contains("__pcu_capture . sub (lhs , rhs)"));
        assert_eq!(expanded.matches("__pcu_capture . mul").count(), 2);

        let unsupported: ItemFn = syn::parse_quote! {
            fn arithmetic(lhs: &[f32], rhs: &[f32]) -> Result<PcuTensor<f32>, PcuExecutionError> {
                let invalid = lhs / rhs;
                Ok(invalid)
            }
        };
        assert!(expand_owned_return(&unsupported, &syn::parse_quote!(::pcu_alias)).is_err());
    }

    #[test]
    fn try_rejects_plain_values_and_repeated_unwrapping() {
        for body in ["Ok(input?)", "Ok(pcu::identity(input)??)", "Ok((&input)?)"] {
            let source: ItemFn = syn::parse_str(&format!(
                "fn invalid(input: &[f32]) -> Result<PcuTensor<f32>, PcuExecutionError> {{ {body} }}"
            ))
            .unwrap();
            assert!(expand_owned_return(&source, &syn::parse_quote!(::pcu_alias)).is_err());
        }
        let source: ItemFn = syn::parse_quote! {
            fn valid(input: &[f32]) -> Result<PcuTensor<f32>, PcuExecutionError> {
                Ok((pcu::identity(input))?)
            }
        };
        assert!(expand_owned_return(&source, &syn::parse_quote!(::pcu_alias)).is_ok());
    }

    #[test]
    fn nested_graph_expressions_flatten_left_to_right_and_enforce_depth_limit() {
        let source: ItemFn = syn::parse_quote! {
            fn nested(lhs: &[f32], rhs: &[f32]) -> Result<PcuTensor<f32>, PcuExecutionError> {
                Ok(pcu::relu((lhs + rhs) * rhs)?)
            }
        };
        let expanded = expand(&source);
        let text = expanded;
        let add = text.find("__pcu_capture . add").unwrap();
        let multiply = text.find("__pcu_capture . mul").unwrap();
        let relu = text.find("__pcu_capture . relu").unwrap();
        assert!(add < multiply && multiply < relu);

        let nested_depth = 70;
        let expression = format!(
            "{}pcu::relu(lhs){}",
            "(".repeat(nested_depth),
            ")".repeat(nested_depth),
        );
        let source: ItemFn = syn::parse_str(&format!(
            "fn nested(lhs: &[f32]) -> Result<PcuTensor<f32>, PcuExecutionError> {{ {expression} }}"
        ))
        .expect("generated nested syntax parses");
        let error = expand_owned_return(&source, &syn::parse_quote!(::pcu_alias))
            .expect_err("excessive nesting is rejected before recursive expansion");
        assert!(
            error.to_string().contains("maximum nesting depth"),
            "{error}"
        );
    }

    #[test]
    fn nested_marked_helpers_are_captured_and_literals_are_rejected() {
        let source: ItemFn = syn::parse_quote! {
            fn nested(lhs: &[f32], rhs: &[f32]) -> Result<PcuTensor<f32>, PcuExecutionError> {
                Ok(pcu::relu(add_pair(lhs, rhs)?)?)
            }
        };
        let expanded = expand(&source);
        let text = expanded;
        assert!(
            text.find("add_pair :: __pcu_capture").unwrap()
                < text.find("__pcu_capture . relu").unwrap()
        );

        let literal: ItemFn = syn::parse_quote! {
            fn nested(lhs: &[f32]) -> Result<PcuTensor<f32>, PcuExecutionError> {
                Ok(pcu::relu(lhs + 1.0)?)
            }
        };
        assert!(expand_owned_return(&literal, &syn::parse_quote!(::pcu_alias)).is_err());
    }

    #[test]
    fn companion_keeps_cfg_but_not_function_only_attributes() {
        let source: ItemFn = syn::parse_quote! {
            #[cfg(feature = "special")]
            #[inline(always)]
            fn relu(input: &[f32]) -> Result<PcuTensor<f32>, PcuExecutionError> {
                pcu::relu(input)
            }
        };
        let expanded = expand(&source);
        assert!(expanded.contains("cfg (feature = \"special\")"));
        assert!(!expanded.contains("mod relu { # [inline"));
    }

    #[test]
    fn generated_capture_local_avoids_user_bindings() {
        let source: ItemFn = syn::parse_quote! {
            fn relu(__pcu_capture: &[f32]) -> Result<PcuTensor<f32>, PcuExecutionError> {
                let __pcu_capture_ = pcu::relu(__pcu_capture);
                Ok(__pcu_capture_)
            }
        };
        let expanded = expand(&source);
        assert!(expanded.contains("___pcu_capture"));
    }

    #[test]
    fn control_flow_and_arbitrary_host_calls_are_not_lowered() {
        let branch: ItemFn = syn::parse_quote! {
            fn relu(input: &[f32]) -> Result<PcuTensor<f32>, PcuExecutionError> {
                if input.is_empty() { pcu::identity(input) } else { pcu::relu(input) }
            }
        };
        assert!(expand_owned_return(&branch, &syn::parse_quote!(::pcu_alias)).is_err());
        let invalid: ItemFn = syn::parse_quote! {
            fn relu(input: &[f32]) -> Result<PcuTensor<f32>, PcuExecutionError> {
                let host = vec![1.0_f32];
                Ok(input)
            }
        };
        assert!(expand_owned_return(&invalid, &syn::parse_quote!(::pcu_alias)).is_err());

        let shadowed_helper: ItemFn = syn::parse_quote! {
            fn relu(input: &[f32]) -> Result<PcuTensor<f32>, PcuExecutionError> {
                let relu = pcu::identity(input)?;
                let output = relu(input)?;
                Ok(output)
            }
        };
        assert!(expand_owned_return(&shadowed_helper, &syn::parse_quote!(::pcu_alias)).is_err());
    }
}
