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

/// Expand the first rank-one f32 owned-result composition profile.
pub fn expand_owned_return(
    function: &ItemFn,
    crate_path: &Path,
) -> Result<proc_macro2::TokenStream, Error> {
    let input_idents = validate_owned_function(function)?;
    let companion_crate_path = super::rebase_companion_path(crate_path.clone());
    let mut source_names = source_binding_names(function);
    source_names.extend(input_idents.iter().cloned());
    let capture_ident = fresh_ident(
        "__pcu_capture",
        &input_idents[0],
        &source_names.iter().collect::<Vec<_>>(),
    );
    let parsed = parse_program(function, &input_idents, &capture_ident)?;
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
    let capture_marker = format_ident!("__PcuCaptureRecursionMarker");
    let emission = OwnedEmission {
        function,
        crate_path: &companion_crate_path,
        input_idents: &input_idents,
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
    input_idents: &'a [syn::Ident],
    capture_ident: &'a syn::Ident,
    wrapper_marker: &'a syn::Ident,
    capture_marker: &'a syn::Ident,
    site_ident: &'a syn::Ident,
    source_ident: &'a syn::Ident,
    entry_capture_ident: &'a syn::Ident,
    program: &'a CapturedProgram,
}

fn emit_owned_items(emission: &OwnedEmission<'_>) -> proc_macro2::TokenStream {
    let OwnedEmission {
        function,
        crate_path: companion_crate_path,
        input_idents,
        capture_ident,
        wrapper_marker,
        capture_marker,
        site_ident,
        source_ident,
        entry_capture_ident,
        program: parsed,
    } = emission;
    let input_count = input_idents.len();
    let input_arguments = input_idents.iter().map(|ident| {
        quote! {
            #ident: &(impl #companion_crate_path::global::PcuTensorSource<f32> + ?Sized)
        }
    });
    let sources = input_idents.iter().map(|ident| {
        quote! {
            #companion_crate_path::global::PcuTensorSource::as_tensor_source(#ident)
                .map_err(#companion_crate_path::global::argument_error)?
        }
    });
    let input_type = quote! { #companion_crate_path::global::PcuTensorGraphValue };
    let capture_parameters = input_idents
        .iter()
        .map(|ident| quote! { #ident: #input_type });
    let capture_values = (0..input_count)
        .map(syn::Index::from)
        .map(|index| quote! { __pcu_inputs[#index] });
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
    let capture_body = &parsed.statements;
    let output_tokens = &parsed.output;
    quote! {
        #(#wrapper_attrs)*
        #[allow(dead_code)]
        #function_visibility fn #function_ident(
            #(#input_arguments),*
        ) #output {
            struct #wrapper_marker;
            static #site_ident: #companion_crate_path::global::PcuHostCallSite =
                #companion_crate_path::global::PcuHostCallSite::new();
            let #source_ident = [#(#sources),*];
            #companion_crate_path::global::call_owned_tensor_capture(
                &#site_ident,
                ::core::any::TypeId::of::<#wrapper_marker>(),
                #source_ident,
                #function_ident::__pcu_capture_entry,
            )
        }

        #[doc(hidden)]
        #(#companion_cfg)*
        #function_visibility mod #function_ident {
            #[allow(unused_imports)]
            use super::*;

            struct #capture_marker;

            pub fn __pcu_capture_entry(
                #entry_capture_ident: &mut #companion_crate_path::global::PcuTensorGraphCapture,
                __pcu_inputs: [#input_type; #input_count],
            ) -> ::core::result::Result<#input_type, #companion_crate_path::PcuExecutionError> {
                __pcu_capture(#entry_capture_ident, #(#capture_values),*)
            }

            // This cannot widen access beyond the companion module's original function visibility.
            // The module carries the source function's visibility; this child must be public
            // so a restricted module's permitted ancestors can reach the companion as well.
            pub fn __pcu_capture(
                #capture_ident: &mut #companion_crate_path::global::PcuTensorGraphCapture,
                #(#capture_parameters),*
            ) -> ::core::result::Result<#input_type, #companion_crate_path::PcuExecutionError> {
                #capture_ident.enter(::core::any::TypeId::of::<#capture_marker>())?;
                let __pcu_capture_result = (|| -> ::core::result::Result<
                    #input_type,
                    #companion_crate_path::PcuExecutionError,
                > {
                    #(#capture_body)*
                    ::core::result::Result::Ok(#output_tokens)
                })();
                #capture_ident.leave();
                __pcu_capture_result
            }
        }
    }
}

fn validate_owned_function(function: &ItemFn) -> Result<Vec<syn::Ident>, Error> {
    if function.sig.asyncness.is_some()
        || function.sig.constness.is_some()
        || function.sig.unsafety.is_some()
        || function.sig.abi.is_some()
        || !function.sig.generics.params.is_empty()
        || function.sig.generics.where_clause.is_some()
        || function.sig.variadic.is_some()
    {
        return Err(Error::new_spanned(
            &function.sig,
            "owned tensor compositions must be plain, non-generic, safe Rust functions",
        ));
    }
    if !supported_return_type(&function.sig.output) {
        return Err(Error::new_spanned(
            &function.sig.output,
            "owned tensor composition must return `Result<PcuTensor<f32>, PcuExecutionError>`",
        ));
    }
    if function.sig.inputs.is_empty() || function.sig.inputs.len() > 32 {
        return Err(Error::new_spanned(
            &function.sig.inputs,
            "owned tensor compositions require between 1 and 32 rank-one `&[f32]` inputs",
        ));
    }
    let mut input_idents = Vec::with_capacity(function.sig.inputs.len());
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
        if !is_f32_slice(argument.ty.as_ref()) {
            return Err(Error::new_spanned(
                argument,
                "each owned tensor composition input must have type `&[f32]`",
            ));
        }
        if input_idents.contains(&pattern.ident) {
            return Err(Error::new_spanned(
                &argument.pat,
                "owned tensor composition input names must be unique",
            ));
        }
        input_idents.push(pattern.ident.clone());
    }
    Ok(input_idents)
}

struct CapturedProgram {
    statements: Vec<proc_macro2::TokenStream>,
    output: proc_macro2::TokenStream,
}

#[derive(Clone)]
enum Value {
    Input(usize),
    Temporary(syn::Ident),
}

enum Operation {
    Identity,
    Relu,
    Add,
    Sub,
    Mul,
    Helper(Path),
}

fn parse_program(
    function: &ItemFn,
    inputs: &[syn::Ident],
    capture: &syn::Ident,
) -> Result<CapturedProgram, Error> {
    if function.block.stmts.is_empty() {
        return Err(composition_body_error(&function.block));
    }
    let (statements, terminal) = function
        .block
        .stmts
        .split_at(function.block.stmts.len() - 1);
    let mut bindings: Vec<(syn::Ident, Value)> = Vec::new();
    let mut emitted = Vec::new();
    let mut generated = source_binding_names(function);
    generated.extend_from_slice(inputs);
    generated.push(capture.clone());
    for statement in statements {
        let syn::Stmt::Local(local) = statement else {
            return Err(composition_body_error(statement));
        };
        let (ident, expression) = local_binding(local)?;
        if inputs.iter().any(|input| input == ident)
            || bindings.iter().any(|(name, _)| name == ident)
        {
            return Err(Error::new_spanned(
                &local.pat,
                "owned tensor composition does not allow shadowed value names",
            ));
        }
        let value = lower_expression(
            expression,
            inputs,
            &bindings,
            capture,
            &mut generated,
            &mut emitted,
            0,
        )?;
        bindings.push((ident.clone(), value));
    }
    let [syn::Stmt::Expr(expression, None)] = terminal else {
        return Err(composition_body_error(&function.block));
    };
    let expression = unwrap_result_return(expression)?;
    let output = lower_expression(
        expression,
        inputs,
        &bindings,
        capture,
        &mut generated,
        &mut emitted,
        0,
    )?;
    let output = value_tokens(&output, inputs);
    Ok(CapturedProgram {
        statements: emitted,
        output,
    })
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

fn lower_expression(
    expression: &Expr,
    inputs: &[syn::Ident],
    bindings: &[(syn::Ident, Value)],
    capture: &syn::Ident,
    generated: &mut Vec<syn::Ident>,
    emitted: &mut Vec<proc_macro2::TokenStream>,
    depth: usize,
) -> Result<Value, Error> {
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
            lower_expression(
                &syntax.expr,
                inputs,
                bindings,
                capture,
                generated,
                emitted,
                depth + 1,
            )
        }
        Expr::Paren(paren) => {
            if !paren.attrs.is_empty() {
                return Err(composition_body_error(expression));
            }
            lower_expression(
                &paren.expr,
                inputs,
                bindings,
                capture,
                generated,
                emitted,
                depth + 1,
            )
        }
        Expr::Group(group) => {
            if !group.attrs.is_empty() {
                return Err(composition_body_error(expression));
            }
            lower_expression(
                &group.expr,
                inputs,
                bindings,
                capture,
                generated,
                emitted,
                depth + 1,
            )
        }
        Expr::Binary(binary) => {
            lower_binary(binary, inputs, bindings, capture, generated, emitted, depth)
        }
        Expr::Call(call) => lower_call(call, inputs, bindings, capture, generated, emitted, depth),
        Expr::Reference(reference) => {
            if reference.mutability.is_some() || !reference.attrs.is_empty() {
                return Err(composition_body_error(expression));
            }
            lower_expression(
                &reference.expr,
                inputs,
                bindings,
                capture,
                generated,
                emitted,
                depth + 1,
            )
        }
        Expr::Path(_) => expression_value(expression, inputs, bindings)?
            .ok_or_else(|| composition_body_error(expression)),
        _ => Err(composition_body_error(expression)),
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

fn lower_binary(
    binary: &syn::ExprBinary,
    inputs: &[syn::Ident],
    bindings: &[(syn::Ident, Value)],
    capture: &syn::Ident,
    generated: &mut Vec<syn::Ident>,
    emitted: &mut Vec<proc_macro2::TokenStream>,
    depth: usize,
) -> Result<Value, Error> {
    if !binary.attrs.is_empty() {
        return Err(composition_body_error(binary));
    }
    let operation = match &binary.op {
        syn::BinOp::Add(_) => Operation::Add,
        syn::BinOp::Sub(_) => Operation::Sub,
        syn::BinOp::Mul(_) => Operation::Mul,
        _ => return Err(composition_body_error(binary)),
    };
    let lhs = lower_expression(
        &binary.left,
        inputs,
        bindings,
        capture,
        generated,
        emitted,
        depth + 1,
    )?;
    let rhs = lower_expression(
        &binary.right,
        inputs,
        bindings,
        capture,
        generated,
        emitted,
        depth + 1,
    )?;
    Ok(emit_captured_operation(
        operation,
        &[lhs, rhs],
        inputs,
        capture,
        generated,
        emitted,
    ))
}

fn lower_call(
    call: &ExprCall,
    inputs: &[syn::Ident],
    bindings: &[(syn::Ident, Value)],
    capture: &syn::Ident,
    generated: &mut Vec<syn::Ident>,
    emitted: &mut Vec<proc_macro2::TokenStream>,
    depth: usize,
) -> Result<Value, Error> {
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
            .any(|segment| !matches!(segment.arguments, syn::PathArguments::None))
    {
        return Err(composition_body_error(&call.func));
    }
    let operation = operation_for_path(&path.path, &call.func, inputs, bindings)?;
    let expected = match &operation {
        Operation::Identity | Operation::Relu => Some(1),
        Operation::Add | Operation::Sub | Operation::Mul => Some(2),
        Operation::Helper(_) => None,
    };
    if expected.is_some_and(|expected| call.args.len() != expected)
        || (call.args.is_empty() && matches!(&operation, Operation::Helper(_)))
    {
        return Err(composition_body_error(&call.args));
    }
    let mut values = Vec::with_capacity(call.args.len());
    for argument in &call.args {
        values.push(lower_expression(
            argument,
            inputs,
            bindings,
            capture,
            generated,
            emitted,
            depth + 1,
        )?);
    }
    Ok(emit_captured_operation(
        operation, &values, inputs, capture, generated, emitted,
    ))
}

fn operation_for_path(
    path: &Path,
    span: &impl quote::ToTokens,
    inputs: &[syn::Ident],
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
        return match path.segments[1].ident.to_string().as_str() {
            "relu" => Ok(Operation::Relu),
            "identity" => Ok(Operation::Identity),
            "add" => Ok(Operation::Add),
            "sub" => Ok(Operation::Sub),
            "mul" => Ok(Operation::Mul),
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
    Ok(Operation::Helper(companion_path(path)))
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

fn emit_captured_operation(
    operation: Operation,
    sources: &[Value],
    inputs: &[syn::Ident],
    capture: &syn::Ident,
    generated: &mut Vec<syn::Ident>,
    emitted: &mut Vec<proc_macro2::TokenStream>,
) -> Value {
    let reserved = generated.iter().collect::<Vec<_>>();
    let temporary = fresh_ident("__pcu_graph_value", &inputs[0], &reserved);
    generated.push(temporary.clone());
    let values = sources.iter().map(|value| value_tokens(value, inputs));
    let call = match operation {
        Operation::Identity => quote! { #capture.identity(#(#values),*)? },
        Operation::Relu => quote! { #capture.relu(#(#values),*)? },
        Operation::Add => quote! { #capture.add(#(#values),*)? },
        Operation::Sub => quote! { #capture.sub(#(#values),*)? },
        Operation::Mul => quote! { #capture.mul(#(#values),*)? },
        Operation::Helper(path) => quote! { #path(#capture, #(#values),*)? },
    };
    emitted.push(quote! { let #temporary = #call; });
    Value::Temporary(temporary)
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
    if let Some(index) = inputs.iter().position(|input| input == identifier) {
        return Ok(Some(Value::Input(index)));
    }
    Ok(bindings
        .iter()
        .find(|(name, _)| name == identifier)
        .map(|(_, value)| value.clone()))
}

fn expression_path_ident(path: &syn::ExprPath) -> Result<Option<&syn::Ident>, Error> {
    if !path.attrs.is_empty() || path.qself.is_some() || path.path.leading_colon.is_some() {
        return Err(composition_body_error(path));
    }
    Ok(path.path.get_ident())
}

fn value_tokens(value: &Value, inputs: &[syn::Ident]) -> proc_macro2::TokenStream {
    match value {
        Value::Input(index) => {
            let input = &inputs[*index];
            quote! { #input }
        }
        Value::Temporary(name) => quote! { #name },
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

fn is_f32_slice(ty: &Type) -> bool {
    let Type::Reference(reference) = ty else {
        return false;
    };
    if reference.mutability.is_some() || reference.lifetime.is_some() {
        return false;
    }
    let Type::Slice(slice) = reference.elem.as_ref() else {
        return false;
    };
    matches!(slice.elem.as_ref(), Type::Path(path) if path.qself.is_none() && path.path.is_ident("f32"))
}

fn supported_return_type(output: &ReturnType) -> bool {
    let ReturnType::Type(_, ty) = output else {
        return false;
    };
    let Type::Path(result) = ty.as_ref() else {
        return false;
    };
    let Some(result_segment) = result.path.segments.last() else {
        return false;
    };
    if result_segment.ident != "Result" {
        return false;
    }
    let syn::PathArguments::AngleBracketed(result_args) = &result_segment.arguments else {
        return false;
    };
    if result_args.args.len() != 2 {
        return false;
    }
    let Some(syn::GenericArgument::Type(Type::Path(tensor))) = result_args.args.first() else {
        return false;
    };
    let Some(tensor_segment) = tensor.path.segments.last() else {
        return false;
    };
    if tensor_segment.ident != "PcuTensor" {
        return false;
    }
    let syn::PathArguments::AngleBracketed(tensor_args) = &tensor_segment.arguments else {
        return false;
    };
    if tensor_args.args.len() != 1
        || !matches!(tensor_args.args.first(), Some(syn::GenericArgument::Type(Type::Path(scalar))) if scalar.path.is_ident("f32"))
    {
        return false;
    }
    matches!(result_args.args.last(), Some(syn::GenericArgument::Type(Type::Path(error))) if error.path.segments.last().is_some_and(|segment| segment.ident == "PcuExecutionError"))
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
        assert!(expanded.contains("PcuTensorSource :: as_tensor_source (lhs)"));
        assert!(expanded.contains("PcuTensorSource :: as_tensor_source (rhs)"));
        assert!(expanded.contains("PcuTensorGraphValue ; 2usize"));
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
        assert!(error.to_string().contains("maximum nesting depth"));
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
