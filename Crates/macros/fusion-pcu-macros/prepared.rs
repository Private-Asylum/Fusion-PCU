use proc_macro2::TokenStream;
#[rustfmt::skip]
use quote::{
    format_ident,
    quote,
    ToTokens,
};
#[rustfmt::skip]
use syn::{
    GenericParam,
    Generics,
    Ident,
    ItemFn,
    Lifetime,
    Path,
    Type,
    Visibility,
};

pub struct Argument {
    pub ident: Ident,
    pub binding: u32,
    pub read_write: bool,
    pub scalar: TokenStream,
    pub ty: Type,
}

pub struct Input<'a> {
    pub pcu: &'a Path,
    pub function: &'a ItemFn,
    pub visibility: &'a Visibility,
    pub function_ident: &'a Ident,
    pub arguments: &'a [Argument],
    pub generic_arguments: &'a [TokenStream],
    pub prepare_error: &'a TokenStream,
    pub bindings_call: TokenStream,
    pub builder_call: TokenStream,
}

#[allow(clippy::cognitive_complexity, clippy::too_many_lines)]
pub fn generate(input: Input<'_>) -> TokenStream {
    let Input {
        pcu,
        function,
        visibility: vis,
        function_ident,
        arguments: args,
        generic_arguments,
        prepare_error,
        bindings_call,
        builder_call,
    } = input;
    let host_prepare_fn = format_ident!("{}_prepare", function_ident);
    let device_prepare_fn = format_ident!("{}_device", host_prepare_fn);
    let backend_ident = unique_generic_ident("__PcuBackend", &function.sig.generics);
    let device_backend_ident = unique_generic_ident("__PcuDeviceBackend", &function.sig.generics);
    let mut prepare_params = function.sig.generics.params.clone();
    prepare_params.push(syn::parse_quote!(#backend_ident));
    let mut device_prepare_params = function.sig.generics.params.clone();
    device_prepare_params.push(syn::parse_quote!(#device_backend_ident));
    let mut prepare_where = function
        .sig
        .generics
        .where_clause
        .clone()
        .unwrap_or_else(|| syn::parse_quote!(where #backend_ident: #pcu::PcuHostKernelBackend));
    if function.sig.generics.where_clause.is_some() {
        prepare_where.predicates.push(syn::parse_quote!(
            #backend_ident: #pcu::PcuHostKernelBackend
        ));
    }
    let mut device_prepare_where = function.sig.generics.where_clause.clone().unwrap_or_else(
        || syn::parse_quote!(where #device_backend_ident: #pcu::PcuDeviceKernelBackend),
    );
    if function.sig.generics.where_clause.is_some() {
        device_prepare_where.predicates.push(syn::parse_quote!(
            #device_backend_ident: #pcu::PcuDeviceKernelBackend
        ));
    }
    let mut prepared_ident = format_ident!("__pcu_prepared");
    while args.iter().any(|arg| arg.ident == prepared_ident)
        || function
            .sig
            .generics
            .params
            .iter()
            .any(|param| match param {
                GenericParam::Type(param) => param.ident == prepared_ident,
                GenericParam::Const(param) => param.ident == prepared_ident,
                GenericParam::Lifetime(param) => param.lifetime.ident == prepared_ident,
            })
    {
        prepared_ident = format_ident!("_{}", prepared_ident);
    }
    let argument_idents = args.iter().map(|arg| &arg.ident).collect::<Vec<_>>();
    let argument_types = args.iter().map(|arg| &arg.ty).collect::<Vec<_>>();
    let call_lifetimes = argument_lifetimes(args.len(), &function.sig.generics, function);
    let host_call_types = args
        .iter()
        .zip(&call_lifetimes)
        .map(|(arg, lifetime)| {
            let mut ty = arg.ty.clone();
            if let Type::Reference(reference) = &mut ty {
                reference.lifetime = Some(lifetime.clone());
            }
            ty
        })
        .collect::<Vec<_>>();
    let argument_count = args.len();
    let host_arguments = args.iter().map(|arg| {
        let ident = &arg.ident;
        let slot = arg.binding;
        let target = quote! { #pcu::PcuBindingRef::new(0, #slot) };
        if arg.read_write {
            quote! { #pcu::PcuHostArgument::read_write(#target, #ident) }
        } else {
            quote! { #pcu::PcuHostArgument::read(#target, #ident) }
        }
    });
    let capture_generics = generic_arguments
        .iter()
        .cloned()
        .chain(core::iter::once(backend_ident.to_token_stream()))
        .collect::<Vec<_>>();
    let device_capture_generics = generic_arguments
        .iter()
        .cloned()
        .chain(core::iter::once(device_backend_ident.to_token_stream()))
        .collect::<Vec<_>>();
    let device_argument_types = args.iter().zip(&call_lifetimes).map(|(arg, lifetime)| {
        let scalar = &arg.scalar;
        let borrow = if arg.read_write {
            quote! { &#lifetime mut }
        } else {
            quote! { &#lifetime }
        };
        quote! {
            #borrow #pcu::PcuDeviceBuffer<#scalar, <#device_backend_ident as #pcu::PcuDeviceKernelBackend>::Resource>
        }
    }).collect::<Vec<_>>();
    let device_closure_types = args.iter().map(|arg| {
        let scalar = &arg.scalar;
        let borrow = if arg.read_write { quote! { &mut } } else { quote! { & } };
        quote! {
            #borrow #pcu::PcuDeviceBuffer<#scalar, <#device_backend_ident as #pcu::PcuDeviceKernelBackend>::Resource>
        }
    }).collect::<Vec<_>>();
    let device_arguments = args.iter().map(|arg| {
        let ident = &arg.ident;
        let slot = arg.binding;
        let target = quote! { #pcu::PcuBindingRef::new(0, #slot) };
        let access = if arg.read_write {
            quote! { read_write }
        } else {
            quote! { read }
        };
        quote! { #pcu::PcuDeviceArgument::#access(#target, #ident) }
    });

    quote! {
        // Optional execution entry points may be unused by kernels that only inspect their IR.
        #[allow(dead_code)]
        #vis fn #host_prepare_fn <#prepare_params>(backend: &#backend_ident)
            -> ::core::result::Result<
                impl for<#(#call_lifetimes),*> ::core::ops::FnMut(#(#host_call_types),*)
                    -> ::core::result::Result<(), <<#backend_ident as #pcu::PcuHostKernelBackend>::Prepared as #pcu::PcuPreparedHostKernel>::Error>
                    + use<#(#capture_generics),*>,
                #pcu::PcuKernelPrepareError<#prepare_error, <#backend_ident as #pcu::PcuHostKernelBackend>::Error>,
            >
        #prepare_where
        {
            let bindings = #bindings_call;
            let builder = #builder_call.map_err(#pcu::PcuKernelPrepareError::Ir)?;
            let mut #prepared_ident = <#backend_ident as #pcu::PcuHostKernelBackend>::prepare_host_kernel(backend, &builder.ir())
                .map_err(#pcu::PcuKernelPrepareError::Backend)?;
            ::core::result::Result::Ok(move |#(#argument_idents: #argument_types),*| {
                let mut arguments: [#pcu::PcuHostArgument<'_>; #argument_count] = [#(#host_arguments),*];
                <<#backend_ident as #pcu::PcuHostKernelBackend>::Prepared as #pcu::PcuPreparedHostKernel>::call(
                    &mut #prepared_ident,
                    &mut arguments,
                )
            })
        }

        // The resident-device entry point is likewise optional for IR-only users.
        #[allow(dead_code)]
        #vis fn #device_prepare_fn <#device_prepare_params>(backend: &#device_backend_ident)
            -> ::core::result::Result<
                impl for<#(#call_lifetimes),*> ::core::ops::FnMut(#(#device_argument_types),*)
                    -> ::core::result::Result<(), <<#device_backend_ident as #pcu::PcuDeviceKernelBackend>::Prepared as #pcu::PcuPreparedDeviceKernel>::Error>
                    + use<#(#device_capture_generics),*>,
                #pcu::PcuKernelPrepareError<#prepare_error, <#device_backend_ident as #pcu::PcuDeviceKernelBackend>::Error>,
            >
        #device_prepare_where
        {
            let bindings = #bindings_call;
            let builder = #builder_call.map_err(#pcu::PcuKernelPrepareError::Ir)?;
            let mut #prepared_ident = <#device_backend_ident as #pcu::PcuDeviceKernelBackend>::prepare_device_kernel(backend, &builder.ir())
                .map_err(#pcu::PcuKernelPrepareError::Backend)?;
            ::core::result::Result::Ok(move |#(#argument_idents: #device_closure_types),*| {
                let mut arguments: [#pcu::PcuDeviceArgument<'_, <#device_backend_ident as #pcu::PcuDeviceKernelBackend>::Resource>; #argument_count] = [#(#device_arguments),*];
                <<#device_backend_ident as #pcu::PcuDeviceKernelBackend>::Prepared as #pcu::PcuPreparedDeviceKernel>::call(
                    &mut #prepared_ident,
                    &mut arguments,
                )
            })
        }
    }
}

fn argument_lifetimes(count: usize, generics: &Generics, function: &ItemFn) -> Vec<Lifetime> {
    (0..count)
        .map(|index| {
            let mut name = format_ident!("__pcu_arg{index}");
            while generics.params.iter().any(|param| {
                matches!(param, GenericParam::Lifetime(param) if param.lifetime.ident == name)
            }) {
                name = format_ident!("_{}", name);
            }
            Lifetime::new(&format!("'{name}"), function.sig.ident.span())
        })
        .collect()
}

fn unique_generic_ident(name: &str, generics: &Generics) -> Ident {
    let mut ident = format_ident!("{name}");
    while generics.params.iter().any(|param| match param {
        GenericParam::Type(param) => param.ident == ident,
        GenericParam::Const(param) => param.ident == ident,
        GenericParam::Lifetime(param) => param.lifetime.ident == ident,
    }) {
        ident = format_ident!("_{ident}");
    }
    ident
}
