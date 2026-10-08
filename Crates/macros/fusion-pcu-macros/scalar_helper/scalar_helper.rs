//! Bounded per-function scalar signatures; source statements share typed SSA lowering.
#[rustfmt::skip]
use super::{
    scalar::concrete_scalar,
    transparent_type,
    ScalarKind,
};
#[rustfmt::skip]
use syn::{
    Error,
    Expr,
    FnArg,
    Ident,
    ItemFn,
    Pat,
    ReturnType,
    Stmt,
    Type,
};

pub struct ScalarHelper {
    pub ident: Ident,
    pub parameters: Vec<(Ident, bool)>,
    pub scalar: ScalarKind,
    pub body: Expr,
    pub locals: Vec<Stmt>,
}

fn scalar_kind(ty: &Type) -> Option<ScalarKind> {
    match transparent_type(ty) {
        Type::Path(path) if path.qself.is_none() => concrete_scalar(&path.path),
        _ => None,
    }
}

pub fn parse(function: &ItemFn) -> Result<ScalarHelper, Error> {
    if function.sig.asyncness.is_some()
        || function.sig.constness.is_some()
        || !matches!(function.sig.safety, syn::Safety::Default)
        || function.sig.abi.is_some()
        || !function.sig.generics.params.is_empty()
        || function.sig.generics.where_clause.is_some()
        || function.sig.variadic.is_some()
    {
        return Err(Error::new_spanned(
            &function.sig,
            "PCU helpers must be plain, non-generic, safe Rust functions",
        ));
    }
    let ReturnType::Type(_, return_type) = &function.sig.output else {
        return Err(Error::new_spanned(
            &function.sig.output,
            "PCU helpers must declare an explicit supported scalar return type",
        ));
    };
    let Some(scalar) = scalar_kind(return_type) else {
        return Err(Error::new_spanned(
            return_type,
            "this PCU helper profile supports f32/f64 and concrete fixed-width integer scalar types",
        ));
    };
    let mut parameters = Vec::with_capacity(function.sig.inputs.len());
    for input in &function.sig.inputs {
        let FnArg::Typed(argument) = input else {
            return Err(Error::new_spanned(
                input,
                "PCU helpers cannot have a receiver",
            ));
        };
        let Pat::Ident(pattern) = argument.pat.as_ref() else {
            return Err(Error::new_spanned(
                &argument.pat,
                "PCU helper parameters must be simple identifiers",
            ));
        };
        if !argument.attrs.is_empty()
            || !pattern.attrs.is_empty()
            || pattern.by_ref.is_some()
            || pattern.subpat.is_some()
        {
            return Err(Error::new_spanned(
                argument,
                "PCU scalar helper parameters must be plain identifiers without attributes or patterns",
            ));
        }
        if scalar_kind(&argument.ty) != Some(scalar) {
            return Err(Error::new_spanned(
                &argument.ty,
                "PCU helper parameters must match the declared scalar result type",
            ));
        }
        if parameters
            .iter()
            .any(|(parameter, _): &(Ident, bool)| parameter == &pattern.ident)
        {
            return Err(Error::new_spanned(
                &pattern.ident,
                "PCU helper parameter names must be unique",
            ));
        }
        parameters.push((pattern.ident.clone(), pattern.mutability.is_some()));
    }
    let [locals @ .., Stmt::Expr(body, None)] = function.block.stmts.as_slice() else {
        return Err(Error::new_spanned(
            &function.block,
            "PCU scalar helpers require initialized locals or straight-line reassignment followed by a scalar result expression; control flow remains unsupported",
        ));
    };
    Ok(ScalarHelper {
        ident: function.sig.ident.clone(),
        parameters,
        scalar,
        body: body.clone(),
        locals: locals.to_vec(),
    })
}

#[cfg(test)]
#[path = "tests/tests.rs"]
mod tests;
