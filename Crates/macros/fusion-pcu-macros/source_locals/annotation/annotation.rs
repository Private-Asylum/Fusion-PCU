//! Scalar annotations constrain typed SSA; they never request a conversion.
#[rustfmt::skip]
use super::super::{
    scalar,
    transparent_type,
    RuntimeExprEmitter,
    ScalarKind,
};
#[rustfmt::skip]
use syn::{
    Error,
    Local,
    Pat,
    PatIdent,
    Stmt,
    Type,
};

pub(super) fn pattern(local: &Local) -> Result<&PatIdent, Error> {
    let pat = match &local.pat {
        Pat::Type(annotation) if annotation.attrs.is_empty() => annotation.pat.as_ref(),
        pat => pat,
    };
    let Pat::Ident(pattern) = pat else {
        return Err(Error::new_spanned(
            &local.pat,
            "PCU scalar locals require one plain name, optionally with a supported scalar type; destructuring remains unsupported",
        ));
    };
    if !pattern.attrs.is_empty() {
        return Err(Error::new_spanned(
            pattern,
            "PCU local pattern attributes are unsupported",
        ));
    }
    Ok(pattern)
}

pub(super) fn expected_type(
    statement: &Stmt,
    emitter: &RuntimeExprEmitter<'_>,
) -> Result<Option<ScalarKind>, Error> {
    let Stmt::Local(local) = statement else {
        unreachable!("scalar_local validated this statement")
    };
    let Pat::Type(annotation) = &local.pat else {
        return Ok(None);
    };
    let ty = transparent_type(&annotation.ty);
    if matches!(ty, Type::Infer(_)) {
        return Ok(None);
    }
    if let Type::Path(path) = ty
        && path.qself.is_none()
    {
        if let Some(kind) = scalar::concrete_scalar(&path.path) {
            return Ok(Some(kind));
        }
        if [
            emitter.generic_float,
            emitter.generic_integer,
            emitter.generic_transport,
        ]
        .into_iter()
        .flatten()
        .any(|generic| path.path.is_ident(generic))
        {
            return Ok(Some(ScalarKind::Generic));
        }
    }
    Err(Error::new_spanned(
        ty,
        "PCU local annotations support concrete primitive scalars, the function's sealed scalar parameter, or `_`; aliases, references and compound types require separate type resolution",
    ))
}
