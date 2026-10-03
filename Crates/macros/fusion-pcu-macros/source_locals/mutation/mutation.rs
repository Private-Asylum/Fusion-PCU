//! Straight-line mutation creates new SSA values, never mutable runtime slots.
use super::RuntimeExprEmitter;
#[rustfmt::skip]
use syn::{
    BinOp,
    Error,
    Expr,
    ExprBinary,
    Ident,
};
use syn::spanned::Spanned;

pub(super) const fn is_compound_assignment(expression: &ExprBinary) -> bool {
    matches!(
        expression.op,
        BinOp::AddAssign(_)
            | BinOp::SubAssign(_)
            | BinOp::MulAssign(_)
            | BinOp::DivAssign(_)
            | BinOp::RemAssign(_)
            | BinOp::BitXorAssign(_)
            | BinOp::BitAndAssign(_)
            | BinOp::BitOrAssign(_)
            | BinOp::ShlAssign(_)
            | BinOp::ShrAssign(_)
    )
}

/// A plain local has no address/index side effects. Reuse its previous SSA ID,
/// then emit the ordinary checked RHS before binding its new SSA value.
pub(super) fn compound_expression(expression: &ExprBinary) -> Result<(&Ident, Expr), Error> {
    let Expr::Path(path) = expression.left.as_ref() else {
        return Err(Error::new_spanned(
            &expression.left,
            "PCU compound assignment currently requires a plain scalar local; resource updates need a separate indexed load/store contract",
        ));
    };
    let Some(name) = path.path.get_ident().filter(|_| path.qself.is_none()) else {
        return Err(Error::new_spanned(
            path,
            "PCU compound assignment requires a plain scalar local",
        ));
    };
    let op = match expression.op {
        BinOp::AddAssign(_) => syn::parse_quote_spanned!(expression.op.span()=> +),
        BinOp::SubAssign(_) => syn::parse_quote_spanned!(expression.op.span()=> -),
        BinOp::MulAssign(_) => syn::parse_quote_spanned!(expression.op.span()=> *),
        BinOp::DivAssign(_) => syn::parse_quote_spanned!(expression.op.span()=> /),
        _ => {
            return Err(Error::new_spanned(
                expression,
                "PCU scalar compound assignments support +=, -=, *= and checked floating /=; other operators require a specified arithmetic contract",
            ));
        }
    };
    Ok((
        name,
        Expr::Binary(ExprBinary {
            attrs: expression.attrs.clone(),
            left: expression.left.clone(),
            op,
            right: expression.right.clone(),
        }),
    ))
}

#[derive(Default)]
pub(super) struct LocalScope {
    declarations: Vec<(Ident, bool)>,
}

impl LocalScope {
    pub(super) fn declare(&mut self, name: &Ident, mutable: bool) {
        self.declarations.push((name.clone(), mutable));
    }

    pub(super) fn assign(
        &self,
        emitter: &mut RuntimeExprEmitter<'_>,
        name: &Ident,
        expression: &Expr,
    ) -> Result<(), Error> {
        if !self
            .declarations
            .iter()
            .rev()
            .find(|(declared, _)| declared == name)
            .is_some_and(|(_, mutable)| *mutable)
        {
            return Err(Error::new(
                name.span(),
                "PCU scalar assignment requires an existing mutable local; shadowing replaces its mutability",
            ));
        }
        let scalar = emitter
            .values
            .iter()
            .rev()
            .find(|(declared, _, _)| declared == name)
            .expect("every scalar declaration has an emitted SSA value")
            .2;
        let previous_expected = emitter.expected_scalar;
        emitter.expected_scalar = scalar;
        // The RHS sees the previous value. Existing aliases retain its SSA ID.
        let result = emitter.emit_expr(expression);
        emitter.expected_scalar = previous_expected;
        let (value, actual_scalar) = result?;
        if actual_scalar != scalar {
            return Err(Error::new_spanned(
                expression,
                "PCU scalar assignment cannot change a local's type",
            ));
        }
        emitter.values.push((name.clone(), value, scalar));
        Ok(())
    }
}
