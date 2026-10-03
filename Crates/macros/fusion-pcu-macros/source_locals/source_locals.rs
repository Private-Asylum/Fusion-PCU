//! Scalar source statements preserve evaluation order and typed SSA reuse.
#[rustfmt::skip]
use super::{
    generic_float,
    generic_integer,
    generic_transport,
    is_ident_expr,
    parse_matrix_locals,
    normalize_matrix_local_indices,
    ScalarKind,
    validate_assignment_target,
    MatrixIndexNormalizer,
    MatrixLocals,
    RuntimeExprEmitter,
};
use quote::quote;
use proc_macro2::TokenStream as TokenStream2;
use syn::visit_mut::VisitMut;
#[rustfmt::skip]
use syn::{
    Attribute,
    BinOp,
    Error,
    Expr,
    ExprAssign,
    Ident,
    Stmt,
};

#[path = "annotation/annotation.rs"]
mod annotation;
#[path = "mutation/mutation.rs"]
mod mutation;

#[derive(Default)]
struct ExpressionAttributes {
    error: Option<Error>,
}

impl VisitMut for ExpressionAttributes {
    fn visit_attribute_mut(&mut self, attribute: &mut Attribute) {
        if self.error.is_none() {
            self.error = Some(Error::new_spanned(
                attribute,
                "attributes inside PCU scalar statements are unsupported; conditional statement removal must not be silently ignored",
            ));
        }
    }
}

pub fn reject_store_attributes(assignment: &ExprAssign) -> Result<(), Error> {
    let mut assignment = assignment.clone();
    let mut validator = ExpressionAttributes::default();
    validator.visit_expr_assign_mut(&mut assignment);
    validator.error.map_or(Ok(()), Err)
}

fn scalar_local(statement: &Stmt) -> Result<(&Ident, &Expr, bool), Error> {
    let Stmt::Local(local) = statement else {
        return Err(Error::new_spanned(
            statement,
            "PCU scalar statements require initialized locals, local assignments or same-lane indexed stores",
        ));
    };
    let pattern = annotation::pattern(local)?;
    if !local.attrs.is_empty() || pattern.by_ref.is_some() || pattern.subpat.is_some() {
        return Err(Error::new_spanned(
            local,
            "PCU scalar locals must be plain bindings without local attributes",
        ));
    }
    let Some(initializer) = &local.init else {
        return Err(Error::new_spanned(
            local,
            "PCU scalar locals must have an initializer",
        ));
    };
    if initializer.diverge.is_some() {
        return Err(Error::new_spanned(
            &initializer.expr,
            "PCU scalar local initializers cannot diverge",
        ));
    }
    Ok((
        &pattern.ident,
        &initializer.expr,
        pattern.mutability.is_some(),
    ))
}

/// Preserve canonical coordinates independently of subsequent scalar locals.
pub fn split_coordinates<'a>(
    statements: &'a [Stmt],
    invocation: &Ident,
    stride: Option<Ident>,
) -> Result<(Option<MatrixLocals>, &'a [Stmt]), Error> {
    let looks_like_row = statements.first().is_some_and(|statement| {
        scalar_local(statement).is_ok_and(|(_, expression, _)| {
            matches!(expression, Expr::Binary(binary)
                if matches!(binary.op, BinOp::Div(_))
                && is_ident_expr(&binary.left, invocation))
        })
    });
    let locals = if looks_like_row {
        let [row, column, rest @ ..] = statements else {
            return Err(Error::new_spanned(
                &statements[0],
                "PCU matrix row coordinates require a matching column coordinate",
            ));
        };
        (
            Some(parse_matrix_locals(row, column, invocation, stride)?),
            rest,
        )
    } else {
        (None, statements)
    };
    for statement in locals.1 {
        if !matches!(statement, Stmt::Expr(Expr::Assign(_), Some(_)))
            && !matches!(statement, Stmt::Expr(Expr::Binary(binary), Some(_))
                if mutation::is_compound_assignment(binary))
        {
            scalar_local(statement)?;
        }
    }
    Ok(locals)
}

/// Typed lowering runs once during IR preparation. Each local initializer is
/// emitted in order, including unused checked computations with observable faults.
/// Repeated references reuse the SSA result instead of duplicating its expression.
pub fn lower(
    emitter: &mut RuntimeExprEmitter<'_>,
    statements: &[Stmt],
    const_generics: &[Ident],
    coordinates: Option<&MatrixLocals>,
    stride: Option<&Ident>,
) -> Result<(), Error> {
    lower_statements(
        emitter,
        statements,
        const_generics,
        coordinates,
        stride,
        true,
        &[],
    )
}

/// Helpers have no invocation binding. Their function name is not a reserved local.
pub fn lower_helper(
    emitter: &mut RuntimeExprEmitter<'_>,
    statements: &[Stmt],
    parameters: &[(Ident, bool)],
) -> Result<(), Error> {
    lower_statements(emitter, statements, &[], None, None, false, parameters)
}

/// Apply the same attribute and expression checks to a helper's final result.
pub fn lower_result(
    emitter: &mut RuntimeExprEmitter<'_>,
    expression: &Expr,
) -> Result<(TokenStream2, ScalarKind), Error> {
    let expression = prepare_expression(emitter, expression, None)?;
    emitter.emit_expr(&expression)
}

fn lower_statements(
    emitter: &mut RuntimeExprEmitter<'_>,
    statements: &[Stmt],
    const_generics: &[Ident],
    coordinates: Option<&MatrixLocals>,
    stride: Option<&Ident>,
    has_invocation: bool,
    parameters: &[(Ident, bool)],
) -> Result<(), Error> {
    let mut scope = mutation::LocalScope::default();
    for (name, mutable) in parameters {
        scope.declare(name, *mutable);
    }
    for statement in statements {
        if let Stmt::Expr(Expr::Binary(binary), Some(_)) = statement
            && mutation::is_compound_assignment(binary)
        {
            let (name, expression) = mutation::compound_expression(binary)?;
            let expression = prepare_expression(emitter, &expression, coordinates)?;
            scope.assign(emitter, name, &expression)?;
            continue;
        }
        if let Stmt::Expr(Expr::Assign(assignment), Some(_)) = statement {
            let assignment = if let Some(locals) = coordinates {
                normalize_matrix_local_indices(
                    assignment,
                    emitter.invocation_ident,
                    locals,
                    emitter.bindings,
                )?
            } else {
                assignment.clone()
            };
            reject_store_attributes(&assignment)?;
            if let Expr::Path(path) = assignment.left.as_ref() {
                let Some(name) = path.path.get_ident().filter(|_| path.qself.is_none()) else {
                    return Err(Error::new_spanned(
                        path,
                        "PCU local assignment requires a plain name",
                    ));
                };
                let expression = prepare_expression(emitter, &assignment.right, coordinates)?;
                scope.assign(emitter, name, &expression)?;
            } else {
                lower_store(emitter, &assignment)?;
            }
            continue;
        }
        let (name, expression, mutable) = scalar_local(statement)?;
        if emitter
            .bindings
            .iter()
            .any(|binding| binding.ident == *name)
            || (has_invocation && *name == *emitter.invocation_ident)
            || stride.is_some_and(|stride| stride == name)
            || const_generics.contains(name)
            || coordinates.is_some_and(|locals| locals.row == *name || locals.column == *name)
        {
            return Err(Error::new(
                name.span(),
                "PCU scalar locals cannot shadow resource, invocation, stride, shape or coordinate bindings",
            ));
        }
        let expected = annotation::expected_type(statement, emitter)?;
        let expression = prepare_expression(emitter, expression, coordinates)?;
        let previous_expected = emitter.expected_scalar;
        if let Some(expected) = expected {
            emitter.expected_scalar = expected;
        }
        let result = emitter.emit_expr(&expression);
        emitter.expected_scalar = previous_expected;
        let (value, scalar) = result?;
        if expected.is_some_and(|expected| expected != scalar) {
            return Err(Error::new_spanned(
                expression,
                "PCU local initializer does not match its declared scalar type; use an admitted explicit conversion",
            ));
        }
        emitter.values.push((name.clone(), value, scalar));
        scope.declare(name, mutable);
    }
    Ok(())
}

fn prepare_expression(
    emitter: &RuntimeExprEmitter<'_>,
    expression: &Expr,
    coordinates: Option<&MatrixLocals>,
) -> Result<Expr, Error> {
    let mut expression = expression.clone();
    let mut attributes = ExpressionAttributes::default();
    attributes.visit_expr_mut(&mut expression);
    if let Some(error) = attributes.error {
        return Err(error);
    }
    if let Some(locals) = coordinates {
        let mut normalizer = MatrixIndexNormalizer {
            invocation: emitter.invocation_ident,
            locals,
            bindings: emitter.bindings,
            error: None,
        };
        normalizer.visit_expr_mut(&mut expression);
        if let Some(error) = normalizer.error {
            return Err(error);
        }
    }
    if emitter.generic_float.is_some() {
        generic_float::validate_expression(&expression, emitter.crate_path)?;
    } else if emitter.generic_integer.is_some() {
        generic_integer::validate_expression(&expression)?;
    } else if emitter.generic_transport.is_some() {
        generic_transport::validate_expression(&expression)?;
    }
    Ok(expression)
}

/// A resource store preserves source order; it does not replace an existing
/// scalar local's SSA value. Later explicit resource loads remain distinct.
pub fn lower_store(
    emitter: &mut RuntimeExprEmitter<'_>,
    assignment: &ExprAssign,
) -> Result<(), Error> {
    let binding =
        validate_assignment_target(assignment, emitter.bindings, emitter.invocation_ident)?;
    if emitter.generic_float.is_some() {
        generic_float::validate_expression(&assignment.right, emitter.crate_path)?;
    } else if emitter.generic_integer.is_some() {
        generic_integer::validate_expression(&assignment.right)?;
    } else if emitter.generic_transport.is_some() {
        generic_transport::validate_expression(&assignment.right)?;
    }
    let binding_slot = binding.binding;
    let scalar = binding.scalar;
    let previous_expected = emitter.expected_scalar;
    emitter.expected_scalar = scalar;
    let result = emitter.emit_expr(&assignment.right);
    emitter.expected_scalar = previous_expected;
    let (value, actual_scalar) = result?;
    if actual_scalar != scalar {
        return Err(Error::new_spanned(
            &assignment.right,
            "PCU store value type must match the output binding element type",
        ));
    }
    let pcu = emitter.crate_path;
    let index = if emitter.grid_stride {
        quote! { GridStrideId }
    } else {
        quote! { InvocationId }
    };
    emitter.statements.push(quote! {
        __pcu_context.store_value(
            #pcu::PcuBindingRef::new(0, #binding_slot),
            #pcu::PcuDispatchIndex::#index,
            #value,
        )?;
    });
    Ok(())
}

pub fn reject_separate_profile(statements: &[Stmt]) -> Result<(), Error> {
    if let Some(statement) = statements.first() {
        return Err(Error::new_spanned(
            statement,
            "PCU scalar locals require a concrete checked scalar or `PcuCheckedFloat`/`PcuCheckedInteger` profile; raw transport and wrapping profiles remain separately bounded",
        ));
    }
    Ok(())
}

#[cfg(test)]
#[path = "tests/tests.rs"]
mod tests;
