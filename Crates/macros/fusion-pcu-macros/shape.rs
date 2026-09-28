//! Bounded compile-time helpers for canonical fixed rank-two array maps.

use proc_macro2::TokenStream;
use quote::ToTokens;
#[rustfmt::skip]
use syn::{
    BinOp,
    Expr,
    ExprIndex,
    Ident,
    Type,
};

#[derive(Clone)]
pub struct FixedMatrixShape {
    pub rows: Expr,
    pub columns: Expr,
}

pub fn nested_array_type(ty: &Type) -> Option<(&Type, FixedMatrixShape)> {
    let Type::Array(outer) = ty else {
        return None;
    };
    let Type::Array(inner) = outer.elem.as_ref() else {
        return None;
    };
    Some((
        inner.elem.as_ref(),
        FixedMatrixShape {
            rows: outer.len.clone(),
            columns: inner.len.clone(),
        },
    ))
}

pub fn same_dimension(left: &Expr, right: &Expr) -> bool {
    normalized(left).to_string() == normalized(right).to_string()
}

pub fn is_matrix_element_count(expression: &Expr, rows: &Expr, columns: &Expr) -> bool {
    let Expr::Binary(binary) = strip_parens(expression) else {
        return false;
    };
    matches!(binary.op, BinOp::Mul(_))
        && ((same_dimension(&binary.left, rows) && same_dimension(&binary.right, columns))
            || (same_dimension(&binary.left, columns) && same_dimension(&binary.right, rows)))
}

fn normalized(expression: &Expr) -> TokenStream {
    match expression {
        Expr::Paren(paren) => normalized(&paren.expr),
        Expr::Group(group) => normalized(&group.expr),
        expression => expression.to_token_stream(),
    }
}

/// Recognizes only `matrix[invocation / C][invocation % C]`.
pub fn is_canonical_matrix_index(index: &ExprIndex, invocation: &Ident, columns: &Expr) -> bool {
    let Expr::Index(row) = index.expr.as_ref() else {
        return false;
    };
    is_canonical_matrix_components(&row.index, &index.index, invocation, columns)
}

pub fn is_canonical_matrix_components(
    row: &Expr,
    column: &Expr,
    invocation: &Ident,
    columns: &Expr,
) -> bool {
    is_dimension_op(row, invocation, columns, false)
        && is_dimension_op(column, invocation, columns, true)
}

/// Recognizes only `matrix[row][column]` for the immutable locals validated by the caller.
pub fn is_matrix_local_index(index: &ExprIndex, row: &Ident, column: &Ident) -> bool {
    let Expr::Index(row_index) = index.expr.as_ref() else {
        return false;
    };
    is_identifier(&row_index.index, row) && is_identifier(&index.index, column)
}

/// Recognizes local declarations `row = invocation / C; column = invocation % C`.
pub fn matrix_local_extent(row: &Expr, column: &Expr, invocation: &Ident) -> Option<Expr> {
    let row_binary = dimension_binary(row, invocation, false)?;
    let column_binary = dimension_binary(column, invocation, true)?;
    if !same_dimension(&row_binary.right, &column_binary.right) {
        return None;
    }
    Some(*row_binary.right.clone())
}

pub fn matrix_base(index: &ExprIndex) -> Option<&Expr> {
    let Expr::Index(row) = index.expr.as_ref() else {
        return None;
    };
    Some(&row.expr)
}

fn is_dimension_op(expression: &Expr, invocation: &Ident, columns: &Expr, remainder: bool) -> bool {
    let Some(binary) = dimension_binary(expression, invocation, remainder) else {
        return false;
    };
    same_dimension(&binary.right, columns)
}

fn dimension_binary<'a>(
    expression: &'a Expr,
    invocation: &Ident,
    remainder: bool,
) -> Option<&'a syn::ExprBinary> {
    let Expr::Binary(binary) = strip_parens(expression) else {
        return None;
    };
    let correct_operator = if remainder {
        matches!(binary.op, BinOp::Rem(_))
    } else {
        matches!(binary.op, BinOp::Div(_))
    };
    (correct_operator && is_identifier(&binary.left, invocation)).then_some(binary)
}

fn strip_parens(mut expression: &Expr) -> &Expr {
    loop {
        match expression {
            Expr::Paren(paren) => expression = &paren.expr,
            Expr::Group(group) => expression = &group.expr,
            _ => return expression,
        }
    }
}

fn is_identifier(expression: &Expr, ident: &Ident) -> bool {
    let expression = strip_parens(expression);
    matches!(expression, Expr::Path(path) if path.qself.is_none() && path.path.get_ident() == Some(ident))
}
