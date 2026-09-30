//! Compile-time primitive parameters, separated from tensor resources and runtime bindings.

#[rustfmt::skip]
use syn::{
    Error,
    Expr,
    Lit,
    UnOp,
};

/// A literal is immutable across calls. A runtime expression must become a real resource
/// binding before it can participate in a cached graph; evaluating it during capture is unsound.
pub(super) fn finite_f32_literal(expression: &Expr) -> Result<Expr, Error> {
    let candidate = super::unwrap_transparent(expression)?;
    let literal = match candidate {
        Expr::Unary(unary) if unary.attrs.is_empty() && matches!(unary.op, UnOp::Neg(_)) => {
            super::unwrap_transparent(&unary.expr)?
        }
        other => other,
    };
    if let Expr::Lit(literal) = literal
        && literal.attrs.is_empty()
        && let Lit::Float(number) = &literal.lit
        && matches!(number.suffix(), "" | "f32")
        && number.base10_parse::<f32>().is_ok_and(f32::is_finite)
    {
        return Ok(expression.clone());
    }
    Err(Error::new_spanned(
        expression,
        "PCU SGD currently requires a finite F32 literal rate; dynamic rates require a resource binding and cannot be frozen into the prepared cache",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rates_admit_finite_f32_literals_and_preserve_signed_zero_syntax() {
        for source in ["0.25", "-0.25_f32", "(-0.0_f32)", "1e-45", "3.4e38_f32"] {
            let expression: Expr = syn::parse_str(source).unwrap();
            assert!(finite_f32_literal(&expression).is_ok(), "{source}");
        }
    }

    #[test]
    fn rates_reject_nonfinite_other_types_and_runtime_or_effectful_expressions() {
        for source in [
            "1e100_f32",
            "-1e100",
            "0.25_f64",
            "1",
            "RATE",
            "rate",
            "get_rate()",
            "0.25 + 0.25",
            "f32::NAN",
            "1.0 / 0.0",
            "--0.25",
        ] {
            let expression: Expr = syn::parse_str(source).unwrap();
            assert!(finite_f32_literal(&expression).is_err(), "{source}");
        }
    }
}
