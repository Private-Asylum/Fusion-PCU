//! Reserved invocation-level intrinsics; no Rust helper lookup on the warm path.
#[rustfmt::skip]
use syn::{
    Error,
    Expr,
    ExprCall,
    Path,
    PathArguments,
};

/// Recognizes `pcu::relu` or the selected crate's fully qualified `pcu::relu`.
/// The shared spelling lowers scalar invocation values; tensor capture has its
/// own value/ownership lowering. Neither form calls an arbitrary unmarked helper.
pub fn relu_operand<'a>(call: &'a ExprCall, crate_path: &Path) -> Result<Option<&'a Expr>, Error> {
    let Expr::Path(function) = call.func.as_ref() else {
        return Ok(None);
    };
    if function.qself.is_some() {
        return Ok(None);
    }
    let segments = &function.path.segments;
    let matches_suffix = |start: usize| {
        segments.len() == start + 2
            && segments[start].ident == "pcu"
            && segments[start + 1].ident == "relu"
    };
    let relative = matches_suffix(0);
    let qualified = matches_suffix(crate_path.segments.len())
        && segments
            .iter()
            .zip(&crate_path.segments)
            .all(|(actual, expected)| actual.ident == expected.ident);
    if !relative && !qualified {
        return Ok(None);
    }
    if call.args.len() != 1
        || segments
            .iter()
            .any(|segment| !matches!(segment.arguments, PathArguments::None))
    {
        return Err(Error::new_spanned(
            call,
            "PCU scalar `pcu::relu` requires exactly one value and no explicit type arguments",
        ));
    }
    Ok(call.args.first())
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
