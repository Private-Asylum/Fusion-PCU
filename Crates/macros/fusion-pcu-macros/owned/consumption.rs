//! Lexical ownership of an external owner in the bounded captured graph language.
//!
//! Graph value IDs in the neutral IR are copyable, but the source owner token is linear. Both
//! external inputs and operation-produced values move once when used bare; references borrow the
//! owner token so graph fanout is explicit. Neither rule grants destructive backing reuse; the
//! core/backend still prove that separately.

#[rustfmt::skip]
use syn::{
    Error,
    Expr,
    ItemFn,
};

#[rustfmt::skip]
use super::{
    expression_path_ident,
    is_pcu_builtin,
    local_binding,
    unwrap_result_return,
    unwrap_transparent,
};

pub(super) fn validate_body(function: &ItemFn, inputs: &[syn::Ident]) -> Result<(), Error> {
    let Some((terminal, statements)) = function.block.stmts.split_last() else {
        return Err(super::composition_body_error(&function.block));
    };
    let parameter_names = function
        .sig
        .inputs
        .iter()
        .filter_map(|argument| match argument {
            syn::FnArg::Typed(argument) => match argument.pat.as_ref() {
                syn::Pat::Ident(pattern) => Some(pattern.ident.clone()),
                _ => None,
            },
            syn::FnArg::Receiver(_) => None,
        })
        .collect::<Vec<_>>();
    let mut state = Ownership::new(&parameter_names, inputs);
    for statement in statements {
        let syn::Stmt::Local(local) = statement else {
            return Err(super::composition_body_error(statement));
        };
        let (name, expression) = local_binding(local)?;
        if let Some(owner) = state.owner_path(expression)? {
            state.visit(expression, false, 0)?;
            let new_owner = state.add_owner();
            state.bind(name.clone(), BindingValue::Owner(new_owner));
            debug_assert!(state.owner_was_moved(owner));
        } else if let Some(owner) = state.borrowed_owner_path(expression)? {
            state.visit(expression, true, 0)?;
            state.bind(name.clone(), BindingValue::View(owner));
        } else if let Some(owner) = state.view_alias_path(expression)? {
            state.visit(expression, true, 0)?;
            state.bind(name.clone(), BindingValue::View(owner));
        } else if state.graph_value_alias(expression)? {
            state.visit(expression, true, 0)?;
            state.bind(name.clone(), BindingValue::GraphValue);
        } else if is_operation(expression)? {
            state.visit(expression, false, 0)?;
            let owner = state.add_owner();
            state.bind(name.clone(), BindingValue::Owner(owner));
        } else {
            return Err(Error::new_spanned(
                expression,
                "consuming tensor locals must describe operations or immutable owner aliases",
            ));
        }
    }
    let syn::Stmt::Expr(expression, None) = terminal else {
        return Err(super::composition_body_error(terminal));
    };
    let result = unwrap_result_return(unwrap_transparent(expression)?)?;
    if matches!(unwrap_transparent(result)?, Expr::Reference(_)) {
        return Err(Error::new_spanned(
            result,
            "an owned tensor result cannot return a reference",
        ));
    }
    if state.view_alias_path(result)?.is_some() {
        return Err(Error::new_spanned(
            result,
            "an owned tensor result cannot return a borrowed owner alias",
        ));
    }
    state.visit(result, false, 0)
}

fn is_operation(expression: &Expr) -> Result<bool, Error> {
    match unwrap_transparent(expression)? {
        Expr::Try(expression) => Ok(matches!(
            unwrap_transparent(&expression.expr)?,
            Expr::Call(_) | Expr::Binary(_)
        )),
        Expr::Call(_) | Expr::Binary(_) => Ok(true),
        _ => Ok(false),
    }
}

struct Ownership {
    bindings: Vec<(syn::Ident, BindingValue)>,
    owners_moved: Vec<bool>,
}

#[derive(Clone, Copy)]
enum BindingValue {
    Owner(usize),
    View(usize),
    GraphValue,
}

impl Ownership {
    fn new(parameters: &[syn::Ident], owners: &[syn::Ident]) -> Self {
        let mut state = Self {
            bindings: Vec::with_capacity(parameters.len()),
            owners_moved: vec![false; owners.len()],
        };
        for parameter in parameters {
            let value = owners
                .iter()
                .position(|owner| owner == parameter)
                .map_or(BindingValue::GraphValue, BindingValue::Owner);
            state.bind(parameter.clone(), value);
        }
        state
    }

    fn bind(&mut self, name: syn::Ident, value: BindingValue) {
        self.bindings.push((name, value));
    }

    fn resolve(&self, name: &syn::Ident) -> Option<BindingValue> {
        self.bindings
            .iter()
            .rev()
            .find(|(binding, _)| binding == name)
            .map(|(_, value)| *value)
    }

    fn add_owner(&mut self) -> usize {
        let id = self.owners_moved.len();
        self.owners_moved.push(false);
        id
    }

    fn owner_path(&self, expression: &Expr) -> Result<Option<usize>, Error> {
        let Expr::Path(path) = unwrap_transparent(expression)? else {
            return Ok(None);
        };
        let Some(name) = expression_path_ident(path)? else {
            return Ok(None);
        };
        Ok(match self.resolve(name) {
            Some(BindingValue::Owner(owner)) => Some(owner),
            _ => None,
        })
    }

    fn borrowed_owner_path(&self, expression: &Expr) -> Result<Option<usize>, Error> {
        let Expr::Reference(reference) = unwrap_transparent(expression)? else {
            return Ok(None);
        };
        if !reference.attrs.is_empty() || reference.mutability.is_some() {
            return Ok(None);
        }
        let Expr::Path(path) = unwrap_transparent(&reference.expr)? else {
            return Ok(None);
        };
        let Some(name) = expression_path_ident(path)? else {
            return Ok(None);
        };
        Ok(match self.resolve(name) {
            Some(BindingValue::Owner(owner) | BindingValue::View(owner)) => Some(owner),
            _ => None,
        })
    }

    fn view_alias_path(&self, expression: &Expr) -> Result<Option<usize>, Error> {
        let Expr::Path(path) = unwrap_transparent(expression)? else {
            return Ok(None);
        };
        let Some(name) = expression_path_ident(path)? else {
            return Ok(None);
        };
        Ok(match self.resolve(name) {
            Some(BindingValue::View(owner)) => Some(owner),
            _ => None,
        })
    }

    fn graph_value_alias(&self, expression: &Expr) -> Result<bool, Error> {
        let source = match unwrap_transparent(expression)? {
            Expr::Path(path) => Some(path),
            Expr::Reference(reference)
                if reference.attrs.is_empty() && reference.mutability.is_none() =>
            {
                match unwrap_transparent(&reference.expr)? {
                    Expr::Path(path) => Some(path),
                    _ => None,
                }
            }
            _ => None,
        };
        let Some(path) = source else {
            return Ok(false);
        };
        let Some(name) = expression_path_ident(path)? else {
            return Ok(false);
        };
        Ok(matches!(self.resolve(name), Some(BindingValue::GraphValue)))
    }
}

impl Ownership {
    fn visit(&mut self, expression: &Expr, borrowed: bool, depth: usize) -> Result<(), Error> {
        if depth > 64 {
            return Err(super::composition_body_error(expression));
        }
        match unwrap_transparent(expression)? {
            Expr::Path(path) => {
                let Some(identifier) = expression_path_ident(path)? else {
                    return Ok(());
                };
                match self.resolve(identifier) {
                    Some(BindingValue::Owner(owner)) => {
                        if self.owner_was_moved(owner) {
                            return Err(Error::new_spanned(
                                path,
                                "use of consumed PCU owner after move",
                            ));
                        }
                        if !borrowed {
                            self.owners_moved[owner] = true;
                        }
                    }
                    Some(BindingValue::View(owner)) if self.owner_was_moved(owner) => {
                        return Err(Error::new_spanned(
                            path,
                            "use of borrowed PCU owner view after its owner was moved",
                        ));
                    }
                    _ => {}
                }
                Ok(())
            }
            Expr::Reference(reference)
                if reference.attrs.is_empty() && reference.mutability.is_none() =>
            {
                self.visit(&reference.expr, true, depth + 1)
            }
            Expr::Try(expression) => self.visit(&expression.expr, borrowed, depth + 1),
            Expr::Binary(binary) => {
                // Borrowing the expression's result does not borrow its operands.
                self.visit(&binary.left, false, depth + 1)?;
                self.visit(&binary.right, false, depth + 1)
            }
            Expr::Call(call) => self.visit_call(call, depth),
            _ => Err(super::composition_body_error(expression)),
        }
    }

    fn visit_call(&mut self, call: &syn::ExprCall, depth: usize) -> Result<(), Error> {
        let Expr::Path(path) = unwrap_transparent(&call.func)? else {
            return Err(super::composition_body_error(&call.func));
        };
        if path
            .path
            .segments
            .first()
            .is_some_and(|segment| segment.ident == "pcu")
            && !["identity", "relu", "add", "sub", "mul", "matmul"]
                .iter()
                .any(|name| is_pcu_builtin(&path.path, name))
        {
            return Err(super::composition_body_error(&call.func));
        }
        for argument in &call.args {
            self.visit(argument, false, depth + 1)?;
        }
        Ok(())
    }
}

impl Ownership {
    fn owner_was_moved(&self, owner: usize) -> bool {
        self.owners_moved.get(owner).copied().unwrap_or(true)
    }
}

#[cfg(test)]
mod tests {
    use super::validate_body;

    #[test]
    fn operation_temporaries_fan_out_but_external_owner_moves_only_once() {
        for body in [
            "let a = pcu::relu(input)?; let b = pcu::add(&a, &a)?; Ok(b)",
            "let a = pcu::identity(&input)?; let b = pcu::relu(input)?; Ok(pcu::add(a, b)?)",
            "Ok(pcu::relu(pcu::identity(input)?)?)",
            "Ok(input)",
            "let a = helper(&input)?; let b = pcu::relu(input)?; Ok(pcu::add(a, b)?)",
            "let a = helper(input)?; Ok(a)",
            "let owner = input; let alias = owner; let view = &alias; let a = helper(view)?; Ok(pcu::relu(alias)?)",
            "let view = &input; let a = helper(view)?; Ok(pcu::relu(input)?)",
        ] {
            let function: syn::ItemFn = syn::parse_str(&format!(
                "fn test(input: PcuTensor<f32>) -> Result<PcuTensor<f32>, Error> {{ {body} }}"
            ))
            .unwrap();
            validate_body(&function, &[syn::parse_quote!(input)]).unwrap();
        }
    }

    #[test]
    fn erased_graph_ids_cannot_hide_external_use_after_move_or_aliases() {
        for body in [
            "Ok(pcu::add(input, input)?)",
            "let a = pcu::relu(input)?; Ok(pcu::add(a, &input)?)",
            "let a = pcu::relu(input)?; Ok(input)",
            "let a = helper(input)?; Ok(pcu::relu(input)?)",
            "let owner = input; let alias = owner; let moved = pcu::relu(alias)?; Ok(pcu::relu(owner)?)",
            "let view = &input; let moved = pcu::relu(input)?; Ok(pcu::add(view, moved)?)",
            "let view = &input; let moved = pcu::relu(input)?; Ok(view)",
            "Ok(&input)",
            "Ok(pcu::add(&(pcu::relu(input)?), &input)?)",
        ] {
            let function: syn::ItemFn = syn::parse_str(&format!(
                "fn test(input: PcuTensor<f32>) -> Result<PcuTensor<f32>, Error> {{ {body} }}"
            ))
            .unwrap();
            assert!(
                validate_body(&function, &[syn::parse_quote!(input)]).is_err(),
                "accepted {body}"
            );
        }
    }

    #[test]
    fn immutable_aliases_move_once_and_borrowed_views_cannot_escape() {
        for body in [
            "let owner = input; let alias = owner; Ok(pcu::relu(alias)?)",
            "let view = &input; let observed = helper(view)?; let moved = pcu::relu(input)?; Ok(pcu::add(observed, moved)?)",
        ] {
            let function: syn::ItemFn = syn::parse_str(&format!(
                "fn test(input: PcuTensor<f32>) -> Result<PcuTensor<f32>, Error> {{ {body} }}"
            ))
            .unwrap();
            validate_body(&function, &[syn::parse_quote!(input)]).unwrap();
        }
        for body in [
            "let owner = input; let alias = owner; let moved = pcu::relu(alias)?; Ok(pcu::relu(owner)?)",
            "let view = &input; let moved = pcu::relu(input)?; Ok(pcu::add(view, moved)?)",
            "let view = &input; let moved = pcu::relu(input)?; Ok(view)",
        ] {
            let function: syn::ItemFn = syn::parse_str(&format!(
                "fn test(input: PcuTensor<f32>) -> Result<PcuTensor<f32>, Error> {{ {body} }}"
            ))
            .unwrap();
            assert!(validate_body(&function, &[syn::parse_quote!(input)]).is_err());
        }
    }

    #[test]
    fn multiple_external_owners_have_independent_move_states() {
        let valid: syn::ItemFn = syn::parse_quote! {
            fn test(left: PcuTensor<f32>, right: PcuTensor<f32>)
                -> Result<PcuTensor<f32>, Error>
            {
                let moved = pcu::relu(left)?;
                Ok(pcu::add(moved, right)?)
            }
        };
        let inputs = [syn::parse_quote!(left), syn::parse_quote!(right)];
        validate_body(&valid, &inputs).unwrap();

        for body in [
            "Ok(pcu::add(left, left)?)",
            "let owner = left; let moved = pcu::relu(owner)?; Ok(pcu::add(moved, left)?)",
        ] {
            let invalid: syn::ItemFn = syn::parse_str(&format!(
                "fn test(left: PcuTensor<f32>, right: PcuTensor<f32>) -> Result<PcuTensor<f32>, Error> {{ {body} }}"
            ))
            .unwrap();
            assert!(validate_body(&invalid, &inputs).is_err());
        }
    }

    #[test]
    fn shadowed_bindings_resolve_rhs_first_and_keep_stable_owner_roots() {
        let valid: syn::ItemFn = syn::parse_quote! {
            fn test(input: PcuTensor<f32>, right: PcuTensor<f32>)
                -> Result<PcuTensor<f32>, Error>
            {
                let view = &input;
                let input = pcu::relu(right)?;
                let observed = helper(view)?;
                Ok(pcu::add(observed, input)?)
            }
        };
        validate_body(
            &valid,
            &[syn::parse_quote!(input), syn::parse_quote!(right)],
        )
        .unwrap();

        let alias_move: syn::ItemFn = syn::parse_quote! {
            fn test(input: PcuTensor<f32>, right: PcuTensor<f32>)
                -> Result<PcuTensor<f32>, Error>
            {
                let view = &input;
                let alias = input;
                let input = pcu::relu(right)?;
                let observed = helper(view)?;
                let moved = pcu::relu(alias)?;
                Ok(pcu::add(observed, pcu::add(input, moved)?)?)
            }
        };
        validate_body(
            &alias_move,
            &[syn::parse_quote!(input), syn::parse_quote!(right)],
        )
        .unwrap_err();
    }

    #[test]
    fn shadowing_does_not_hide_old_owner_moves_from_views() {
        for body in [
            "let view = &input; let alias = input; let input = pcu::relu(right)?; let moved = pcu::relu(alias)?; Ok(pcu::add(view, moved)?)",
            "let alias = input; let moved = pcu::relu(alias)?; let input = pcu::relu(right)?; let again = pcu::relu(alias)?; Ok(pcu::add(input, again)?)",
        ] {
            let function: syn::ItemFn = syn::parse_str(&format!(
                "fn test(input: PcuTensor<f32>, right: PcuTensor<f32>) -> Result<PcuTensor<f32>, Error> {{ {body} }}"
            ))
            .unwrap();
            validate_body(
                &function,
                &[syn::parse_quote!(input), syn::parse_quote!(right)],
            )
            .unwrap_err();
        }
    }

    #[test]
    fn borrowed_graph_values_can_be_aliased_and_shadowed() {
        let function: syn::ItemFn = syn::parse_quote! {
            fn test(input: PcuTensor<f32>, data: &[f32])
                -> Result<PcuTensor<f32>, Error>
            {
                let data = data;
                let data = pcu::identity(data)?;
                let moved = pcu::relu(input)?;
                Ok(pcu::add(data, moved)?)
            }
        };
        validate_body(&function, &[syn::parse_quote!(input)]).unwrap();
    }
}
