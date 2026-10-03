//! Concrete source type vocabulary, separate from provider capability admission.
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ScalarKind {
    F32,
    F64,
    U8,
    U16,
    U32,
    U64,
    U128,
    I8,
    I16,
    I32,
    I64,
    I128,
    Generic,
}

impl ScalarKind {
    pub(super) fn rust_type(self) -> TokenStream2 {
        match self {
            Self::F32 => quote! { f32 },
            Self::F64 => quote! { f64 },
            Self::U8 => quote! { u8 },
            Self::U16 => quote! { u16 },
            Self::U32 => quote! { u32 },
            Self::U64 => quote! { u64 },
            Self::U128 => quote! { u128 },
            Self::I8 => quote! { i8 },
            Self::I16 => quote! { i16 },
            Self::I32 => quote! { i32 },
            Self::I64 => quote! { i64 },
            Self::I128 => quote! { i128 },
            Self::Generic => unreachable!("generic scalar has no concrete Rust type token"),
        }
    }
}

/// Concrete Rust primitive spellings; sealed generic bounds cover PCU wrappers.
pub fn concrete_scalar(path: &syn::Path) -> Option<ScalarKind> {
    [
        ("f32", ScalarKind::F32),
        ("f64", ScalarKind::F64),
        ("u8", ScalarKind::U8),
        ("u16", ScalarKind::U16),
        ("u32", ScalarKind::U32),
        ("u64", ScalarKind::U64),
        ("u128", ScalarKind::U128),
        ("i8", ScalarKind::I8),
        ("i16", ScalarKind::I16),
        ("i32", ScalarKind::I32),
        ("i64", ScalarKind::I64),
        ("i128", ScalarKind::I128),
    ]
    .into_iter()
    .find_map(|(name, kind)| path.is_ident(name).then_some(kind))
}

#[cfg(test)]
#[path = "tests/tests.rs"]
mod tests;
