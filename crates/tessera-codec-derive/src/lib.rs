// SPDX-License-Identifier: Apache-2.0 OR MIT

//! `#[derive(Encode, Decode)]` for the Tessera Canonical Encoding (RFC-0003 'Composite types').
//!
//! - A struct encodes its fields in declaration order, with no padding and no field tags.
//! - An enum encodes a `u16` tag equal to the variant's declaration index (from 0), then the
//!   variant's fields as a struct. A decoder rejects any other tag.
//!
//! Use the macros through `tessera_codec`, which re-exports them next to the traits they
//! implement.

#![forbid(unsafe_code)]

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote};
use syn::spanned::Spanned;
use syn::{Data, DeriveInput, Fields, Generics, Ident, parse_macro_input, parse_quote};

/// Implements `tessera_codec::Encode`.
#[proc_macro_derive(Encode)]
pub fn derive_encode(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    expand_encode(&input)
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}

/// Implements `tessera_codec::Decode`.
#[proc_macro_derive(Decode)]
pub fn derive_decode(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    expand_decode(&input)
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}

/// Fails on `#[cfg]` or `#[cfg_attr]`: a field or variant that exists only in some builds would
/// change the wire format between them. Configuration-dependent schemas are separate types.
fn reject_cfg(attrs: &[syn::Attribute]) -> syn::Result<()> {
    for attr in attrs {
        if attr.path().is_ident("cfg") || attr.path().is_ident("cfg_attr") {
            return Err(syn::Error::new(
                attr.span(),
                "a TCE field or variant cannot depend on configuration: the encoding would \
                 differ between builds",
            ));
        }
    }
    Ok(())
}

/// Checks every field and variant of `data` with [`reject_cfg`].
fn check_layout(data: &Data) -> syn::Result<()> {
    match data {
        Data::Struct(data) => data.fields.iter().try_for_each(|f| reject_cfg(&f.attrs)),
        Data::Enum(data) => data.variants.iter().try_for_each(|v| {
            reject_cfg(&v.attrs)?;
            v.fields.iter().try_for_each(|f| reject_cfg(&f.attrs))
        }),
        Data::Union(_) => Ok(()),
    }
}

/// The generics of `input` with `bound` added to every type parameter.
fn bounded(generics: &Generics, bound: &syn::Path) -> Generics {
    let mut generics = generics.clone();
    for param in generics.type_params_mut() {
        param.bounds.push(parse_quote!(#bound));
    }
    generics
}

/// The variants of an enum, after checking that TCE can tag them.
fn variants(data: &syn::DataEnum) -> syn::Result<Vec<&syn::Variant>> {
    if data.variants.len() > usize::from(u16::MAX) + 1 {
        return Err(syn::Error::new(
            data.enum_token.span(),
            "a TCE enum has at most 65,536 variants: its tag is a u16",
        ));
    }
    for variant in &data.variants {
        if let Some((eq, _)) = &variant.discriminant {
            return Err(syn::Error::new(
                eq.span(),
                "a TCE tag is the variant's declaration index; an explicit discriminant would \
                 suggest otherwise",
            ));
        }
    }
    Ok(data.variants.iter().collect())
}

/// A pattern that binds every field of `fields` under `path`, and the bindings in order.
fn destructure(path: &TokenStream2, fields: &Fields) -> (TokenStream2, Vec<Ident>) {
    let bindings: Vec<Ident> = (0..fields.len()).map(|i| format_ident!("__f{i}")).collect();
    let pattern = match fields {
        Fields::Named(named) => {
            let names = named.named.iter().map(|f| &f.ident);
            quote!(#path { #(#names: #bindings),* })
        }
        Fields::Unnamed(_) => quote!(#path(#(#bindings),*)),
        Fields::Unit => quote!(#path),
    };
    (pattern, bindings)
}

/// An expression building `path` from fields decoded in declaration order.
fn construct(path: &TokenStream2, fields: &Fields) -> TokenStream2 {
    let decode = quote!(::tessera_codec::Decode::decode_from(__reader)?);
    match fields {
        Fields::Named(named) => {
            let names = named.named.iter().map(|f| &f.ident);
            quote!(#path { #(#names: #decode),* })
        }
        Fields::Unnamed(unnamed) => {
            let values = unnamed.unnamed.iter().map(|_| &decode);
            quote!(#path(#(#values),*))
        }
        Fields::Unit => quote!(#path),
    }
}

/// The sum of the fields' minimum sizes.
fn min_size(fields: &Fields) -> TokenStream2 {
    let types = fields.iter().map(|f| &f.ty);
    quote!(0usize #(+ <#types as ::tessera_codec::Decode>::MIN_SIZE)*)
}

fn expand_encode(input: &DeriveInput) -> syn::Result<TokenStream2> {
    check_layout(&input.data)?;
    let name = &input.ident;
    let generics = bounded(&input.generics, &parse_quote!(::tessera_codec::Encode));
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();
    let body = match &input.data {
        Data::Struct(data) => {
            let (pattern, fields) = destructure(&quote!(Self), &data.fields);
            quote! {
                let #pattern = self;
                #(::tessera_codec::Encode::encode_to(#fields, __out);)*
            }
        }
        Data::Enum(data) if data.variants.is_empty() => quote!(match *self {}),
        Data::Enum(data) => {
            let arms = variants(data)?
                .into_iter()
                .enumerate()
                .map(|(index, variant)| {
                    let tag = u16::try_from(index).unwrap_or(u16::MAX);
                    let ident = &variant.ident;
                    let (pattern, fields) = destructure(&quote!(Self::#ident), &variant.fields);
                    quote! {
                        #pattern => {
                            ::tessera_codec::Encode::encode_to(&#tag, __out);
                            #(::tessera_codec::Encode::encode_to(#fields, __out);)*
                        }
                    }
                });
            quote!(match self { #(#arms)* })
        }
        Data::Union(data) => {
            return Err(syn::Error::new(
                data.union_token.span(),
                "TCE has no unions",
            ));
        }
    };
    Ok(quote! {
        impl #impl_generics ::tessera_codec::Encode for #name #ty_generics #where_clause {
            fn encode_to<__TesseraSink: ::tessera_codec::Sink + ?Sized>(
                &self,
                __out: &mut __TesseraSink,
            ) {
                #body
            }
        }
    })
}

fn expand_decode(input: &DeriveInput) -> syn::Result<TokenStream2> {
    check_layout(&input.data)?;
    let name = &input.ident;
    let generics = bounded(&input.generics, &parse_quote!(::tessera_codec::Decode));
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();
    let (min_size, value) = match &input.data {
        Data::Struct(data) => (
            min_size(&data.fields),
            construct(&quote!(Self), &data.fields),
        ),
        Data::Enum(data) => {
            let variants = variants(data)?;
            let sizes = variants.iter().map(|v| min_size(&v.fields));
            let arms = variants.iter().enumerate().map(|(index, variant)| {
                let tag = u16::try_from(index).unwrap_or(u16::MAX);
                let ident = &variant.ident;
                let value = construct(&quote!(Self::#ident), &variant.fields);
                quote!(#tag => #value,)
            });
            // The tag, then the smallest variant; an enum without variants has only its tag.
            let min_size = quote! {
                2usize + {
                    let mut __min = ::core::primitive::usize::MAX;
                    #({
                        let __size = #sizes;
                        if __size < __min {
                            __min = __size;
                        }
                    })*
                    if __min == ::core::primitive::usize::MAX { 0 } else { __min }
                }
            };
            let value = quote! {{
                let __at = __reader.position();
                let __tag: ::core::primitive::u16 = ::tessera_codec::Decode::decode_from(__reader)?;
                match __tag {
                    #(#arms)*
                    _ => return ::core::result::Result::Err(__reader.noncanonical(__at)),
                }
            }};
            (min_size, value)
        }
        Data::Union(data) => {
            return Err(syn::Error::new(
                data.union_token.span(),
                "TCE has no unions",
            ));
        }
    };
    Ok(quote! {
        impl #impl_generics ::tessera_codec::Decode for #name #ty_generics #where_clause {
            const MIN_SIZE: ::core::primitive::usize = #min_size;

            // An enum without variants decodes no value: everything after its tag is unreachable.
            #[allow(unreachable_code)]
            fn decode_from(
                __reader: &mut ::tessera_codec::Reader<'_>,
            ) -> ::core::result::Result<Self, ::tessera_codec::DecodeError> {
                __reader.enter()?;
                let __value = #value;
                __reader.leave();
                ::core::result::Result::Ok(__value)
            }
        }
    })
}
