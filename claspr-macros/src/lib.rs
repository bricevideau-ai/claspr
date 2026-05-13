//! Proc-macro frontend for [claspr] — write a kernel function once
//! with kernel-style attributes and signature, get a host-side launch
//! wrapper for free.
//!
//! ## Status
//!
//! This is the **stage 3 first sketch**. The macro generates the
//! host-side launch wrapper from the stub's signature; the kernel
//! source still lives in a separate kernel crate that the user
//! maintains and that `claspr-build` compiles. Auto-generation of the
//! kernel sub-crate from the stub function comes in a follow-up.
//!
//! ## Example
//!
//! ```ignore
//! mod kernels {
//!     include!(concat!(env!("OUT_DIR"), "/collatz_kernels.rs"));
//! }
//!
//! // Stub mirrors the kernel signature exactly. Body is discarded by
//! // the proc-macro; leave it empty or copy-paste the real kernel
//! // body for documentation / future single-source extraction.
//! #[claspr::kernel]
//! fn collatz_kernel(
//!     #[spirv(global_invocation_id)] _id: USizeVec3,
//!     #[spirv(cross_workgroup)] data: &mut [u32],
//! ) {}
//!
//! fn main() -> claspr::Result<()> {
//!     let ctx = claspr::Context::new()?;
//!     let kernels = kernels::Kernels::load(&ctx)?;
//!     let mut data: Vec<u32> = (1..=1024).collect();
//!     let buf = ctx.upload(&data)?;
//!     // Generated wrapper:
//!     //   fn collatz_kernel(
//!     //       ctx: &claspr::Context,
//!     //       kernel: &claspr::Kernel,
//!     //       grid: impl claspr::IntoLaunchSpec,
//!     //       data: &claspr::DeviceSlice<u32>,
//!     //   ) -> claspr::Result<claspr::Event>
//!     collatz_kernel(&ctx, &kernels.collatz_kernel, [data.len()], &buf)?;
//!     ctx.download(&buf, &mut data)?;
//!     Ok(())
//! }
//! ```
//!
//! [claspr]: https://github.com/bricevideau-ai/claspr

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::{
    Attribute, FnArg, ItemFn, Pat, PatType, Type, TypeReference, TypeSlice, parse_macro_input,
    spanned::Spanned,
};

/// Mark a function as a claspr kernel — generates a host-side launch
/// wrapper.
///
/// See the crate-level docs for the rough shape; behaviour follows
/// these rules when walking the stub's parameters:
///
/// - A parameter with a `#[spirv(<builtin>)]` attribute (anything except
///   `cross_workgroup` or no attribute) is **dropped** — these are
///   SPIR-V built-in inputs filled in by the OpenCL runtime, not
///   host-side arguments.
/// - A parameter `#[spirv(cross_workgroup)] name: &mut [T]` or `&[T]`
///   is **translated** to `name: &::claspr::DeviceSlice<T>`.
/// - Any other parameter is passed through verbatim — used for scalar
///   `T` arguments, which work directly via the
///   `claspr::scalar_arg!`-emitted `KernelArg` impl.
///
/// The macro supports *only* the subset of stub signatures collatz
/// uses today (slices + scalars). Image params, samplers, workgroup
/// memory, and user structs will arrive as we exercise more samples
/// through the macro.
#[proc_macro_attribute]
pub fn kernel(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let func = parse_macro_input!(item as ItemFn);
    match expand_kernel(&func) {
        Ok(ts) => ts.into(),
        Err(e) => e.to_compile_error().into(),
    }
}

fn expand_kernel(func: &ItemFn) -> syn::Result<TokenStream2> {
    let vis = &func.vis;
    let name = &func.sig.ident;

    let mut host_params: Vec<TokenStream2> = Vec::new();
    let mut launch_args: Vec<TokenStream2> = Vec::new();

    for input in &func.sig.inputs {
        let FnArg::Typed(pt) = input else {
            return Err(syn::Error::new(
                input.span(),
                "claspr::kernel does not accept `self` parameters",
            ));
        };
        match classify_param(pt)? {
            ParamRole::Builtin => continue,
            ParamRole::Host {
                name: pname,
                ty: pty,
            } => {
                host_params.push(quote! { #pname: #pty });
                launch_args.push(quote! { #pname });
            }
        }
    }

    // Single-element tuples need a trailing comma.
    let launch_tuple = if launch_args.len() == 1 {
        let only = &launch_args[0];
        quote! { ( #only, ) }
    } else {
        quote! { ( #(#launch_args),* ) }
    };

    Ok(quote! {
        #vis fn #name(
            ctx: &::claspr::Context,
            kernel: &::claspr::Kernel,
            grid: impl ::claspr::IntoLaunchSpec,
            #(#host_params),*
        ) -> ::claspr::Result<::claspr::Event> {
            ctx.launch(kernel, grid, #launch_tuple)
        }
    })
}

enum ParamRole {
    /// Drop: SPIR-V builtin filled in by the runtime.
    Builtin,
    /// Keep: host-side argument with translated type and original name.
    Host {
        name: TokenStream2,
        ty: TokenStream2,
    },
}

fn classify_param(pt: &PatType) -> syn::Result<ParamRole> {
    let attr = find_spirv_attr(&pt.attrs);
    let kind = attr.as_ref().map(spirv_attr_kind);

    // Anything tagged with a SPIR-V builtin name (global_invocation_id,
    // workgroup_id, local_invocation_id, …) is a runtime-filled input,
    // not a host kernel argument. The complete list lives in
    // `spirv-std::spirv` — rather than enumerate every keyword, we
    // treat everything except `cross_workgroup` as a builtin. This
    // matches the rust-gpu convention where `cross_workgroup` is the
    // *only* attribute that names a host-supplied buffer.
    if matches!(kind, Some(SpirvKind::Builtin)) {
        return Ok(ParamRole::Builtin);
    }

    let pname = match &*pt.pat {
        Pat::Ident(pi) => {
            let id = &pi.ident;
            quote! { #id }
        }
        other => {
            return Err(syn::Error::new(
                other.span(),
                "claspr::kernel parameter pattern must be a plain identifier",
            ));
        }
    };

    let ty_translated = if matches!(kind, Some(SpirvKind::CrossWorkgroup)) {
        translate_cross_workgroup_ty(&pt.ty)?
    } else {
        // No spirv attribute (or an unrecognised one): pass type through.
        let ty = &pt.ty;
        quote! { #ty }
    };

    Ok(ParamRole::Host {
        name: pname,
        ty: ty_translated,
    })
}

fn find_spirv_attr(attrs: &[Attribute]) -> Option<Attribute> {
    attrs.iter().find(|a| a.path().is_ident("spirv")).cloned()
}

enum SpirvKind {
    /// `#[spirv(cross_workgroup)]` — host-supplied buffer; translate type.
    CrossWorkgroup,
    /// Anything else (`#[spirv(global_invocation_id)]`, `workgroup`, …)
    /// — drop from host signature.
    Builtin,
}

fn spirv_attr_kind(attr: &Attribute) -> SpirvKind {
    // We want to know whether the inner meta starts with `cross_workgroup`.
    // Use the parse_args-as-token-stream path so we don't have to enumerate
    // every spirv attribute syn might know about.
    let tokens = attr.meta.require_list().ok().map(|l| l.tokens.clone());
    let Some(tokens) = tokens else {
        return SpirvKind::Builtin;
    };
    let first_ident = tokens
        .into_iter()
        .find_map(|tt| match tt {
            proc_macro2::TokenTree::Ident(i) => Some(i.to_string()),
            _ => None,
        })
        .unwrap_or_default();
    if first_ident == "cross_workgroup" {
        SpirvKind::CrossWorkgroup
    } else {
        SpirvKind::Builtin
    }
}

/// Translate a `cross_workgroup` parameter type.
///
/// `&mut [T]` and `&[T]` both become `&::claspr::DeviceSlice<T>`. Any
/// other shape is a hard error — those paths haven't been wired up
/// yet (image, sampler, workgroup-memory parameters will land as
/// follow-ups).
fn translate_cross_workgroup_ty(ty: &Type) -> syn::Result<TokenStream2> {
    let Type::Reference(TypeReference { elem, .. }) = ty else {
        return Err(syn::Error::new(
            ty.span(),
            "expected a reference type (`&[T]` or `&mut [T]`) for a #[spirv(cross_workgroup)] \
             parameter; other shapes are not yet supported by claspr::kernel",
        ));
    };
    let Type::Slice(TypeSlice { elem: inner, .. }) = &**elem else {
        return Err(syn::Error::new(
            elem.span(),
            "expected a slice type `[T]` after the reference; other shapes are not yet supported \
             by claspr::kernel",
        ));
    };
    Ok(quote! { &::claspr::DeviceSlice<#inner> })
}
