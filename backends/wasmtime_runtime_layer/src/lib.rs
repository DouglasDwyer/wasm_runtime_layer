#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::missing_docs_in_private_items)]
#![cfg_attr(not(feature = "std"), no_std)]

//! `wasmtime_runtime_layer` implements the `wasm_runtime_layer` abstraction interface over WebAssembly runtimes for `Wasmtime`.
//!
//! ## Optional features
//!
//! **cranelift** - Enables executing WASM modules and components with the Cranelift compiler, as described in the Wasmtime documentation. Enabled by default.
//!
//! **winch** - Enables executing WASM modules and components with the Winch compiler, as described in the Wasmtime documentation.

extern crate alloc;

use alloc::{
    boxed::Box,
    string::{String, ToString},
    sync::Arc,
    vec::Vec,
};
use core::ops::{Deref, DerefMut};

use anyhow::{bail, Error, Result};
use fxhash::FxHashMap;
use ref_cast::RefCast;
use smallvec::SmallVec;
use wasm_runtime_layer::{
    backend::{
        AsContext, AsContextMut, Export, Extern, Imports, Ref, Val, WasmEngine, WasmExternRef,
        WasmFunc, WasmGlobal, WasmInstance, WasmMemory, WasmModule, WasmStore, WasmStoreContext,
        WasmStoreContextMut, WasmTable,
    },
    ExportType, ExternType, FuncType, GlobalType, ImportType, MemoryType, RefType, TableType,
    ValType,
};

/// The default amount of arguments and return values for which to allocate
/// stack space.
const DEFAULT_ARGUMENT_SIZE: usize = 4;

/// A vector which allocates up to the default number of arguments on the stack.
type ArgumentVec<T> = SmallVec<[T; DEFAULT_ARGUMENT_SIZE]>;

/// Generate the boilerplate delegation code for a newtype wrapper.
macro_rules! delegate {
    (#[derive($($derive:ident),*)] $newtype:ident [$($gens:tt)*] ($inner:ty) $($impl:tt)*) => {
        #[derive($($derive,)* RefCast)]
        #[repr(transparent)]
        #[doc = concat!("Newtype wrapper around [`", stringify!($inner), "`].")]
        pub struct $newtype $($impl)* ($inner);

        impl$($impl)* $newtype $($gens)* {
            #[must_use]
            #[doc = concat!(
                "Create a [`wasm_runtime_layer::", stringify!($newtype), "`]-compatible `",
                stringify!($newtype),
                "` from a [`",
                stringify!($inner),
                "`]."
            )]
            pub fn new(inner: $inner) -> Self {
                Self(inner)
            }

            #[must_use]
            #[doc = concat!(
                "Consume a `",
                stringify!($newtype),
                "` to obtain the inner [`",
                stringify!($inner),
                "`]."
            )]
            pub fn into_inner(self) -> $inner {
                self.0
            }
        }

        impl$($impl)* From<$inner> for $newtype $($gens)* {
            fn from(inner: $inner) -> Self {
                Self::new(inner)
            }
        }

        impl$($impl)* From<$newtype $($gens)*> for $inner {
            fn from(wrapper: $newtype $($gens)*) -> Self {
                wrapper.into_inner()
            }
        }

        impl$($impl)* Deref for $newtype $($gens)* {
            type Target = $inner;

            fn deref(&self) -> &Self::Target {
                &self.0
            }
        }

        impl$($impl)* DerefMut for $newtype $($gens)* {
            fn deref_mut(&mut self) -> &mut Self::Target {
                &mut self.0
            }
        }

        impl$($impl)* AsRef<$inner> for $newtype $($gens)* {
            fn as_ref(&self) -> &$inner {
                &self.0
            }
        }

        impl$($impl)* AsMut<$inner> for $newtype $($gens)* {
            fn as_mut(&mut self) -> &mut $inner {
                &mut self.0
            }
        }

        impl$($impl)* AsRef<$newtype $($gens)*> for $inner {
            fn as_ref(&self) -> &$newtype $($gens)* {
                $newtype::ref_cast(self)
            }
        }

        impl$($impl)* AsMut<$newtype $($gens)*> for $inner {
            fn as_mut(&mut self) -> &mut $newtype $($gens)* {
                $newtype::ref_cast_mut(self)
            }
        }
    }
}

delegate! { #[derive(Clone, Default)] Engine[](wasmtime::Engine) }
delegate! { #[derive(Clone)] ExternRef[](wasmtime::Rooted<wasmtime::ExternRef>) }
delegate! { #[derive(Clone)] Func[](wasmtime::Func) }
delegate! { #[derive(Clone)] Global[](wasmtime::Global) }
delegate! { #[derive(Clone)] Memory[](wasmtime::Memory) }
delegate! { #[derive(Clone)] Module[](wasmtime::Module) }
delegate! { #[derive()] Store[<T>](wasmtime::Store<T>) <T: 'static> }
delegate! { #[derive()] StoreContext[<'a, T>](wasmtime::StoreContext<'a, T>) <'a, T: 'static> }
delegate! { #[derive()] StoreContextMut[<'a, T>](wasmtime::StoreContextMut<'a, T>) <'a, T: 'static> }
delegate! { #[derive(Clone)] Table[](wasmtime::Table) }

impl WasmEngine for Engine {
    type ExternRef = ExternRef;
    type Func = Func;
    type Global = Global;
    type Instance = Instance;
    type Memory = Memory;
    type Module = Module;
    type Store<T: 'static> = Store<T>;
    type StoreContext<'a, T: 'static> = StoreContext<'a, T>;
    type StoreContextMut<'a, T: 'static> = StoreContextMut<'a, T>;
    type Table = Table;
}

impl WasmExternRef<Engine> for ExternRef {
    fn new<T: 'static + Send + Sync>(mut ctx: impl AsContextMut<Engine>, object: T) -> Self {
        Self::new(
            wasmtime::ExternRef::new(ctx.as_context_mut().into_inner(), object)
                .expect("out of memory"),
        )
    }

    fn downcast<'a, 's: 'a, T: 'static, S: 'static>(
        &'a self,
        ctx: StoreContext<'s, S>,
    ) -> Result<&'a T> {
        self.data(ctx.into_inner())?
            .ok_or_else(|| Error::msg("extern ref must not be a wrapped anyref"))?
            .downcast_ref::<T>()
            .ok_or_else(|| Error::msg("Incorrect extern ref type."))
    }
}

impl WasmFunc<Engine> for Func {
    fn new<T: 'static>(
        mut ctx: impl AsContextMut<Engine, UserState = T>,
        ty: FuncType,
        func: impl 'static
            + Send
            + Sync
            + Fn(StoreContextMut<T>, &[Val<Engine>], &mut [Val<Engine>]) -> Result<()>,
    ) -> Self {
        let ty = func_type_into(ctx.as_context().engine(), ty);
        Self::new(wasmtime::Func::new(
            ctx.as_context_mut().into_inner(),
            ty,
            move |mut caller, args, results| {
                let input = args
                    .iter()
                    .cloned()
                    .map(value_from)
                    .collect::<Result<ArgumentVec<_>>>()?;
                let mut output = results
                    .iter()
                    .cloned()
                    .map(value_from)
                    .collect::<Result<ArgumentVec<_>>>()?;

                func(
                    StoreContextMut::new(wasmtime::AsContextMut::as_context_mut(&mut caller)),
                    &input,
                    &mut output,
                )?;

                for (i, result) in output.into_iter().enumerate() {
                    results[i] = value_into(result);
                }

                Ok(())
            },
        ))
    }

    fn ty(&self, ctx: impl AsContext<Engine>) -> FuncType {
        func_type_from(self.as_ref().ty(ctx.as_context().into_inner())).unwrap()
    }

    fn call<T>(
        &self,
        mut ctx: impl AsContextMut<Engine>,
        args: &[Val<Engine>],
        results: &mut [Val<Engine>],
    ) -> Result<()> {
        let mut input = ArgumentVec::with_capacity(args.len());
        input.extend(args.iter().cloned().map(value_into));

        let mut output = ArgumentVec::with_capacity(results.len());
        output.extend(results.iter().cloned().map(value_into));

        self.as_ref().call(
            ctx.as_context_mut().into_inner(),
            &input[..],
            &mut output[..],
        )?;

        for (i, result) in output.into_iter().enumerate() {
            results[i] = value_from(result)?;
        }

        Ok(())
    }
}

impl WasmGlobal<Engine> for Global {
    fn new(mut ctx: impl AsContextMut<Engine>, value: Val<Engine>, mutable: bool) -> Self {
        let ty = value_type_into(value.ty());
        let value = value_into(value);
        Self::new(
            wasmtime::Global::new(
                ctx.as_context_mut().into_inner(),
                wasmtime::GlobalType::new(
                    ty,
                    if mutable {
                        wasmtime::Mutability::Var
                    } else {
                        wasmtime::Mutability::Const
                    },
                ),
                value,
            )
            .expect("Could not create global."),
        )
    }

    fn ty(&self, ctx: impl AsContext<Engine>) -> GlobalType {
        global_type_from(self.as_ref().ty(ctx.as_context().into_inner())).unwrap()
    }

    fn set(&self, mut ctx: impl AsContextMut<Engine>, new_value: Val<Engine>) -> Result<()> {
        self.as_ref()
            .set(ctx.as_context_mut().into_inner(), value_into(new_value))
    }

    fn get(&self, mut ctx: impl AsContextMut<Engine>) -> Val<Engine> {
        value_from(self.as_ref().get(ctx.as_context_mut().into_inner())).unwrap()
    }
}

/// Wrapper around [`wasmtime::Instance`].
#[derive(Clone)]
pub struct Instance {
    /// The instance itself.
    instance: wasmtime::Instance,
    /// The instance exports.
    exports: Arc<FxHashMap<String, Export<Engine>>>,
}

impl Instance {
    #[must_use]
    /// Consume an `Instance` to obtain the inner [`wasmtime::Instance`].
    pub fn into_inner(self) -> wasmtime::Instance {
        self.instance
    }
}

impl From<Instance> for wasmtime::Instance {
    fn from(wrapper: Instance) -> Self {
        wrapper.into_inner()
    }
}

impl Deref for Instance {
    type Target = wasmtime::Instance;

    fn deref(&self) -> &Self::Target {
        &self.instance
    }
}

impl AsRef<wasmtime::Instance> for Instance {
    fn as_ref(&self) -> &wasmtime::Instance {
        &self.instance
    }
}

impl WasmInstance<Engine> for Instance {
    fn new(
        mut store: impl AsContextMut<Engine>,
        module: &Module,
        imports: &Imports<Engine>,
    ) -> Result<Self> {
        let mut linker = wasmtime::Linker::new(store.as_context().engine());

        for ((module, name), imp) in imports {
            linker.define(
                store.as_context().into_inner(),
                &module,
                &name,
                extern_into(imp),
            )?;
        }

        let res = linker.instantiate(store.as_context_mut().into_inner(), module)?;
        let exports = Arc::new(
            res.exports(store.as_context_mut())
                .map(|x| -> Result<_> {
                    Ok((
                        x.name().to_string(),
                        Export {
                            name: x.name().to_string(),
                            value: extern_from(x.into_extern())?,
                        },
                    ))
                })
                .collect::<Result<_>>()?,
        );

        Ok(Self {
            instance: res,
            exports,
        })
    }

    fn exports<'a>(&self, _: impl AsContext<Engine>) -> Box<dyn Iterator<Item = Export<Engine>>> {
        Box::new(
            self.exports
                .values()
                .cloned()
                .collect::<Vec<_>>()
                .into_iter(),
        )
    }

    fn get_export(&self, _: impl AsContext<Engine>, name: &str) -> Option<Extern<Engine>> {
        (*self.exports).get(name).map(|x| x.value.clone())
    }
}

impl WasmMemory<Engine> for Memory {
    fn new(mut ctx: impl AsContextMut<Engine>, ty: MemoryType) -> Result<Self> {
        wasmtime::Memory::new(ctx.as_context_mut().into_inner(), memory_type_into(ty))
            .map(Self::new)
    }

    fn ty(&self, ctx: impl AsContext<Engine>) -> MemoryType {
        memory_type_from(self.as_ref().ty(ctx.as_context().into_inner())).unwrap()
    }

    fn grow(&self, mut ctx: impl AsContextMut<Engine>, additional: u32) -> Result<u32> {
        self.as_ref()
            .grow(ctx.as_context_mut().into_inner(), u64::from(additional))
            .and_then(expect_memory32)
    }

    fn current_pages(&self, ctx: impl AsContext<Engine>) -> u32 {
        expect_memory32(self.as_ref().size(ctx.as_context().into_inner())).unwrap()
    }

    fn read(&self, ctx: impl AsContext<Engine>, offset: usize, buffer: &mut [u8]) -> Result<()> {
        self.as_ref()
            .read(ctx.as_context().into_inner(), offset, buffer)
            .map_err(Error::new)
    }

    fn write(
        &self,
        mut ctx: impl AsContextMut<Engine>,
        offset: usize,
        buffer: &[u8],
    ) -> Result<()> {
        self.as_ref()
            .write(ctx.as_context_mut().into_inner(), offset, buffer)
            .map_err(Error::new)
    }
}

impl WasmModule<Engine> for Module {
    fn new(engine: &Engine, bytes: &[u8]) -> Result<Self> {
        let module = wasmtime::Module::from_binary(engine, bytes)?;

        // pre-validate the module imports and exports
        for import in module.imports() {
            extern_type_from(import.ty().clone())?;
        }
        for export in module.exports() {
            extern_type_from(export.ty().clone())?;
        }

        Ok(Self::new(module))
    }

    fn exports(&self) -> Box<dyn '_ + Iterator<Item = ExportType<'_>>> {
        Box::new(self.as_ref().exports().map(|x| ExportType {
            name: x.name(),
            ty: extern_type_from(x.ty()).unwrap(),
        }))
    }

    fn get_export(&self, name: &str) -> Option<ExternType> {
        self.as_ref()
            .get_export(name)
            .map(extern_type_from)
            .transpose()
            .unwrap()
    }

    fn imports(&self) -> Box<dyn '_ + Iterator<Item = ImportType<'_>>> {
        Box::new(self.as_ref().imports().map(|x| ImportType {
            module: x.module(),
            name: x.name(),
            ty: extern_type_from(x.ty()).unwrap(),
        }))
    }
}

impl<T: 'static> WasmStore<T, Engine> for Store<T> {
    fn new(engine: &Engine, data: T) -> Self {
        Self::new(wasmtime::Store::new(engine, data))
    }

    fn engine(&self) -> &Engine {
        Engine::ref_cast(self.as_ref().engine())
    }

    fn data(&self) -> &T {
        self.as_ref().data()
    }

    fn data_mut(&mut self) -> &mut T {
        self.as_mut().data_mut()
    }

    fn into_data(self) -> T {
        self.into_inner().into_data()
    }
}

impl<T: 'static> AsContext<Engine> for Store<T> {
    type UserState = T;

    fn as_context(&self) -> StoreContext<'_, Self::UserState> {
        StoreContext::new(wasmtime::AsContext::as_context(self.as_ref()))
    }
}

impl<T: 'static> AsContextMut<Engine> for Store<T> {
    fn as_context_mut(&mut self) -> StoreContextMut<'_, Self::UserState> {
        StoreContextMut::new(wasmtime::AsContextMut::as_context_mut(self.as_mut()))
    }
}

impl<'a, T: 'static> WasmStoreContext<'a, T, Engine> for StoreContext<'a, T> {
    fn engine(&self) -> &Engine {
        Engine::ref_cast(self.as_ref().engine())
    }

    fn data(&self) -> &T {
        self.as_ref().data()
    }
}

impl<T: 'static> AsContext<Engine> for StoreContext<'_, T> {
    type UserState = T;

    fn as_context(&self) -> StoreContext<'_, T> {
        StoreContext::new(wasmtime::AsContext::as_context(self.as_ref()))
    }
}

impl<T: 'static> AsContext<Engine> for StoreContextMut<'_, T> {
    type UserState = T;

    fn as_context(&self) -> StoreContext<'_, T> {
        StoreContext::new(wasmtime::AsContext::as_context(self.as_ref()))
    }
}

impl<T: 'static> AsContextMut<Engine> for StoreContextMut<'_, T> {
    fn as_context_mut(&mut self) -> StoreContextMut<'_, T> {
        StoreContextMut::new(wasmtime::AsContextMut::as_context_mut(self.as_mut()))
    }
}

impl<'a, T: 'static> WasmStoreContext<'a, T, Engine> for StoreContextMut<'a, T> {
    fn engine(&self) -> &Engine {
        Engine::ref_cast(self.as_ref().engine())
    }

    fn data(&self) -> &T {
        self.as_ref().data()
    }
}

impl<'a, T: 'static> WasmStoreContextMut<'a, T, Engine> for StoreContextMut<'a, T> {
    fn data_mut(&mut self) -> &mut T {
        self.as_mut().data_mut()
    }
}

impl WasmTable<Engine> for Table {
    fn new(mut ctx: impl AsContextMut<Engine>, ty: TableType, init: Ref<Engine>) -> Result<Self> {
        wasmtime::Table::new(
            ctx.as_context_mut().into_inner(),
            table_type_into(ty),
            ref_into(init),
        )
        .map(Self::new)
    }

    fn ty(&self, ctx: impl AsContext<Engine>) -> TableType {
        table_type_from(self.as_ref().ty(ctx.as_context().into_inner())).unwrap()
    }

    fn size(&self, ctx: impl AsContext<Engine>) -> u32 {
        expect_table32(self.as_ref().size(ctx.as_context().into_inner())).unwrap()
    }

    fn grow(
        &self,
        mut ctx: impl AsContextMut<Engine>,
        delta: u32,
        init: Ref<Engine>,
    ) -> Result<u32> {
        self.as_ref()
            .grow(
                ctx.as_context_mut().into_inner(),
                u64::from(delta),
                ref_into(init),
            )
            .and_then(expect_table32)
    }

    fn get(&self, mut ctx: impl AsContextMut<Engine>, index: u32) -> Option<Ref<Engine>> {
        self.as_ref()
            .get(ctx.as_context_mut().into_inner(), u64::from(index))
            .map(ref_from)
            .transpose()
            .unwrap()
    }

    fn set(&self, mut ctx: impl AsContextMut<Engine>, index: u32, elem: Ref<Engine>) -> Result<()> {
        self.as_ref().set(
            ctx.as_context_mut().into_inner(),
            u64::from(index),
            ref_into(elem),
        )
    }
}

/// Convert a [`wasmtime::Val`] to a [`Val<Engine>`].
fn value_from(value: wasmtime::Val) -> Result<Val<Engine>> {
    match value {
        wasmtime::Val::I32(x) => Ok(Val::I32(x)),
        wasmtime::Val::I64(x) => Ok(Val::I64(x)),
        wasmtime::Val::F32(x) => Ok(Val::F32(f32::from_bits(x))),
        wasmtime::Val::F64(x) => Ok(Val::F64(f64::from_bits(x))),
        wasmtime::Val::V128(x) => Ok(Val::V128(x.as_u128())),
        wasmtime::Val::FuncRef(x) => Ok(Val::FuncRef(x.map(Func::new))),
        wasmtime::Val::ExternRef(x) => Ok(Val::ExternRef(x.map(ExternRef::new))),
        wasmtime::Val::AnyRef(_) => {
            bail!("anyref is not supported in the wasm_runtime_layer")
        }
        wasmtime::Val::ExnRef(_) => {
            bail!("exnref is not supported in the wasm_runtime_layer")
        }
        wasmtime::Val::ContRef(_) => {
            bail!("contref is not supported in the wasm_runtime_layer")
        }
    }
}

/// Convert a [`Val<Engine>`] to a [`wasmtime::Val`].
fn value_into(value: Val<Engine>) -> wasmtime::Val {
    match value {
        Val::I32(x) => wasmtime::Val::I32(x),
        Val::I64(x) => wasmtime::Val::I64(x),
        Val::F32(x) => wasmtime::Val::F32(x.to_bits()),
        Val::F64(x) => wasmtime::Val::F64(x.to_bits()),
        Val::V128(x) => wasmtime::Val::V128(x.into()),
        Val::FuncRef(x) => wasmtime::Val::FuncRef(x.map(Func::into_inner)),
        Val::ExternRef(x) => wasmtime::Val::ExternRef(x.map(ExternRef::into_inner)),
    }
}

/// Convert a [`wasmtime::ValType`] to a [`ValType`].
fn value_type_from(ty: wasmtime::ValType) -> Result<ValType> {
    match ty {
        wasmtime::ValType::I32 => Ok(ValType::I32),
        wasmtime::ValType::I64 => Ok(ValType::I64),
        wasmtime::ValType::F32 => Ok(ValType::F32),
        wasmtime::ValType::F64 => Ok(ValType::F64),
        wasmtime::ValType::V128 => Ok(ValType::V128),
        wasmtime::ValType::Ref(ty) => match ty {
            _ if wasmtime::RefType::eq(&ty, &wasmtime::RefType::FUNCREF) => Ok(ValType::FuncRef),
            _ if wasmtime::RefType::eq(&ty, &wasmtime::RefType::EXTERNREF) => {
                Ok(ValType::ExternRef)
            }
            ty => bail!("ref type {ty:?} is not supported in the wasm_runtime_layer"),
        },
    }
}

/// Convert a [`ValType`] to a [`wasmtime::ValType`].
fn value_type_into(ty: ValType) -> wasmtime::ValType {
    match ty {
        ValType::I32 => wasmtime::ValType::I32,
        ValType::I64 => wasmtime::ValType::I64,
        ValType::F32 => wasmtime::ValType::F32,
        ValType::F64 => wasmtime::ValType::F64,
        ValType::V128 => wasmtime::ValType::V128,
        ValType::FuncRef => wasmtime::ValType::Ref(wasmtime::RefType::FUNCREF),
        ValType::ExternRef => wasmtime::ValType::Ref(wasmtime::RefType::EXTERNREF),
    }
}

/// Convert a [`Ref<Engine>`] to a [`wasmtime::Ref`].
fn ref_into(r#ref: Ref<Engine>) -> wasmtime::Ref {
    match r#ref {
        Ref::FuncRef(x) => wasmtime::Ref::Func(x.map(Func::into_inner)),
        Ref::ExternRef(x) => wasmtime::Ref::Extern(x.map(ExternRef::into_inner)),
    }
}

/// Convert a [`wasmtime::Ref`] to a [`Ref<Engine>`].
fn ref_from(r#ref: wasmtime::Ref) -> Result<Ref<Engine>> {
    match r#ref {
        wasmtime::Ref::Func(x) => Ok(Ref::FuncRef(x.map(Func::from))),
        wasmtime::Ref::Extern(x) => Ok(Ref::ExternRef(x.map(ExternRef::from))),
        wasmtime::Ref::Any(_) => {
            bail!("anyref is not supported in the wasm_runtime_layer")
        }
        wasmtime::Ref::Exn(_) => {
            bail!("exnref is not supported in the wasm_runtime_layer")
        }
    }
}

/// Convert a [`wasmtime::ValType`] to a [`RefType`].
fn ref_type_from(ty: &wasmtime::RefType) -> Result<RefType> {
    match ty {
        _ if wasmtime::RefType::eq(ty, &wasmtime::RefType::FUNCREF) => Ok(RefType::FuncRef),
        _ if wasmtime::RefType::eq(ty, &wasmtime::RefType::EXTERNREF) => Ok(RefType::ExternRef),
        ty => bail!("ref type {ty:?} is not supported in the wasm_runtime_layer"),
    }
}

/// Convert a [`RefType`] to a [`wasmtime::RefType`].
fn ref_type_into(ty: RefType) -> wasmtime::RefType {
    match ty {
        RefType::FuncRef => wasmtime::RefType::FUNCREF,
        RefType::ExternRef => wasmtime::RefType::EXTERNREF,
    }
}

/// Convert a [`wasmtime::FuncType`] to a [`FuncType`].
fn func_type_from(ty: wasmtime::FuncType) -> Result<FuncType> {
    let params = ty
        .params()
        .map(value_type_from)
        .collect::<Result<ArgumentVec<_>>>()?;
    let results = ty
        .results()
        .map(value_type_from)
        .collect::<Result<ArgumentVec<_>>>()?;

    Ok(FuncType::new(params, results))
}

/// Convert a [`FuncType`] to a [`wasmtime::FuncType`].
fn func_type_into(engine: &Engine, ty: FuncType) -> wasmtime::FuncType {
    wasmtime::FuncType::new(
        Engine::ref_cast(engine),
        ty.params().iter().map(|&x| value_type_into(x)),
        ty.results().iter().map(|&x| value_type_into(x)),
    )
}

/// Convert a [`wasmtime::GlobalType`] to a [`GlobalType`].
fn global_type_from(ty: wasmtime::GlobalType) -> Result<GlobalType> {
    Ok(GlobalType::new(
        value_type_from(ty.content().clone())?,
        match ty.mutability() {
            wasmtime::Mutability::Const => false,
            wasmtime::Mutability::Var => true,
        },
    ))
}

/// Convert a [`wasmtime::MemoryType`] to a [`MemoryType`].
fn memory_type_from(ty: wasmtime::MemoryType) -> Result<MemoryType> {
    Ok(MemoryType::new(
        expect_memory32(ty.minimum())?,
        ty.maximum().map(expect_memory32).transpose()?,
    ))
}

/// Convert a memory size `u64` to a `u32` or return an error
fn expect_memory32(x: u64) -> Result<u32> {
    match x.try_into() {
        Ok(x) => Ok(x),
        Err(_) => bail!("memory64 is not supported in the wasm_runtime_layer"),
    }
}

/// Convert a [`MemoryType`] to a [`wasmtime::MemoryType`].
fn memory_type_into(ty: MemoryType) -> wasmtime::MemoryType {
    wasmtime::MemoryType::new(ty.initial_pages(), ty.maximum_pages())
}

/// Convert a [`wasmtime::TableType`] to a [`TableType`].
fn table_type_from(ty: wasmtime::TableType) -> Result<TableType> {
    Ok(TableType::new(
        ref_type_from(ty.element())?,
        expect_table32(ty.minimum())?,
        ty.maximum().map(expect_table32).transpose()?,
    ))
}

/// Convert a table size `u64` to a `u32` or return an error
fn expect_table32(x: u64) -> Result<u32> {
    match x.try_into() {
        Ok(x) => Ok(x),
        Err(_) => bail!("table64 is not supported in the wasm_runtime_layer"),
    }
}

/// Convert a [`TableType`] to a [`wasmtime::TableType`].
fn table_type_into(ty: TableType) -> wasmtime::TableType {
    wasmtime::TableType::new(ref_type_into(ty.element()), ty.minimum(), ty.maximum())
}

/// Convert a [`wasmtime::Extern`] to an [`Extern<Engine>`].
fn extern_from(value: wasmtime::Extern) -> Result<Extern<Engine>> {
    match value {
        wasmtime::Extern::Func(x) => Ok(Extern::Func(Func::new(x))),
        wasmtime::Extern::Global(x) => Ok(Extern::Global(Global::new(x))),
        wasmtime::Extern::Memory(x) => Ok(Extern::Memory(Memory::new(x))),
        wasmtime::Extern::Table(x) => Ok(Extern::Table(Table::new(x))),
        wasmtime::Extern::Tag(_) => {
            bail!("tags are not supported in the wasm_runtime_layer")
        }
        wasmtime::Extern::SharedMemory(_) => {
            bail!("shared memories are not supported in the wasm_runtime_layer")
        }
    }
}

/// Convert an [`Extern<Engine>`] to a [`wasmtime::Extern`].
fn extern_into(value: Extern<Engine>) -> wasmtime::Extern {
    match value {
        Extern::Func(x) => wasmtime::Extern::Func(x.into_inner()),
        Extern::Global(x) => wasmtime::Extern::Global(x.into_inner()),
        Extern::Memory(x) => wasmtime::Extern::Memory(x.into_inner()),
        Extern::Table(x) => wasmtime::Extern::Table(x.into_inner()),
    }
}

/// Convert a [`wasmtime::ExternType`] to an [`ExternType`].
fn extern_type_from(ty: wasmtime::ExternType) -> Result<ExternType> {
    match ty {
        wasmtime::ExternType::Func(x) => Ok(ExternType::Func(func_type_from(x)?)),
        wasmtime::ExternType::Global(x) => Ok(ExternType::Global(global_type_from(x)?)),
        wasmtime::ExternType::Memory(x) => Ok(ExternType::Memory(memory_type_from(x)?)),
        wasmtime::ExternType::Table(x) => Ok(ExternType::Table(table_type_from(x)?)),
        wasmtime::ExternType::Tag(_) => {
            bail!("tags are not supported in the wasm_runtime_layer")
        }
    }
}
