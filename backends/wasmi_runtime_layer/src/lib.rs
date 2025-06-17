#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::missing_docs_in_private_items)]
#![cfg_attr(not(feature = "std"), no_std)]

//! `wasmi_runtime_layer` implements the `wasm_runtime_layer` abstraction interface over WebAssembly runtimes for `Wasmi`.

extern crate alloc;

use alloc::{boxed::Box, string::ToString, vec::Vec};
use core::{
    fmt,
    ops::{Deref, DerefMut},
};

use anyhow::{Error, Result};
use ref_cast::RefCast;
use smallvec::SmallVec;
use wasm_runtime_layer::{
    backend::{
        AsContext, AsContextMut, Export, Extern, Imports, Value, WasmEngine, WasmExternRef,
        WasmFunc, WasmGlobal, WasmInstance, WasmMemory, WasmModule, WasmStore, WasmStoreContext,
        WasmStoreContextMut, WasmTable,
    },
    ExportType, ExternType, FuncType, GlobalType, ImportType, MemoryType, TableType, ValueType,
};

/// The default amount of arguments and return values for which to allocate
/// stack space.
const DEFAULT_ARGUMENT_SIZE: usize = 4;

/// A vector which allocates up to the default number of arguments on the stack.
type ArgumentVec<T> = SmallVec<[T; DEFAULT_ARGUMENT_SIZE]>;

/// Generate the boilerplate delegation code for a newtype wrapper.
macro_rules! delegate {
    (#[derive($($derive:ident),*)] $newtype:ident($inner:ty) $($tt:tt)*) => {
        #[derive($($derive,)* RefCast)]
        #[repr(transparent)]
        #[doc = concat!("Newtype wrapper around [`", stringify!($inner), "`].")]
        pub struct $newtype$($tt)*($inner);

        impl$($tt)* $newtype$($tt)* {
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

        impl$($tt)* From<$inner> for $newtype$($tt)* {
            fn from(inner: $inner) -> Self {
                Self::new(inner)
            }
        }

        impl$($tt)* From<$newtype$($tt)*> for $inner {
            fn from(wrapper: $newtype$($tt)*) -> Self {
                wrapper.into_inner()
            }
        }

        impl$($tt)* Deref for $newtype$($tt)* {
            type Target = $inner;

            fn deref(&self) -> &Self::Target {
                &self.0
            }
        }

        impl$($tt)* DerefMut for $newtype$($tt)* {
            fn deref_mut(&mut self) -> &mut Self::Target {
                &mut self.0
            }
        }

        impl$($tt)* AsRef<$inner> for $newtype$($tt)* {
            fn as_ref(&self) -> &$inner {
                &self.0
            }
        }

        impl$($tt)* AsMut<$inner> for $newtype$($tt)* {
            fn as_mut(&mut self) -> &mut $inner {
                &mut self.0
            }
        }

        impl$($tt)* AsRef<$newtype$($tt)*> for $inner {
            fn as_ref(&self) -> &$newtype$($tt)* {
                $newtype::ref_cast(self)
            }
        }

        impl$($tt)* AsMut<$newtype$($tt)*> for $inner {
            fn as_mut(&mut self) -> &mut $newtype$($tt)* {
                $newtype::ref_cast_mut(self)
            }
        }
    }
}

delegate! { #[derive(Clone, Default)] Engine(wasmi::Engine) }
// delegate! { #[derive(Clone)] ExternRef(wasmi::ExternRef) }
delegate! { #[derive(Clone)] Func(wasmi::Func) }
delegate! { #[derive(Clone)] Global(wasmi::Global) }
delegate! { #[derive(Clone)] Instance(wasmi::Instance) }
delegate! { #[derive(Clone)] Memory(wasmi::Memory) }
delegate! { #[derive(Clone)] Module(wasmi::Module) }
delegate! { #[derive()] Store(wasmi::Store<T>) <T> }
delegate! { #[derive()] StoreContext(wasmi::StoreContext<'a, T>) <'a, T> }
delegate! { #[derive()] StoreContextMut(wasmi::StoreContextMut<'a, T>) <'a, T> }
delegate! { #[derive(Clone)] Table(wasmi::Table) }

impl WasmEngine for Engine {
    type ExternRef = ExternRef;
    type Func = Func;
    type Global = Global;
    type Instance = Instance;
    type Memory = Memory;
    type Module = Module;
    type Store<T> = Store<T>;
    type StoreContext<'a, T: 'a> = StoreContext<'a, T>;
    type StoreContextMut<'a, T: 'a> = StoreContextMut<'a, T>;
    type Table = Table;
}

#[derive(Clone)]
#[repr(transparent)]
/// Newtype wrapper around [`wasmi::ExternRef`], which ensures it is always [`Some`].
pub struct ExternRef(wasmi::ExternRef);

impl ExternRef {
    /// Create an optional [`wasm_runtime_layer::ExternRef`]-compatible `ExternRef`
    /// from a [`wasmi::ExternRef`].
    pub fn new(inner: wasmi::ExternRef) -> Option<Self> {
        if inner.is_null() {
            None
        } else {
            Some(Self(inner))
        }
    }

    #[must_use]
    /// Consume an `ExternRef` to obtain the inner [`wasmi::ExternRef`].
    pub fn into_inner(self) -> wasmi::ExternRef {
        self.0
    }
}

impl From<ExternRef> for wasmi::ExternRef {
    fn from(wrapper: ExternRef) -> Self {
        wrapper.into_inner()
    }
}

impl Deref for ExternRef {
    type Target = wasmi::ExternRef;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl AsRef<wasmi::ExternRef> for ExternRef {
    fn as_ref(&self) -> &wasmi::ExternRef {
        &self.0
    }
}

impl WasmExternRef<Engine> for ExternRef {
    fn new<T: 'static + Send + Sync>(mut ctx: impl AsContextMut<Engine>, object: T) -> Self {
        Self(wasmi::ExternRef::new::<T>(
            ctx.as_context_mut().into_inner(),
            object,
        ))
    }

    fn downcast<'a, 's: 'a, T: 'static, S: 'a>(
        &'a self,
        store: StoreContext<'s, S>,
    ) -> Result<&'a T> {
        self.as_ref()
            .data(store)
            .ok_or_else(|| Error::msg("externref None should be external"))?
            .downcast_ref()
            .ok_or_else(|| Error::msg("Incorrect extern ref type."))
    }
}

impl WasmFunc<Engine> for Func {
    fn new<T>(
        mut ctx: impl AsContextMut<Engine, UserState = T>,
        ty: FuncType,
        func: impl 'static
            + Send
            + Sync
            + Fn(StoreContextMut<T>, &[Value<Engine>], &mut [Value<Engine>]) -> Result<()>,
    ) -> Self {
        Self::new(wasmi::Func::new(
            ctx.as_context_mut().into_inner(),
            func_type_into(ty),
            move |mut caller, args, results| {
                let mut input = ArgumentVec::with_capacity(args.len());
                input.extend(args.iter().cloned().map(value_from));

                let mut output = ArgumentVec::with_capacity(results.len());
                output.extend(results.iter().cloned().map(value_from));

                func(
                    StoreContextMut::new(wasmi::AsContextMut::as_context_mut(&mut caller)),
                    &input,
                    &mut output,
                )
                .map_err(HostError)
                .map_err(wasmi::Error::host)?;

                for (i, result) in output.into_iter().enumerate() {
                    results[i] = value_into(result);
                }

                Ok(())
            },
        ))
    }

    fn ty(&self, ctx: impl AsContext<Engine>) -> FuncType {
        func_type_from(self.as_ref().ty(ctx.as_context().into_inner()))
    }

    fn call<T>(
        &self,
        mut ctx: impl AsContextMut<Engine>,
        args: &[Value<Engine>],
        results: &mut [Value<Engine>],
    ) -> Result<()> {
        let mut input = ArgumentVec::with_capacity(args.len());
        input.extend(args.iter().cloned().map(value_into));

        let mut output = ArgumentVec::with_capacity(results.len());
        output.extend(results.iter().cloned().map(value_into));

        self.as_ref()
            .call(
                ctx.as_context_mut().into_inner(),
                &input[..],
                &mut output[..],
            )
            .map_err(Error::new)?;

        for (i, result) in output.into_iter().enumerate() {
            results[i] = value_from(result);
        }

        Ok(())
    }
}

impl WasmGlobal<Engine> for Global {
    fn new(mut ctx: impl AsContextMut<Engine>, value: Value<Engine>, mutable: bool) -> Self {
        Self::new(wasmi::Global::new(
            ctx.as_context_mut().into_inner(),
            value_into(value),
            if mutable {
                wasmi::Mutability::Var
            } else {
                wasmi::Mutability::Const
            },
        ))
    }

    fn ty(&self, ctx: impl AsContext<Engine>) -> GlobalType {
        global_type_from(self.as_ref().ty(ctx.as_context().into_inner()))
    }

    fn set(&self, mut ctx: impl AsContextMut<Engine>, new_value: Value<Engine>) -> Result<()> {
        self.as_ref()
            .set(ctx.as_context_mut().into_inner(), value_into(new_value))
            .map_err(Error::new)
    }

    fn get(&self, ctx: impl AsContextMut<Engine>) -> Value<Engine> {
        value_from(self.as_ref().get(ctx.as_context().into_inner()))
    }
}

impl WasmInstance<Engine> for Instance {
    fn new(
        mut store: impl AsContextMut<Engine>,
        module: &Module,
        imports: &Imports<Engine>,
    ) -> Result<Self> {
        let mut linker = wasmi::Linker::new(store.as_context().engine().as_ref());

        for ((module, name), imp) in imports {
            linker
                .define(&module, &name, extern_into(imp))
                .map_err(Error::new)?;
        }

        let pre = linker
            .instantiate(store.as_context_mut().into_inner(), module.as_ref())
            .map_err(Error::new)?;
        Ok(Self::new(
            pre.start(store.as_context_mut().into_inner())
                .map_err(Error::new)?,
        ))
    }

    fn exports<'a>(
        &self,
        store: impl AsContext<Engine>,
    ) -> Box<dyn Iterator<Item = Export<Engine>>> {
        Box::new(
            self.as_ref()
                .exports(store.as_context().into_inner())
                .map(|x| Export {
                    name: x.name().to_string(),
                    value: extern_from(x.into_extern()),
                })
                .collect::<Vec<_>>()
                .into_iter(),
        )
    }

    fn get_export(&self, store: impl AsContext<Engine>, name: &str) -> Option<Extern<Engine>> {
        self.as_ref()
            .get_export(store.as_context().into_inner(), name)
            .map(extern_from)
    }
}

impl WasmMemory<Engine> for Memory {
    fn new(mut ctx: impl AsContextMut<Engine>, ty: MemoryType) -> Result<Self> {
        wasmi::Memory::new(ctx.as_context_mut().into_inner(), memory_type_into(ty))
            .map(Self::new)
            .map_err(Error::new)
    }

    fn ty(&self, ctx: impl AsContext<Engine>) -> MemoryType {
        memory_type_from(self.as_ref().ty(ctx.as_context().into_inner()))
    }

    fn grow(&self, mut ctx: impl AsContextMut<Engine>, additional: u32) -> Result<u32> {
        self.as_ref()
            .grow(ctx.as_context_mut().into_inner(), additional as u64)
            .map(expect_memory32)
            .map_err(Error::new)
    }

    fn current_pages(&self, ctx: impl AsContext<Engine>) -> u32 {
        expect_memory32(self.as_ref().size(ctx.as_context().into_inner()))
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
        Ok(Self::new(
            wasmi::Module::new(engine.as_ref(), bytes).map_err(Error::new)?,
        ))
    }

    fn exports(&self) -> Box<dyn '_ + Iterator<Item = ExportType<'_>>> {
        Box::new(self.as_ref().exports().map(|x| ExportType {
            name: x.name(),
            ty: extern_type_from(x.ty().clone()),
        }))
    }

    fn get_export(&self, name: &str) -> Option<ExternType> {
        self.as_ref().get_export(name).map(extern_type_from)
    }

    fn imports(&self) -> Box<dyn '_ + Iterator<Item = ImportType<'_>>> {
        Box::new(self.as_ref().imports().map(|x| ImportType {
            module: x.module(),
            name: x.name(),
            ty: extern_type_from(x.ty().clone()),
        }))
    }
}

impl<T> WasmStore<T, Engine> for Store<T> {
    fn new(engine: &Engine, data: T) -> Self {
        Self::new(wasmi::Store::new(engine.as_ref(), data))
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

impl<T> AsContext<Engine> for Store<T> {
    type UserState = T;

    fn as_context(&self) -> StoreContext<'_, Self::UserState> {
        StoreContext::new(wasmi::AsContext::as_context(self.as_ref()))
    }
}

impl<T> AsContextMut<Engine> for Store<T> {
    fn as_context_mut(&mut self) -> StoreContextMut<'_, Self::UserState> {
        StoreContextMut::new(wasmi::AsContextMut::as_context_mut(self.as_mut()))
    }
}

impl<'a, T> WasmStoreContext<'a, T, Engine> for StoreContext<'a, T> {
    fn engine(&self) -> &Engine {
        Engine::ref_cast(self.as_ref().engine())
    }

    fn data(&self) -> &T {
        self.as_ref().data()
    }
}

impl<T> AsContext<Engine> for StoreContext<'_, T> {
    type UserState = T;

    fn as_context(&self) -> StoreContext<T> {
        StoreContext::new(wasmi::AsContext::as_context(self.as_ref()))
    }
}

impl<T> AsContext<Engine> for StoreContextMut<'_, T> {
    type UserState = T;

    fn as_context(&self) -> StoreContext<T> {
        StoreContext::new(wasmi::AsContext::as_context(self.as_ref()))
    }
}

impl<T> AsContextMut<Engine> for StoreContextMut<'_, T> {
    fn as_context_mut(&mut self) -> StoreContextMut<T> {
        StoreContextMut::new(wasmi::AsContextMut::as_context_mut(self.as_mut()))
    }
}

impl<'a, T> WasmStoreContext<'a, T, Engine> for StoreContextMut<'a, T> {
    fn engine(&self) -> &Engine {
        Engine::ref_cast(self.as_ref().engine())
    }

    fn data(&self) -> &T {
        self.as_ref().data()
    }
}

impl<'a, T> WasmStoreContextMut<'a, T, Engine> for StoreContextMut<'a, T> {
    fn data_mut(&mut self) -> &mut T {
        self.as_mut().data_mut()
    }
}

impl WasmTable<Engine> for Table {
    fn new(mut ctx: impl AsContextMut<Engine>, ty: TableType, init: Value<Engine>) -> Result<Self> {
        wasmi::Table::new(
            ctx.as_context_mut().into_inner(),
            table_type_into(ty),
            value_into(init),
        )
        .map(Self::new)
        .map_err(Error::new)
    }

    fn ty(&self, ctx: impl AsContext<Engine>) -> TableType {
        table_type_from(self.as_ref().ty(ctx.as_context().into_inner()))
    }

    fn size(&self, ctx: impl AsContext<Engine>) -> u32 {
        expect_table32(self.as_ref().size(ctx.as_context().into_inner()))
    }

    fn grow(
        &self,
        mut ctx: impl AsContextMut<Engine>,
        delta: u32,
        init: Value<Engine>,
    ) -> Result<u32> {
        self.as_ref()
            .grow(
                ctx.as_context_mut().into_inner(),
                delta as u64,
                value_into(init),
            )
            .map(expect_table32)
            .map_err(Error::new)
    }

    fn get(&self, ctx: impl AsContextMut<Engine>, index: u32) -> Option<Value<Engine>> {
        self.as_ref()
            .get(ctx.as_context().into_inner(), index as u64)
            .map(value_from)
    }

    fn set(
        &self,
        mut ctx: impl AsContextMut<Engine>,
        index: u32,
        value: Value<Engine>,
    ) -> Result<()> {
        self.as_ref()
            .set(
                ctx.as_context_mut().into_inner(),
                index as u64,
                value_into(value),
            )
            .map_err(Error::new)
    }
}

/// Convert a [`wasmi::Val`] to a [`Value<Engine>`].
fn value_from(value: wasmi::Val) -> Value<Engine> {
    match value {
        wasmi::Val::I32(x) => Value::I32(x),
        wasmi::Val::I64(x) => Value::I64(x),
        wasmi::Val::F32(x) => Value::F32(x.to_float()),
        wasmi::Val::F64(x) => Value::F64(x.to_float()),
        wasmi::Val::V128(_) => unimplemented!("v128 is not supported in the wasm_runtime_layer"),
        wasmi::Val::FuncRef(x) => Value::FuncRef(x.func().copied().map(Func::new)),
        wasmi::Val::ExternRef(x) => Value::ExternRef(ExternRef::new(x)),
    }
}

/// Convert a [`Value<Engine>`] to a [`wasmi::Val`].
fn value_into(value: Value<Engine>) -> wasmi::Val {
    match value {
        Value::I32(x) => wasmi::Val::I32(x),
        Value::I64(x) => wasmi::Val::I64(x),
        Value::F32(x) => wasmi::Val::F32(wasmi::core::F32::from_float(x)),
        Value::F64(x) => wasmi::Val::F64(wasmi::core::F64::from_float(x)),
        Value::FuncRef(x) => wasmi::Val::FuncRef(wasmi::FuncRef::new(x.map(Func::into_inner))),
        Value::ExternRef(x) => wasmi::Val::ExternRef(
            x.map(ExternRef::into_inner)
                .unwrap_or_else(wasmi::ExternRef::null),
        ),
    }
}

/// Convert a [`wasmi::core::ValType`] to a [`ValueType`].
fn value_type_from(ty: wasmi::core::ValType) -> ValueType {
    match ty {
        wasmi::core::ValType::I32 => ValueType::I32,
        wasmi::core::ValType::I64 => ValueType::I64,
        wasmi::core::ValType::F32 => ValueType::F32,
        wasmi::core::ValType::F64 => ValueType::F64,
        wasmi::core::ValType::V128 => {
            unimplemented!("v128 is not supported in the wasm_runtime_layer")
        }
        wasmi::core::ValType::FuncRef => ValueType::FuncRef,
        wasmi::core::ValType::ExternRef => ValueType::ExternRef,
    }
}

/// Convert a [`ValueType`] to a [`wasmi::core::ValType`].
fn value_type_into(ty: ValueType) -> wasmi::core::ValType {
    match ty {
        ValueType::I32 => wasmi::core::ValType::I32,
        ValueType::I64 => wasmi::core::ValType::I64,
        ValueType::F32 => wasmi::core::ValType::F32,
        ValueType::F64 => wasmi::core::ValType::F64,
        ValueType::FuncRef => wasmi::core::ValType::FuncRef,
        ValueType::ExternRef => wasmi::core::ValType::ExternRef,
    }
}

/// Convert a [`wasmi::FuncType`] to a [`FuncType`].
fn func_type_from(ty: wasmi::FuncType) -> FuncType {
    FuncType::new(
        ty.params().iter().cloned().map(value_type_from),
        ty.results().iter().cloned().map(value_type_from),
    )
}

/// Convert a [`FuncType`] to a [`wasmi::FuncType`].
fn func_type_into(ty: FuncType) -> wasmi::FuncType {
    wasmi::FuncType::new(
        ty.params().iter().map(|&x| value_type_into(x)),
        ty.results().iter().map(|&x| value_type_into(x)),
    )
}

/// Convert a [`wasmi::GlobalType`] to a [`GlobalType`].
fn global_type_from(ty: wasmi::GlobalType) -> GlobalType {
    GlobalType::new(value_type_from(ty.content()), ty.mutability().is_mut())
}

/// Convert a [`wasmi::MemoryType`] to a [`MemoryType`].
fn memory_type_from(ty: wasmi::MemoryType) -> MemoryType {
    MemoryType::new(
        expect_memory32(ty.minimum()),
        ty.maximum().map(expect_memory32),
    )
}

/// Convert a memory size `u64` to a `u32` or panic
fn expect_memory32(x: u64) -> u32 {
    x.try_into().expect("memory64 is not supported")
}

/// Convert a [`MemoryType`] to a [`wasmi::MemoryType`].
fn memory_type_into(ty: MemoryType) -> wasmi::MemoryType {
    wasmi::MemoryType::new(ty.initial_pages(), ty.maximum_pages())
}

/// Convert a [`wasmi::TableType`] to a [`TableType`].
fn table_type_from(ty: wasmi::TableType) -> TableType {
    TableType::new(
        value_type_from(ty.element()),
        expect_table32(ty.minimum()),
        ty.maximum().map(expect_table32),
    )
}

/// Convert a table size `u64` to a `u32` or panic
fn expect_table32(x: u64) -> u32 {
    x.try_into().expect("table64 is not supported")
}

/// Convert a [`TableType`] to a [`wasmi::TableType`].
fn table_type_into(ty: TableType) -> wasmi::TableType {
    wasmi::TableType::new(value_type_into(ty.element()), ty.minimum(), ty.maximum())
}

/// Convert a [`wasmi::Extern`] to an [`Extern<Engine>`].
fn extern_from(value: wasmi::Extern) -> Extern<Engine> {
    match value {
        wasmi::Extern::Func(x) => Extern::Func(Func::new(x)),
        wasmi::Extern::Global(x) => Extern::Global(Global::new(x)),
        wasmi::Extern::Memory(x) => Extern::Memory(Memory::new(x)),
        wasmi::Extern::Table(x) => Extern::Table(Table::new(x)),
    }
}

/// Convert an [`Extern<Engine>`] to a [`wasmi::Extern`].
fn extern_into(value: Extern<Engine>) -> wasmi::Extern {
    match value {
        Extern::Func(x) => wasmi::Extern::Func(x.into_inner()),
        Extern::Global(x) => wasmi::Extern::Global(x.into_inner()),
        Extern::Memory(x) => wasmi::Extern::Memory(x.into_inner()),
        Extern::Table(x) => wasmi::Extern::Table(x.into_inner()),
    }
}

/// Convert a [`wasmi::ExternType`] to an [`ExternType`].
fn extern_type_from(ty: wasmi::ExternType) -> ExternType {
    match ty {
        wasmi::ExternType::Func(x) => ExternType::Func(func_type_from(x)),
        wasmi::ExternType::Global(x) => ExternType::Global(global_type_from(x)),
        wasmi::ExternType::Memory(x) => ExternType::Memory(memory_type_from(x)),
        wasmi::ExternType::Table(x) => ExternType::Table(table_type_from(x)),
    }
}

/// Represents a `wasmi` error derived from `anyhow`.
#[derive(Debug)]
struct HostError(anyhow::Error);

impl fmt::Display for HostError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl wasmi::core::HostError for HostError {}
