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

use anyhow::{Error, Result};
use fxhash::FxHashMap;
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
            + Fn(StoreContextMut<T>, &[Value<Engine>], &mut [Value<Engine>]) -> Result<()>,
    ) -> Self {
        let ty = func_type_into(ctx.as_context().engine(), ty);
        Self::new(wasmtime::Func::new(
            ctx.as_context_mut().into_inner(),
            ty,
            move |mut caller, args, results| {
                let mut input = ArgumentVec::with_capacity(args.len());
                input.extend(args.iter().cloned().map(value_from));

                let mut output = ArgumentVec::with_capacity(results.len());
                output.extend(results.iter().cloned().map(value_from));

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

        self.as_ref().call(
            ctx.as_context_mut().into_inner(),
            &input[..],
            &mut output[..],
        )?;

        for (i, result) in output.into_iter().enumerate() {
            results[i] = value_from(result);
        }

        Ok(())
    }
}

impl WasmGlobal<Engine> for Global {
    fn new(mut ctx: impl AsContextMut<Engine>, value: Value<Engine>, mutable: bool) -> Self {
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
        global_type_from(self.as_ref().ty(ctx.as_context().into_inner()))
    }

    fn set(&self, mut ctx: impl AsContextMut<Engine>, new_value: Value<Engine>) -> Result<()> {
        self.as_ref()
            .set(ctx.as_context_mut().into_inner(), value_into(new_value))
    }

    fn get(&self, mut ctx: impl AsContextMut<Engine>) -> Value<Engine> {
        value_from(self.as_ref().get(ctx.as_context_mut().into_inner()))
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
                .map(|x| {
                    (
                        x.name().to_string(),
                        Export {
                            name: x.name().to_string(),
                            value: extern_from(x.into_extern()),
                        },
                    )
                })
                .collect(),
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
        memory_type_from(self.as_ref().ty(ctx.as_context().into_inner()))
    }

    fn grow(&self, mut ctx: impl AsContextMut<Engine>, additional: u32) -> Result<u32> {
        self.as_ref()
            .grow(ctx.as_context_mut().into_inner(), additional as u64)
            .map(expect_memory32)
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
        wasmtime::Module::from_binary(engine, bytes).map(Self::new)
    }

    fn exports(&self) -> Box<dyn '_ + Iterator<Item = ExportType<'_>>> {
        Box::new(self.as_ref().exports().map(|x| ExportType {
            name: x.name(),
            ty: extern_type_from(x.ty()),
        }))
    }

    fn get_export(&self, name: &str) -> Option<ExternType> {
        self.as_ref().get_export(name).map(extern_type_from)
    }

    fn imports(&self) -> Box<dyn '_ + Iterator<Item = ImportType<'_>>> {
        Box::new(self.as_ref().imports().map(|x| ImportType {
            module: x.module(),
            name: x.name(),
            ty: extern_type_from(x.ty()),
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

    fn as_context(&self) -> StoreContext<T> {
        StoreContext::new(wasmtime::AsContext::as_context(self.as_ref()))
    }
}

impl<T: 'static> AsContext<Engine> for StoreContextMut<'_, T> {
    type UserState = T;

    fn as_context(&self) -> StoreContext<T> {
        StoreContext::new(wasmtime::AsContext::as_context(self.as_ref()))
    }
}

impl<T: 'static> AsContextMut<Engine> for StoreContextMut<'_, T> {
    fn as_context_mut(&mut self) -> StoreContextMut<T> {
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
    fn new(mut ctx: impl AsContextMut<Engine>, ty: TableType, init: Value<Engine>) -> Result<Self> {
        wasmtime::Table::new(
            ctx.as_context_mut().into_inner(),
            table_type_into(ty),
            value_into_ref(init),
        )
        .map(Self::new)
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
                u64::from(delta),
                value_into_ref(init),
            )
            .map(expect_table32)
    }

    fn get(&self, mut ctx: impl AsContextMut<Engine>, index: u32) -> Option<Value<Engine>> {
        self.as_ref()
            .get(ctx.as_context_mut().into_inner(), u64::from(index))
            .map(value_from_ref)
    }

    fn set(
        &self,
        mut ctx: impl AsContextMut<Engine>,
        index: u32,
        value: Value<Engine>,
    ) -> Result<()> {
        self.as_ref().set(
            ctx.as_context_mut().into_inner(),
            u64::from(index),
            value_into_ref(value),
        )
    }
}

/// Convert a [`wasmtime::Val`] to a [`Value<Engine>`].
fn value_from(value: wasmtime::Val) -> Value<Engine> {
    match value {
        wasmtime::Val::I32(x) => Value::I32(x),
        wasmtime::Val::I64(x) => Value::I64(x),
        wasmtime::Val::F32(x) => Value::F32(f32::from_bits(x)),
        wasmtime::Val::F64(x) => Value::F64(f64::from_bits(x)),
        wasmtime::Val::FuncRef(x) => Value::FuncRef(x.map(Func::new)),
        wasmtime::Val::ExternRef(x) => Value::ExternRef(x.map(ExternRef::new)),
        wasmtime::Val::V128(_) => unimplemented!("v128 is not supported in the wasm_runtime_layer"),
        wasmtime::Val::AnyRef(_) => {
            unimplemented!("anyref is not supported in the wasm_runtime_layer")
        }
    }
}

/// Convert a [`Value<Engine>`] to a [`wasmtime::Val`].
fn value_into(value: Value<Engine>) -> wasmtime::Val {
    match value {
        Value::I32(x) => wasmtime::Val::I32(x),
        Value::I64(x) => wasmtime::Val::I64(x),
        Value::F32(x) => wasmtime::Val::F32(x.to_bits()),
        Value::F64(x) => wasmtime::Val::F64(x.to_bits()),
        Value::FuncRef(x) => wasmtime::Val::FuncRef(x.map(Func::into_inner)),
        Value::ExternRef(x) => wasmtime::Val::ExternRef(x.map(ExternRef::into_inner)),
    }
}

/// Convert a [`wasmtime::ValType`] to a [`ValueType`].
fn value_type_from(ty: wasmtime::ValType) -> ValueType {
    match ty {
        wasmtime::ValType::I32 => ValueType::I32,
        wasmtime::ValType::I64 => ValueType::I64,
        wasmtime::ValType::F32 => ValueType::F32,
        wasmtime::ValType::F64 => ValueType::F64,
        wasmtime::ValType::V128 => {
            unimplemented!("v128 is not supported in the wasm_runtime_layer")
        }
        wasmtime::ValType::Ref(ty) => value_type_from_ref_type(&ty),
    }
}

/// Convert a [`ValueType`] to a [`wasmtime::ValType`].
fn value_type_into(ty: ValueType) -> wasmtime::ValType {
    match ty {
        ValueType::I32 => wasmtime::ValType::I32,
        ValueType::I64 => wasmtime::ValType::I64,
        ValueType::F32 => wasmtime::ValType::F32,
        ValueType::F64 => wasmtime::ValType::F64,
        ValueType::FuncRef => wasmtime::ValType::Ref(wasmtime::RefType::FUNCREF),
        ValueType::ExternRef => wasmtime::ValType::Ref(wasmtime::RefType::EXTERNREF),
    }
}

/// Convert a [`Value<Engine>`] to a [`wasmtime::Ref`].
fn value_into_ref(value: Value<Engine>) -> wasmtime::Ref {
    match value {
        Value::FuncRef(x) => wasmtime::Ref::Func(x.map(Func::into_inner)),
        Value::ExternRef(x) => wasmtime::Ref::Extern(x.map(ExternRef::into_inner)),
        Value::I32(_) | Value::I64(_) | Value::F32(_) | Value::F64(_) => {
            panic!("Attempt to convert non-reference value to a reference")
        }
    }
}

/// Convert a [`wasmtime::Ref`] to a [`Value<Engine>`].
fn value_from_ref(ref_: wasmtime::Ref) -> Value<Engine> {
    match ref_ {
        wasmtime::Ref::Func(x) => Value::FuncRef(x.map(Func::from)),
        wasmtime::Ref::Extern(x) => Value::ExternRef(x.map(ExternRef::from)),
        wasmtime::Ref::Any(_) => {
            unimplemented!("anyref is not supported in the wasm_runtime_layer")
        }
    }
}

/// Convert a [`wasmtime::ValType`] to a [`ValueType`].
fn value_type_from_ref_type(ty: &wasmtime::RefType) -> ValueType {
    match ty {
        _ if wasmtime::RefType::eq(ty, &wasmtime::RefType::FUNCREF) => ValueType::FuncRef,
        _ if wasmtime::RefType::eq(ty, &wasmtime::RefType::EXTERNREF) => ValueType::ExternRef,
        // TODO: is the matching against FuncRef correct here
        _ => unimplemented!("anyref is not supported in the wasm_runtime_layer"),
    }
}

/// Convert a [`ValueType`] to a [`wasmtime::ValType`].
fn value_type_into_ref_type(ty: ValueType) -> wasmtime::RefType {
    match ty {
        ValueType::FuncRef => wasmtime::RefType::FUNCREF,
        ValueType::ExternRef => wasmtime::RefType::EXTERNREF,
        ValueType::I32 | ValueType::I64 | ValueType::F32 | ValueType::F64 => {
            panic!("Attempt to convert non-reference type to a reference type")
        }
    }
}

/// Convert a [`wasmtime::FuncType`] to a [`FuncType`].
fn func_type_from(ty: wasmtime::FuncType) -> FuncType {
    FuncType::new(
        ty.params().map(value_type_from),
        ty.results().map(value_type_from),
    )
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
fn global_type_from(ty: wasmtime::GlobalType) -> GlobalType {
    GlobalType::new(
        value_type_from(ty.content().clone()),
        matches!(ty.mutability(), wasmtime::Mutability::Var),
    )
}

/// Convert a [`wasmtime::MemoryType`] to a [`MemoryType`].
fn memory_type_from(ty: wasmtime::MemoryType) -> MemoryType {
    MemoryType::new(
        expect_memory32(ty.minimum()),
        ty.maximum().map(expect_memory32),
    )
}

/// Convert a memory size `u64` to a `u32` or panic
fn expect_memory32(x: u64) -> u32 {
    x.try_into().expect("memory64 is not supported")
}

/// Convert a [`MemoryType`] to a [`wasmtime::MemoryType`].
fn memory_type_into(ty: MemoryType) -> wasmtime::MemoryType {
    wasmtime::MemoryType::new(ty.initial_pages(), ty.maximum_pages())
}

/// Convert a [`wasmtime::TableType`] to a [`TableType`].
fn table_type_from(ty: wasmtime::TableType) -> TableType {
    TableType::new(
        value_type_from_ref_type(ty.element()),
        expect_table32(ty.minimum()),
        ty.maximum().map(expect_table32),
    )
}

/// Convert a table size `u64` to a `u32` or panic
fn expect_table32(x: u64) -> u32 {
    x.try_into().expect("table64 is not supported")
}

/// Convert a [`TableType`] to a [`wasmtime::TableType`].
fn table_type_into(ty: TableType) -> wasmtime::TableType {
    wasmtime::TableType::new(
        value_type_into_ref_type(ty.element()),
        ty.minimum(),
        ty.maximum(),
    )
}

/// Convert a [`wasmtime::Extern`] to an [`Extern<Engine>`].
fn extern_from(value: wasmtime::Extern) -> Extern<Engine> {
    match value {
        wasmtime::Extern::Func(x) => Extern::Func(Func::new(x)),
        wasmtime::Extern::Global(x) => Extern::Global(Global::new(x)),
        wasmtime::Extern::Memory(x) => Extern::Memory(Memory::new(x)),
        wasmtime::Extern::Table(x) => Extern::Table(Table::new(x)),
        wasmtime::Extern::Tag(_) => {
            unimplemented!("tags are not supported in the wasm_runtime_layer")
        }
        wasmtime::Extern::SharedMemory(_) => {
            unimplemented!("shared memories are not supported in the wasm_runtime_layer")
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
fn extern_type_from(ty: wasmtime::ExternType) -> ExternType {
    match ty {
        wasmtime::ExternType::Func(x) => ExternType::Func(func_type_from(x)),
        wasmtime::ExternType::Global(x) => ExternType::Global(global_type_from(x)),
        wasmtime::ExternType::Memory(x) => ExternType::Memory(memory_type_from(x)),
        wasmtime::ExternType::Table(x) => ExternType::Table(table_type_from(x)),
        wasmtime::ExternType::Tag(_) => {
            unimplemented!("tags are not supported in the wasm_runtime_layer")
        }
    }
}
