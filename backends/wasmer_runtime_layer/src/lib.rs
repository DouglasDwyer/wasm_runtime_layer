#![warn(missing_docs)]
#![warn(clippy::missing_docs_in_private_items)]
#![no_std]

//! `wasmer_runtime_layer` implements the `wasm_runtime_layer` abstraction interface over WebAssembly runtimes for `Wasmer`.

extern crate alloc;

use alloc::{borrow::ToOwned, boxed::Box, string::String, vec::Vec};
use core::{
    marker::PhantomData,
    ops::{Deref, DerefMut},
};

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
use wasmer::{AsStoreMut, AsStoreRef};

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

delegate! { #[derive(Clone, Default)] Engine(wasmer::Engine) }
delegate! { #[derive(Clone)] ExternRef(wasmer::ExternRef) }
delegate! { #[derive(Clone)] Func(wasmer::Function) }
delegate! { #[derive(Clone)] Global(wasmer::Global) }
delegate! { #[derive(Clone)] Instance(wasmer::Instance) }
delegate! { #[derive(Clone)] Memory(wasmer::Memory) }
delegate! { #[derive(Clone)] Table(wasmer::Table) }

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

mod sendsync {
    //! Helper module that provides the [`SendSync`] wrapper, which provides
    //! [`Send`] and [`Sync`] unconditionally but can only be constructed after
    //! proving that `T`: [`Send`] + [`Sync`]

    /// Wrapper that checks `T`: [`Send`] + [`Sync`] only during construction
    pub struct SendSync<T>(T);

    impl<T: Send + Sync> SendSync<T> {
        /// Prove that `T`: [`Send`] + [`Sync`]
        #[must_use]
        pub fn new(v: T) -> Self {
            Self(v)
        }
    }

    impl<T> SendSync<T> {
        /// Obtain a reference to the inner `T`
        pub fn as_ref(&self) -> &T {
            &self.0
        }
    }

    // Safety: SendSync can only be constructed iff T: Send + Sync
    unsafe impl<T> Send for SendSync<T> {}
    unsafe impl<T> Sync for SendSync<T> {}
}

impl WasmExternRef<Engine> for ExternRef {
    fn new<T: 'static + Send + Sync>(mut ctx: impl AsContextMut<Engine>, object: T) -> Self {
        Self::new(wasmer::ExternRef::new(
            ctx.as_context_mut().as_store_mut(),
            sendsync::SendSync::new(object),
        ))
    }

    fn downcast<'a, 's: 'a, T: 'static, S: 's>(
        &'a self,
        ctx: StoreContext<'s, S>,
    ) -> Result<&'a T> {
        let ctx: &wasmer::StoreRef<'s> = ctx.as_store_ref();
        // Safety:
        // - StoreRef<'s> is bounded by 's already
        // - the extended temporary lifetime is not exposed
        // - &'a T is bounded by 'a and 's outlives 'a
        let ctx: &'s wasmer::StoreRef<'s> = unsafe { &*core::ptr::from_ref(ctx) };

        let object: &'a sendsync::SendSync<T> = self
            .as_ref()
            .downcast(ctx)
            .ok_or_else(|| Error::msg("Incorrect extern ref type."))?;
        Ok(object.as_ref())
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
        let mut dummy_results = ArgumentVec::with_capacity(ty.results().len());
        dummy_results.extend(ty.results().iter().map(|ty| match ty {
            ValueType::I32 => Value::I32(0),
            ValueType::I64 => Value::I64(0),
            ValueType::F32 => Value::F32(0.0),
            ValueType::F64 => Value::F64(0.0),
            ValueType::FuncRef => Value::FuncRef(None),
            ValueType::ExternRef => Value::ExternRef(None),
        }));
        let ty = func_type_into(ty);
        let mut ctx: StoreContextMut<T> = ctx.as_context_mut();
        let env = ctx.env.clone();
        Self::new(wasmer::Function::new_with_env(
            ctx.as_store_mut(),
            &env,
            ty,
            move |mut env, args| {
                let mut input = ArgumentVec::with_capacity(args.len());
                input.extend(args.iter().cloned().map(value_from));

                let mut output = dummy_results.clone();

                func(
                    StoreContextMut {
                        env: env.as_ref(),
                        store: env.as_store_mut(),
                        data: PhantomData::<&mut T>,
                    },
                    &input,
                    &mut output,
                )
                .map_err(|err| wasmer::RuntimeError::user(err.into()))?;

                Ok(output.into_iter().map(value_into).collect())
            },
        ))
    }

    fn ty(&self, ctx: impl AsContext<Engine>) -> FuncType {
        func_type_from(self.as_ref().ty(ctx.as_context().as_store_ref()))
    }

    fn call<T>(
        &self,
        mut ctx: impl AsContextMut<Engine>,
        args: &[Value<Engine>],
        results: &mut [Value<Engine>],
    ) -> Result<()> {
        let mut input = ArgumentVec::with_capacity(args.len());
        input.extend(args.iter().cloned().map(value_into));

        let output = self
            .as_ref()
            .call(ctx.as_context_mut().as_store_mut(), &input[..])?;

        for (i, result) in output.into_vec().into_iter().enumerate() {
            results[i] = value_from(result);
        }

        Ok(())
    }
}

impl WasmGlobal<Engine> for Global {
    fn new(mut ctx: impl AsContextMut<Engine>, value: Value<Engine>, mutable: bool) -> Self {
        if mutable {
            Self::new(wasmer::Global::new_mut(
                ctx.as_context_mut().as_store_mut(),
                value_into(value),
            ))
        } else {
            Self::new(wasmer::Global::new(
                ctx.as_context_mut().as_store_mut(),
                value_into(value),
            ))
        }
    }

    fn ty(&self, ctx: impl AsContext<Engine>) -> GlobalType {
        global_type_from(self.as_ref().ty(ctx.as_context().as_store_ref()))
    }

    fn set(&self, mut ctx: impl AsContextMut<Engine>, new_value: Value<Engine>) -> Result<()> {
        self.as_ref()
            .set(ctx.as_context_mut().as_store_mut(), value_into(new_value))
            .map_err(Error::new)
    }

    fn get(&self, mut ctx: impl AsContextMut<Engine>) -> Value<Engine> {
        value_from(self.as_ref().get(ctx.as_context_mut().as_store_mut()))
    }
}

impl WasmInstance<Engine> for Instance {
    fn new(
        mut store: impl AsContextMut<Engine>,
        module: &Module,
        imports: &Imports<Engine>,
    ) -> Result<Self> {
        let mut wimports = wasmer::Imports::new();

        for ((module, name), imp) in imports {
            wimports.define(&module, &name, extern_into(imp));
        }

        wasmer::Instance::new(
            store.as_context_mut().as_store_mut(),
            &module.module,
            &wimports,
        )
        .map(Self::new)
        .map_err(Error::new)
    }

    fn exports(&self, _: impl AsContext<Engine>) -> Box<dyn Iterator<Item = Export<Engine>>> {
        Box::new(
            self.as_ref()
                .exports
                .iter()
                .map(|(n, e)| Export {
                    name: n.clone(),
                    value: extern_from(e.clone()),
                })
                .collect::<Vec<_>>()
                .into_iter(),
        )
    }

    fn get_export(&self, _: impl AsContext<Engine>, name: &str) -> Option<Extern<Engine>> {
        self.as_ref()
            .exports
            .get_extern(name)
            .map(|e| extern_from(e.clone()))
    }
}

impl WasmMemory<Engine> for Memory {
    fn new(mut ctx: impl AsContextMut<Engine>, ty: MemoryType) -> Result<Self> {
        wasmer::Memory::new(ctx.as_context_mut().as_store_mut(), memory_type_into(ty))
            .map(Self::new)
            .map_err(Error::new)
    }

    fn ty(&self, ctx: impl AsContext<Engine>) -> MemoryType {
        memory_type_from(self.as_ref().ty(ctx.as_context().as_store_ref()))
    }

    fn grow(&self, mut ctx: impl AsContextMut<Engine>, additional: u32) -> Result<u32> {
        self.as_ref()
            .grow(
                ctx.as_context_mut().as_store_mut(),
                wasmer::Pages(additional),
            )
            .map(|x| x.0)
            .map_err(Error::new)
    }

    fn current_pages(&self, ctx: impl AsContext<Engine>) -> u32 {
        self.as_ref().view(ctx.as_context().as_store_ref()).size().0
    }

    fn read(&self, ctx: impl AsContext<Engine>, offset: usize, buffer: &mut [u8]) -> Result<()> {
        self.as_ref()
            .view(ctx.as_context().as_store_ref())
            .read(offset as u64, buffer)
            .map_err(Error::new)
    }

    fn write(
        &self,
        mut ctx: impl AsContextMut<Engine>,
        offset: usize,
        buffer: &[u8],
    ) -> Result<()> {
        self.as_ref()
            .view(ctx.as_context_mut().as_store_mut())
            .write(offset as u64, buffer)
            .map_err(Error::new)
    }
}

#[derive(Clone)]
/// Wrapper around [`wasmer::Module`].
pub struct Module {
    /// The wasmer module
    module: wasmer::Module,
    /// The pre-computed imports of the module
    imports: Vec<wasmer::ImportType>,
    /// The pre-computed exports of the module
    exports: FxHashMap<String, wasmer::ExportType>,
}

impl WasmModule<Engine> for Module {
    fn new(engine: &Engine, bytes: &[u8]) -> Result<Self> {
        let module = wasmer::Module::from_binary(engine, bytes)?;
        let imports = module.imports().collect();
        let exports = module.exports().map(|e| (e.name().to_owned(), e)).collect();
        Ok(Self {
            module,
            imports,
            exports,
        })
    }

    fn exports(&self) -> Box<dyn '_ + Iterator<Item = ExportType<'_>>> {
        Box::new(self.exports.values().map(|x| ExportType {
            name: x.name(),
            ty: extern_type_from(x.ty().clone()),
        }))
    }

    fn get_export(&self, name: &str) -> Option<ExternType> {
        self.exports
            .get(name)
            .map(|e| extern_type_from(e.ty().clone()))
    }

    fn imports(&self) -> Box<dyn '_ + Iterator<Item = ImportType<'_>>> {
        Box::new(self.imports.iter().map(|x| ImportType {
            module: x.module(),
            name: x.name(),
            ty: extern_type_from(x.ty().clone()),
        }))
    }
}

#[derive(Copy, Clone)]
/// A type-erased handle to store data
struct DataHandle(*mut ());

// Safety:
// - DataHandle is a private implementation detail
// - Store, StoreContext, and StoreContextMut still inherit Send from T
// - access to the contained data is guarded by access to the Store[Context]
// - if T: !Send then Store: !Send and StoreContextMut: !Send -> all good
// - if T: !Sync then StoreContext: !Send -> all good
unsafe impl Send for DataHandle {}

/// Wrapper around [`wasmer::Store`] that owns its data `T`.
pub struct Store<T: 'static> {
    /// The wasmer store
    store: wasmer::Store,
    /// The function environment that holds the handle to the data
    env: wasmer::FunctionEnv<DataHandle>,
    /// The owned data
    data: Box<T>,
}

impl<T: 'static> WasmStore<T, Engine> for Store<T> {
    fn new(engine: &Engine, data: T) -> Self {
        let mut store = wasmer::Store::new(engine.clone());
        let mut data = Box::new(data);
        let env = wasmer::FunctionEnv::new(
            &mut store,
            DataHandle(core::ptr::addr_of_mut!(*data).cast::<()>()),
        );

        Self { store, env, data }
    }

    fn engine(&self) -> &Engine {
        Engine::ref_cast(self.store.engine())
    }

    fn data(&self) -> &T {
        &self.data
    }

    fn data_mut(&mut self) -> &mut T {
        &mut self.data
    }

    fn into_data(self) -> T {
        *self.data
    }
}

impl<T: 'static> AsContext<Engine> for Store<T> {
    type UserState = T;

    fn as_context(&self) -> StoreContext<'_, Self::UserState> {
        StoreContext {
            store: self.store.as_store_ref(),
            env: self.env.clone(),
            data: PhantomData::<&T>,
        }
    }
}

impl<T: 'static> AsContextMut<Engine> for Store<T> {
    fn as_context_mut(&mut self) -> StoreContextMut<'_, Self::UserState> {
        StoreContextMut {
            store: self.store.as_store_mut(),
            env: self.env.clone(),
            data: PhantomData::<&mut T>,
        }
    }
}

/// Wrapper around [`wasmer::StoreRef`] that references the data `T`.
pub struct StoreContext<'a, T: 'static> {
    /// The reference to the store
    store: wasmer::StoreRef<'a>,
    /// The function environment that holds the handle to the data
    env: wasmer::FunctionEnv<DataHandle>,
    /// The marker for the data's type
    data: PhantomData<&'a T>,
}

impl<'a, T: 'static> StoreContext<'a, T> {
    /// Returns a reference to a [`wasmer::StoreRef`]
    fn as_store_ref(&self) -> &wasmer::StoreRef<'a> {
        &self.store
    }
}

impl<'a, T: 'static> WasmStoreContext<'a, T, Engine> for StoreContext<'a, T> {
    fn engine(&self) -> &Engine {
        Engine::ref_cast(self.store.engine())
    }

    fn data(&self) -> &T {
        let handle = self.env.as_ref(&self.store);
        // Safety: the returned reference borrows the store
        unsafe { &*handle.0.cast::<T>() }
    }
}

impl<T: 'static> AsContext<Engine> for StoreContext<'_, T> {
    type UserState = T;

    fn as_context(&self) -> StoreContext<'_, T> {
        StoreContext {
            store: self.store.as_store_ref(),
            env: self.env.clone(),
            data: self.data,
        }
    }
}

/// Wrapper around [`wasmer::StoreMut`] that references the data `T`.
pub struct StoreContextMut<'a, T: 'static> {
    /// The mutable reference to the store
    store: wasmer::StoreMut<'a>,
    /// The function environment that holds the handle to the data
    env: wasmer::FunctionEnv<DataHandle>,
    /// The marker for the data's type
    data: PhantomData<&'a mut T>,
}

impl<'a, T: 'static> StoreContextMut<'a, T> {
    /// Returns a mutable reference to a [`wasmer::StoreMut`]
    fn as_store_mut(&mut self) -> &mut wasmer::StoreMut<'a> {
        &mut self.store
    }
}

impl<T: 'static> AsContext<Engine> for StoreContextMut<'_, T> {
    type UserState = T;

    fn as_context(&self) -> StoreContext<'_, T> {
        StoreContext {
            store: self.store.as_store_ref(),
            env: self.env.clone(),
            data: PhantomData::<&T>,
        }
    }
}

impl<T: 'static> AsContextMut<Engine> for StoreContextMut<'_, T> {
    fn as_context_mut(&mut self) -> StoreContextMut<'_, T> {
        StoreContextMut {
            store: self.store.as_store_mut(),
            env: self.env.clone(),
            data: self.data,
        }
    }
}

impl<'a, T: 'static> WasmStoreContext<'a, T, Engine> for StoreContextMut<'a, T> {
    fn engine(&self) -> &Engine {
        Engine::ref_cast(self.store.engine())
    }

    fn data(&self) -> &T {
        let handle = self.env.as_ref(&self.store);
        // Safety: the returned reference borrows the store
        unsafe { &*handle.0.cast::<T>() }
    }
}

impl<'a, T: 'static> WasmStoreContextMut<'a, T, Engine> for StoreContextMut<'a, T> {
    fn data_mut(&mut self) -> &mut T {
        let handle = self.env.as_mut(&mut self.store);
        // Safety: the returned reference borrows the store
        unsafe { &mut *handle.0.cast::<T>() }
    }
}

impl WasmTable<Engine> for Table {
    fn new(mut ctx: impl AsContextMut<Engine>, ty: TableType, init: Value<Engine>) -> Result<Self> {
        wasmer::Table::new(
            ctx.as_context_mut().as_store_mut(),
            table_type_into(ty),
            value_into(init),
        )
        .map(Self::new)
        .map_err(Error::new)
    }

    fn ty(&self, ctx: impl AsContext<Engine>) -> TableType {
        table_type_from(self.as_ref().ty(ctx.as_context().as_store_ref()))
    }

    fn size(&self, ctx: impl AsContext<Engine>) -> u32 {
        self.as_ref().size(ctx.as_context().as_store_ref())
    }

    fn grow(
        &self,
        mut ctx: impl AsContextMut<Engine>,
        delta: u32,
        init: Value<Engine>,
    ) -> Result<u32> {
        self.as_ref()
            .grow(ctx.as_context_mut().as_store_mut(), delta, value_into(init))
            .map_err(Error::new)
    }

    fn get(&self, mut ctx: impl AsContextMut<Engine>, index: u32) -> Option<Value<Engine>> {
        self.as_ref()
            .get(ctx.as_context_mut().as_store_mut(), index)
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
                ctx.as_context_mut().as_store_mut(),
                index,
                value_into(value),
            )
            .map_err(Error::new)
    }
}

/// Convert a [`wasmer::Value`] to a [`Value<Engine>`].
fn value_from(value: wasmer::Value) -> Value<Engine> {
    match value {
        wasmer::Value::I32(x) => Value::I32(x),
        wasmer::Value::I64(x) => Value::I64(x),
        wasmer::Value::F32(x) => Value::F32(x),
        wasmer::Value::F64(x) => Value::F64(x),
        wasmer::Value::FuncRef(x) => Value::FuncRef(x.map(Func::new)),
        wasmer::Value::ExternRef(x) => Value::ExternRef(x.map(ExternRef::new)),
        wasmer::Value::V128(_) => unimplemented!("v128 is not supported in the wasm_runtime_layer"),
        wasmer::Value::ExceptionRef(_) => {
            unimplemented!("exceptions are not supported in the wasm_runtime_layer")
        }
    }
}

/// Convert a [`Value<Engine>`] to a [`wasmer::Value`].
fn value_into(value: Value<Engine>) -> wasmer::Value {
    match value {
        Value::I32(x) => wasmer::Value::I32(x),
        Value::I64(x) => wasmer::Value::I64(x),
        Value::F32(x) => wasmer::Value::F32(x),
        Value::F64(x) => wasmer::Value::F64(x),
        Value::FuncRef(x) => wasmer::Value::FuncRef(x.map(Func::into_inner)),
        Value::ExternRef(x) => wasmer::Value::ExternRef(x.map(ExternRef::into_inner)),
    }
}

/// Convert a [`wasmer::Type`] to a [`ValueType`].
fn value_type_from(ty: wasmer::Type) -> ValueType {
    match ty {
        wasmer::Type::I32 => ValueType::I32,
        wasmer::Type::I64 => ValueType::I64,
        wasmer::Type::F32 => ValueType::F32,
        wasmer::Type::F64 => ValueType::F64,
        wasmer::Type::V128 => unimplemented!("v128 is not supported in the wasm_runtime_layer"),
        wasmer::Type::ExternRef => ValueType::ExternRef,
        wasmer::Type::FuncRef => ValueType::FuncRef,
        wasmer::Type::ExceptionRef => {
            unimplemented!("exceptions are not supported in the wasm_runtime_layer")
        }
    }
}

/// Convert a [`ValueType`] to a [`wasmer::Type`].
fn value_type_into(ty: ValueType) -> wasmer::Type {
    match ty {
        ValueType::I32 => wasmer::Type::I32,
        ValueType::I64 => wasmer::Type::I64,
        ValueType::F32 => wasmer::Type::F32,
        ValueType::F64 => wasmer::Type::F64,
        ValueType::FuncRef => wasmer::Type::FuncRef,
        ValueType::ExternRef => wasmer::Type::ExternRef,
    }
}

/// Convert a [`wasmer::FunctionType`] to a [`FuncType`].
fn func_type_from(ty: wasmer::FunctionType) -> FuncType {
    FuncType::new(
        ty.params().iter().copied().map(value_type_from),
        ty.results().iter().copied().map(value_type_from),
    )
}

/// Convert a [`FuncType`] to a [`wasmer::FunctionType`].
fn func_type_into(ty: FuncType) -> wasmer::FunctionType {
    wasmer::FunctionType::new(
        ty.params()
            .iter()
            .map(|&x| value_type_into(x))
            .collect::<Vec<_>>()
            .into_boxed_slice(),
        ty.results()
            .iter()
            .map(|&x| value_type_into(x))
            .collect::<Vec<_>>()
            .into_boxed_slice(),
    )
}

/// Convert a [`wasmer::GlobalType`] to a [`GlobalType`].
fn global_type_from(ty: wasmer::GlobalType) -> GlobalType {
    GlobalType::new(
        value_type_from(ty.ty),
        matches!(ty.mutability, wasmer::Mutability::Var),
    )
}

/// Convert a [`wasmer::MemoryType`] to a [`MemoryType`].
fn memory_type_from(ty: wasmer::MemoryType) -> MemoryType {
    MemoryType::new(ty.minimum.0, ty.maximum.map(|x| x.0))
}

/// Convert a [`MemoryType`] to a [`wasmer::MemoryType`].
fn memory_type_into(ty: MemoryType) -> wasmer::MemoryType {
    wasmer::MemoryType::new(ty.initial_pages(), ty.maximum_pages(), false)
}

/// Convert a [`wasmer::TableType`] to a [`TableType`].
fn table_type_from(ty: wasmer::TableType) -> TableType {
    TableType::new(value_type_from(ty.ty), ty.minimum, ty.maximum)
}

/// Convert a [`TableType`] to a [`wasmer::TableType`].
fn table_type_into(ty: TableType) -> wasmer::TableType {
    wasmer::TableType::new(value_type_into(ty.element()), ty.minimum(), ty.maximum())
}

/// Convert a [`wasmer::Extern`] to an [`Extern<Engine>`].
fn extern_from(value: wasmer::Extern) -> Extern<Engine> {
    match value {
        wasmer::Extern::Function(x) => Extern::Func(Func::new(x)),
        wasmer::Extern::Global(x) => Extern::Global(Global::new(x)),
        wasmer::Extern::Memory(x) => Extern::Memory(Memory::new(x)),
        wasmer::Extern::Table(x) => Extern::Table(Table::new(x)),
        wasmer::Extern::Tag(_) => {
            unimplemented!("tags are not supported in the wasm_runtime_layer")
        }
    }
}

/// Convert an [`Extern<Engine>`] to a [`wasmer::Extern`].
fn extern_into(value: Extern<Engine>) -> wasmer::Extern {
    match value {
        Extern::Func(x) => wasmer::Extern::Function(x.into_inner()),
        Extern::Global(x) => wasmer::Extern::Global(x.into_inner()),
        Extern::Memory(x) => wasmer::Extern::Memory(x.into_inner()),
        Extern::Table(x) => wasmer::Extern::Table(x.into_inner()),
    }
}

/// Convert a [`wasmer::ExternType`] to an [`ExternType`].
fn extern_type_from(ty: wasmer::ExternType) -> ExternType {
    match ty {
        wasmer::ExternType::Function(x) => ExternType::Func(func_type_from(x)),
        wasmer::ExternType::Global(x) => ExternType::Global(global_type_from(x)),
        wasmer::ExternType::Memory(x) => ExternType::Memory(memory_type_from(x)),
        wasmer::ExternType::Table(x) => ExternType::Table(table_type_from(x)),
        wasmer::ExternType::Tag(_) => {
            unimplemented!("tags are not supported in the wasm_runtime_layer")
        }
    }
}
