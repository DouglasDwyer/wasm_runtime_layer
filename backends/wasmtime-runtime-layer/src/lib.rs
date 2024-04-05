#![deny(warnings)]
#![warn(missing_docs)]
#![warn(clippy::missing_docs_in_private_items)]

//! `wasmtime-runtime-layer` implements the `wasm_runtime_layer` abstraction interface over WebAssembly runtimes for `Wasmtime`.

use std::{
    ops::{Deref, DerefMut},
    sync::Arc,
};

use anyhow::{Error, Result};
use fxhash::FxHashMap;
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
        #[derive($($derive),*)]
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

            #[must_use]
            #[allow(clippy::needless_lifetimes)]
            #[doc = concat!(
                "Convert a shared reference to a [`",
                stringify!($inner),
                "`] to a shared reference to a `",
                stringify!($newtype),
                "`."
            )]
            pub fn from_ref<'s>(inner: &'s $inner) -> &'s Self {
                // Safety: $newtype is a transparent newtype around $inner
                #[allow(unsafe_code)]
                unsafe {
                    &*(inner as *const $inner).cast()
                }
            }

            #[must_use]
            #[allow(clippy::needless_lifetimes)]
            #[doc = concat!(
                "Convert a mutable reference to a [`",
                stringify!($inner),
                "`] to a mutable reference to a `",
                stringify!($newtype),
                "`."
            )]
            pub fn from_mut<'s>(inner: &'s mut $inner) -> &'s mut Self {
                // Safety: $newtype is a transparent newtype around $inner
                #[allow(unsafe_code)]
                unsafe {
                    &mut *(inner as *mut $inner).cast()
                }
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
                $newtype::from_ref(self)
            }
        }

        impl$($tt)* AsMut<$newtype$($tt)*> for $inner {
            fn as_mut(&mut self) -> &mut $newtype$($tt)* {
                $newtype::from_mut(self)
            }
        }
    }
}

delegate! { #[derive(Clone, Default)] Engine(wasmtime::Engine) }
delegate! { #[derive(Clone)] ExternRef(wasmtime::ExternRef) }
delegate! { #[derive(Clone)] Func(wasmtime::Func) }
delegate! { #[derive(Clone)] Global(wasmtime::Global) }
delegate! { #[derive(Clone)] Memory(wasmtime::Memory) }
delegate! { #[derive(Clone)] Module(wasmtime::Module) }
delegate! { #[derive()] Store(wasmtime::Store<T>) <T> }
delegate! { #[derive()] StoreContext(wasmtime::StoreContext<'a, T>) <'a, T> }
delegate! { #[derive()] StoreContextMut(wasmtime::StoreContextMut<'a, T>) <'a, T> }
delegate! { #[derive(Clone)] Table(wasmtime::Table) }

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

impl WasmExternRef<Engine> for ExternRef {
    fn new<T: 'static + Send + Sync>(_: impl AsContextMut<Engine>, object: T) -> Self {
        Self::new(wasmtime::ExternRef::new(object))
    }

    fn downcast<'a, 's: 'a, T: 'static, S: 's>(&'a self, _: StoreContext<'s, S>) -> Result<&'a T> {
        self.data()
            .downcast_ref::<T>()
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
        Self::new(wasmtime::Func::new(
            ctx.as_context_mut().into_inner(),
            func_type_into(ty),
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

                for (i, result) in output.iter().enumerate() {
                    results[i] = value_into(result.clone());
                }

                std::result::Result::Ok(())
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

        for (i, result) in output.iter().enumerate() {
            results[i] = value_from(result.clone());
        }

        Ok(())
    }
}

impl WasmGlobal<Engine> for Global {
    fn new(mut ctx: impl AsContextMut<Engine>, value: Value<Engine>, mutable: bool) -> Self {
        let value = value_into(value);
        Self::new(
            wasmtime::Global::new(
                ctx.as_context_mut().into_inner(),
                wasmtime::GlobalType::new(
                    value.ty(),
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
            .map_err(Error::msg)
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
            .map_err(Error::msg)
    }

    fn ty(&self, ctx: impl AsContext<Engine>) -> MemoryType {
        memory_type_from(self.as_ref().ty(ctx.as_context().into_inner()))
    }

    fn grow(&self, mut ctx: impl AsContextMut<Engine>, additional: u32) -> Result<u32> {
        self.as_ref()
            .grow(ctx.as_context_mut().into_inner(), additional as u64)
            .map(|x| x as u32)
            .map_err(Error::msg)
    }

    fn current_pages(&self, ctx: impl AsContext<Engine>) -> u32 {
        self.as_ref().size(ctx.as_context().into_inner()) as u32
    }

    fn read(&self, ctx: impl AsContext<Engine>, offset: usize, buffer: &mut [u8]) -> Result<()> {
        self.as_ref()
            .read(ctx.as_context().into_inner(), offset, buffer)
            .map_err(Error::msg)
    }

    fn write(
        &self,
        mut ctx: impl AsContextMut<Engine>,
        offset: usize,
        buffer: &[u8],
    ) -> Result<()> {
        self.as_ref()
            .write(ctx.as_context_mut().into_inner(), offset, buffer)
            .map_err(Error::msg)
    }
}

impl WasmModule<Engine> for Module {
    fn new(engine: &Engine, mut stream: impl std::io::Read) -> Result<Self> {
        let mut buf = Vec::default();
        stream.read_to_end(&mut buf)?;
        wasmtime::Module::from_binary(engine, &buf).map(Self::new)
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

impl<T> WasmStore<T, Engine> for Store<T> {
    fn new(engine: &Engine, data: T) -> Self {
        Self::new(wasmtime::Store::new(engine, data))
    }

    fn engine(&self) -> &Engine {
        Engine::from_ref(self.as_ref().engine())
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

impl<'a, T> WasmStoreContext<'a, T, Engine> for StoreContext<'a, T> {
    fn engine(&self) -> &Engine {
        Engine::from_ref(self.as_ref().engine())
    }

    fn data(&self) -> &T {
        self.as_ref().data()
    }
}

impl<'a, T> AsContext<Engine> for StoreContext<'a, T> {
    type UserState = T;

    fn as_context(&self) -> StoreContext<T> {
        StoreContext::new(wasmtime::AsContext::as_context(self.as_ref()))
    }
}

impl<'a, T> AsContext<Engine> for StoreContextMut<'a, T> {
    type UserState = T;

    fn as_context(&self) -> StoreContext<T> {
        StoreContext::new(wasmtime::AsContext::as_context(self.as_ref()))
    }
}

impl<'a, T> AsContextMut<Engine> for StoreContextMut<'a, T> {
    fn as_context_mut(&mut self) -> StoreContextMut<T> {
        StoreContextMut::new(wasmtime::AsContextMut::as_context_mut(self.as_mut()))
    }
}

impl<'a, T> WasmStoreContext<'a, T, Engine> for StoreContextMut<'a, T> {
    fn engine(&self) -> &Engine {
        Engine::from_ref(self.as_ref().engine())
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
        wasmtime::Table::new(
            ctx.as_context_mut().into_inner(),
            table_type_into(ty),
            value_into(init),
        )
        .map(Self::new)
        .map_err(Error::msg)
    }

    fn ty(&self, ctx: impl AsContext<Engine>) -> TableType {
        table_type_from(self.as_ref().ty(ctx.as_context().into_inner()))
    }

    fn size(&self, ctx: impl AsContext<Engine>) -> u32 {
        self.as_ref().size(ctx.as_context().into_inner())
    }

    fn grow(
        &self,
        mut ctx: impl AsContextMut<Engine>,
        delta: u32,
        init: Value<Engine>,
    ) -> Result<u32> {
        self.as_ref()
            .grow(ctx.as_context_mut().into_inner(), delta, value_into(init))
            .map_err(Error::msg)
    }

    fn get(&self, mut ctx: impl AsContextMut<Engine>, index: u32) -> Option<Value<Engine>> {
        self.as_ref()
            .get(ctx.as_context_mut().into_inner(), index)
            .map(value_from)
    }

    fn set(
        &self,
        mut ctx: impl AsContextMut<Engine>,
        index: u32,
        value: Value<Engine>,
    ) -> Result<()> {
        self.as_ref()
            .set(ctx.as_context_mut().into_inner(), index, value_into(value))
            .map_err(Error::msg)
    }
}

impl<T> AsContext<Engine> for Store<T> {
    type UserState = T;

    fn as_context(&self) -> StoreContext<'_, Self::UserState> {
        StoreContext::new(wasmtime::AsContext::as_context(self.as_ref()))
    }
}

impl<T> AsContextMut<Engine> for Store<T> {
    fn as_context_mut(&mut self) -> StoreContextMut<'_, Self::UserState> {
        StoreContextMut::new(wasmtime::AsContextMut::as_context_mut(self.as_mut()))
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
        wasmtime::Val::V128(_) => unimplemented!(),
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
        wasmtime::ValType::FuncRef => ValueType::FuncRef,
        wasmtime::ValType::ExternRef => ValueType::ExternRef,
        wasmtime::ValType::V128 => unimplemented!(),
    }
}

/// Convert a [`ValueType`] to a [`wasmtime::ValType`].
fn value_type_into(ty: ValueType) -> wasmtime::ValType {
    match ty {
        ValueType::I32 => wasmtime::ValType::I32,
        ValueType::I64 => wasmtime::ValType::I64,
        ValueType::F32 => wasmtime::ValType::F32,
        ValueType::F64 => wasmtime::ValType::F64,
        ValueType::FuncRef => wasmtime::ValType::FuncRef,
        ValueType::ExternRef => wasmtime::ValType::ExternRef,
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
fn func_type_into(ty: FuncType) -> wasmtime::FuncType {
    wasmtime::FuncType::new(
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
    MemoryType::new(ty.minimum() as u32, ty.maximum().map(|x| x as u32))
}

/// Convert a [`MemoryType`] to a [`wasmtime::MemoryType`].
fn memory_type_into(ty: MemoryType) -> wasmtime::MemoryType {
    wasmtime::MemoryType::new(ty.initial_pages(), ty.maximum_pages())
}

/// Convert a [`wasmtime::TableType`] to a [`TableType`].
fn table_type_from(ty: wasmtime::TableType) -> TableType {
    TableType::new(value_type_from(ty.element()), ty.minimum(), ty.maximum())
}

/// Convert a [`TableType`] to a [`wasmtime::TableType`].
fn table_type_into(ty: TableType) -> wasmtime::TableType {
    wasmtime::TableType::new(value_type_into(ty.element()), ty.minimum(), ty.maximum())
}

/// Convert a [`wasmtime::Extern`] to an [`Extern<Engine>`].
fn extern_from(value: wasmtime::Extern) -> Extern<Engine> {
    match value {
        wasmtime::Extern::Func(x) => Extern::Func(Func::new(x)),
        wasmtime::Extern::Global(x) => Extern::Global(Global::new(x)),
        wasmtime::Extern::Memory(x) => Extern::Memory(Memory::new(x)),
        wasmtime::Extern::Table(x) => Extern::Table(Table::new(x)),
        wasmtime::Extern::SharedMemory(_) => unimplemented!(),
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
    }
}
