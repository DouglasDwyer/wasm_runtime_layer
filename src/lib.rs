#![deny(warnings)]
#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::missing_docs_in_private_items)]

//! Craate

/// Provides traits for implementing runtime backends.
pub mod backend;

use crate::backend::*;
use anyhow::*;
use ref_cast::*;
use smallvec::*;
use std::any::*;
use std::collections::*;
use std::marker::*;
use std::sync::*;

/// The default amount of arguments and return values for which to allocate
/// stack space.
const DEFAULT_ARGUMENT_SIZE: usize = 4;

/// A vector which allocates up to the default number of arguments on the stack.
type ArgumentVec<T> = SmallVec<[T; DEFAULT_ARGUMENT_SIZE]>;

/// Type of a value.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ValueType {
    /// 32-bit signed or unsigned integer.
    I32,
    /// 64-bit signed or unsigned integer.
    I64,
    /// 32-bit floating point number.
    F32,
    /// 64-bit floating point number.
    F64,
    /// An optional function reference.
    FuncRef,
    /// An optional external reference.
    ExternRef,
}

/// The type of a global variable.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct GlobalType {
    /// The value type of the global variable.
    content: ValueType,
    /// The mutability of the global variable.
    mutable: bool,
}

impl GlobalType {
    /// Creates a new [`GlobalType`] from the given [`ValueType`] and [`Mutability`].
    pub fn new(content: ValueType, mutable: bool) -> Self {
        Self { content, mutable }
    }

    /// Returns the [`ValueType`] of the global variable.
    pub fn content(&self) -> ValueType {
        self.content
    }

    /// Returns whether the global variable is mutable.
    pub fn mutable(&self) -> bool {
        self.mutable
    }
}

/// A descriptor for a [`Table`] instance.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct TableType {
    /// The type of values stored in the [`Table`].
    element: ValueType,
    /// The minimum number of elements the [`Table`] must have.
    min: u32,
    /// The optional maximum number of elements the [`Table`] can have.
    ///
    /// If this is `None` then the [`Table`] is not limited in size.
    max: Option<u32>,
}

impl TableType {
    /// Creates a new [`TableType`].
    ///
    /// # Panics
    ///
    /// If `min` is greater than `max`.
    pub fn new(element: ValueType, min: u32, max: Option<u32>) -> Self {
        if let Some(max) = max {
            assert!(min <= max);
        }
        Self { element, min, max }
    }

    /// Returns the [`ValueType`] of elements stored in the [`Table`].
    pub fn element(&self) -> ValueType {
        self.element
    }

    /// Returns minimum number of elements the [`Table`] must have.
    pub fn minimum(&self) -> u32 {
        self.min
    }

    /// The optional maximum number of elements the [`Table`] can have.
    ///
    /// If this returns `None` then the [`Table`] is not limited in size.
    pub fn maximum(&self) -> Option<u32> {
        self.max
    }
}

/// The memory type of a linear memory.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct MemoryType {
    /// The initial amount of pages that a memory should have.
    initial: u32,
    /// The maximum amount of pages to which a memory may grow.
    maximum: Option<u32>,
}

impl MemoryType {
    /// Creates a new memory type with initial and optional maximum pages.
    pub fn new(initial: u32, maximum: Option<u32>) -> Self {
        Self { initial, maximum }
    }

    /// Returns the initial pages of the memory type.
    pub fn initial_pages(self) -> u32 {
        self.initial
    }

    /// Returns the maximum pages of the memory type.
    ///
    /// # Note
    ///
    /// - Returns `None` if there is no limit set.
    /// - Maximum memory size cannot exceed `65536` pages or 4GiB.
    pub fn maximum_pages(self) -> Option<u32> {
        self.maximum
    }
}

/// A function type representing a function's parameter and result types.
///
/// # Note
///
/// Can be cloned cheaply.
#[derive(Clone, PartialEq, Eq)]
pub struct FuncType {
    /// The number of function parameters.
    len_params: usize,
    /// The ordered and merged parameter and result types of the function type.
    ///
    /// # Note
    ///
    /// The parameters and results are ordered and merged in a single
    /// vector starting with parameters in their order and followed
    /// by results in their order.
    /// The `len_params` field denotes how many parameters there are in
    /// the head of the vector before the results.
    params_results: Arc<[ValueType]>,
}

impl std::fmt::Debug for FuncType {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_struct("FuncType")
            .field("params", &self.params())
            .field("results", &self.results())
            .finish()
    }
}

impl FuncType {
    /// Creates a new [`FuncType`].
    pub fn new<P, R>(params: P, results: R) -> Self
    where
        P: IntoIterator<Item = ValueType>,
        R: IntoIterator<Item = ValueType>,
    {
        let mut params_results = params.into_iter().collect::<Vec<_>>();
        let len_params = params_results.len();
        params_results.extend(results);
        Self {
            params_results: params_results.into(),
            len_params,
        }
    }

    /// Returns the parameter types of the function type.
    pub fn params(&self) -> &[ValueType] {
        &self.params_results[..self.len_params]
    }

    /// Returns the result types of the function type.
    pub fn results(&self) -> &[ValueType] {
        &self.params_results[self.len_params..]
    }
}

/// The type of an [`Extern`] item.
///
/// A list of all possible types which can be externally referenced from a WebAssembly module.
#[derive(Clone, Debug)]
pub enum ExternType {
    /// The type of an [`Extern::Global`].
    Global(GlobalType),
    /// The type of an [`Extern::Table`].
    Table(TableType),
    /// The type of an [`Extern::Memory`].
    Memory(MemoryType),
    /// The type of an [`Extern::Func`].
    Func(FuncType),
}

impl From<GlobalType> for ExternType {
    fn from(global: GlobalType) -> Self {
        Self::Global(global)
    }
}

impl From<TableType> for ExternType {
    fn from(table: TableType) -> Self {
        Self::Table(table)
    }
}

impl From<MemoryType> for ExternType {
    fn from(memory: MemoryType) -> Self {
        Self::Memory(memory)
    }
}

impl From<FuncType> for ExternType {
    fn from(func: FuncType) -> Self {
        Self::Func(func)
    }
}

impl ExternType {
    /// Returns the underlying [`GlobalType`] or `None` if it is of a different type.
    pub fn global(&self) -> Option<&GlobalType> {
        match self {
            Self::Global(ty) => Some(ty),
            _ => None,
        }
    }

    /// Returns the underlying [`TableType`] or `None` if it is of a different type.
    pub fn table(&self) -> Option<&TableType> {
        match self {
            Self::Table(ty) => Some(ty),
            _ => None,
        }
    }

    /// Returns the underlying [`MemoryType`] or `None` if it is of a different type.
    pub fn memory(&self) -> Option<&MemoryType> {
        match self {
            Self::Memory(ty) => Some(ty),
            _ => None,
        }
    }

    /// Returns the underlying [`FuncType`] or `None` if it is of a different type.
    pub fn func(&self) -> Option<&FuncType> {
        match self {
            Self::Func(ty) => Some(ty),
            _ => None,
        }
    }
}

/// A descriptor for an exported WebAssembly value of a [`Module`].
///
/// This type is primarily accessed from the [`Module::exports`] method and describes
/// what names are exported from a Wasm [`Module`] and the type of the item that is exported.
#[derive(Clone, Debug)]
pub struct ExportType<'module> {
    /// The name by which the export is known.
    pub name: &'module str,
    /// The type of the exported item.
    pub ty: ExternType,
}

/// A descriptor for an imported value into a Wasm [`Module`].
///
/// This type is primarily accessed from the [`Module::imports`] method.
/// Each [`ImportType`] describes an import into the Wasm module with the `module/name`
/// that it is imported from as well as the type of item that is being imported.
#[derive(Clone, Debug)]
pub struct ImportType<'module> {
    /// The module import name.
    pub module: &'module str,
    /// The name of the imported item.
    pub name: &'module str,
    /// The external item type.
    pub ty: ExternType,
}

/// An external item to a WebAssembly module.
///
/// This is returned from [`Instance::exports`](crate::Instance::exports)
/// or [`Instance::get_export`](crate::Instance::get_export).
#[derive(Clone, Debug)]
pub enum Extern {
    /// A WebAssembly global which acts like a [`Cell<T>`] of sorts, supporting `get` and `set` operations.
    ///
    /// [`Cell<T>`]: https://doc.rust-lang.org/core/cell/struct.Cell.html
    Global(Global),
    /// A WebAssembly table which is an array of funtion references.
    Table(Table),
    /// A WebAssembly linear memory.
    Memory(Memory),
    /// A WebAssembly function which can be called.
    Func(Func),
}

impl<E: WasmEngine> From<&crate::backend::Extern<E>> for Extern {
    fn from(value: &crate::backend::Extern<E>) -> Self {
        match value {
            crate::backend::Extern::Global(x) => Self::Global(Global {
                global: BackendObject::new(x.clone()),
            }),
            crate::backend::Extern::Table(x) => Self::Table(Table {
                table: BackendObject::new(x.clone()),
            }),
            crate::backend::Extern::Memory(x) => Self::Memory(Memory {
                memory: BackendObject::new(x.clone()),
            }),
            crate::backend::Extern::Func(x) => Self::Func(Func {
                func: BackendObject::new(x.clone()),
            }),
        }
    }
}

impl<E: WasmEngine> From<&Extern> for crate::backend::Extern<E> {
    fn from(value: &Extern) -> Self {
        match value {
            Extern::Global(x) => Self::Global(x.global.cast::<E::Global>().clone()),
            Extern::Table(x) => Self::Table(x.table.cast::<E::Table>().clone()),
            Extern::Memory(x) => Self::Memory(x.memory.cast::<E::Memory>().clone()),
            Extern::Func(x) => Self::Func(x.func.cast::<E::Func>().clone()),
        }
    }
}

/// A descriptor for an exported WebAssembly value of an [`Instance`].
///
/// This type is primarily accessed from the [`Instance::exports`] method and describes
/// what names are exported from a Wasm [`Instance`] and the type of the item that is exported.
pub struct Export {
    /// The name by which the export is known.
    pub name: String,
    /// The value of the exported item.
    pub value: Extern,
}

impl<E: WasmEngine> From<crate::backend::Export<E>> for Export {
    fn from(value: crate::backend::Export<E>) -> Self {
        Self {
            name: value.name,
            value: (&value.value).into(),
        }
    }
}

/// All of the import data used when instantiating.
#[derive(Clone, Debug)]
pub struct Imports {
    /// The mapping from names to externals.
    pub(crate) map: HashMap<(String, String), Extern>,
}

impl Imports {
    /// Create a new `Imports`.
    pub fn new() -> Self {
        Self {
            map: HashMap::default(),
        }
    }

    /// Gets an export given a module and a name
    pub fn get_export(&self, module: &str, name: &str) -> Option<Extern> {
        if self.exists(module, name) {
            let ext = &self.map[&(module.to_string(), name.to_string())];
            return Some(ext.clone());
        }
        None
    }

    /// Returns if an export exist for a given module and name.
    pub fn exists(&self, module: &str, name: &str) -> bool {
        self.map
            .contains_key(&(module.to_string(), name.to_string()))
    }

    /// Returns true if the Imports contains namespace with the provided name.
    pub fn contains_namespace(&self, name: &str) -> bool {
        self.map.keys().any(|(k, _)| (k == name))
    }

    /// Register a list of externs into a namespace.
    pub fn register_namespace(
        &mut self,
        ns: &str,
        contents: impl IntoIterator<Item = (String, Extern)>,
    ) {
        for (name, extern_) in contents.into_iter() {
            self.map.insert((ns.to_string(), name.clone()), extern_);
        }
    }

    /// Add a single import with a namespace `ns` and name `name`.
    pub fn define(&mut self, ns: &str, name: &str, val: impl Into<Extern>) {
        self.map
            .insert((ns.to_string(), name.to_string()), val.into());
    }

    /// Iterates through all the imports in this structure
    pub fn iter(&self) -> ImportsIterator {
        ImportsIterator::new(self)
    }
}

/// An iterator over imports.
pub struct ImportsIterator<'a> {
    /// The inner iterator over external items.
    iter: std::collections::hash_map::Iter<'a, (String, String), Extern>,
}

impl<'a> ImportsIterator<'a> {
    /// Creates a new imports iterator.
    fn new(imports: &'a Imports) -> Self {
        let iter = imports.map.iter();
        Self { iter }
    }
}

impl<'a> Iterator for ImportsIterator<'a> {
    type Item = (&'a str, &'a str, &'a Extern);

    fn next(&mut self) -> Option<Self::Item> {
        self.iter
            .next()
            .map(|(k, v)| (k.0.as_str(), k.1.as_str(), v))
    }
}

impl IntoIterator for &Imports {
    type IntoIter = std::collections::hash_map::IntoIter<(String, String), Extern>;
    type Item = ((String, String), Extern);

    fn into_iter(self) -> Self::IntoIter {
        self.map.clone().into_iter()
    }
}

impl Default for Imports {
    fn default() -> Self {
        Self::new()
    }
}

impl Extend<((String, String), Extern)> for Imports {
    fn extend<T: IntoIterator<Item = ((String, String), Extern)>>(&mut self, iter: T) {
        for ((ns, name), ext) in iter.into_iter() {
            self.define(&ns, &name, ext);
        }
    }
}

/// The backing engine for a WebAssembly runtime.
#[derive(RefCast, Clone)]
#[repr(transparent)]
pub struct Engine<E: WasmEngine> {
    /// The engine backend.
    backend: E,
}

impl<E: WasmEngine> Engine<E> {
    /// Creates a new engine using the specified backend.
    pub fn new(backend: E) -> Self {
        Self { backend }
    }

    /// Unwraps this instance into the core backend engine.
    pub fn into_backend(self) -> E {
        self.backend
    }
}

/// The store represents all global state that can be manipulated by
/// WebAssembly programs. It consists of the runtime representation
/// of all instances of functions, tables, memories, and globals that
/// have been allocated during the lifetime of the abstract machine.
///
/// The `Store` holds the engine (that is —amongst many things— used to compile
/// the Wasm bytes into a valid module artifact).
///
/// Spec: <https://webassembly.github.io/spec/core/exec/runtime.html#store>
pub struct Store<T, E: WasmEngine> {
    /// The backing implementation.
    inner: E::Store<T>,
}

impl<T, E: WasmEngine> Store<T, E> {
    /// Creates a new [`Store`] with a specific [`Engine`].
    pub fn new(engine: &Engine<E>, data: T) -> Self {
        Self {
            inner: <E::Store<T> as WasmStore<T, E>>::new(&engine.backend, data),
        }
    }

    /// Returns the [`Engine`] that this store is associated with.
    pub fn engine(&self) -> &Engine<E> {
        Engine::<E>::ref_cast(self.inner.engine())
    }

    /// Returns a shared reference to the user provided data owned by this [`Store`].
    pub fn data(&self) -> &T {
        self.inner.data()
    }

    /// Returns an exclusive reference to the user provided data owned by this [`Store`].
    pub fn data_mut(&mut self) -> &mut T {
        self.inner.data_mut()
    }

    /// Consumes `self` and returns its user provided data.
    pub fn into_data(self) -> T {
        self.inner.into_data()
    }
}

/// A temporary handle to a [`&Store<T>`][`Store`].
///
/// This type is suitable for [`AsContext`] trait bounds on methods if desired.
/// For more information, see [`Store`].
pub struct StoreContext<'a, T: 'a, E: WasmEngine> {
    /// The backing implementation.
    inner: E::StoreContext<'a, T>,
}

impl<'a, T: 'a, E: WasmEngine> StoreContext<'a, T, E> {
    /// Returns the underlying [`Engine`] this store is connected to.
    pub fn engine(&self) -> &Engine<E> {
        Engine::<E>::ref_cast(self.inner.engine())
    }

    /// Access the underlying data owned by this store.
    ///
    /// Same as [`Store::data`].
    pub fn data(&self) -> &T {
        self.inner.data()
    }
}

/// A temporary handle to a [`&mut Store<T>`][`Store`].
///
/// This type is suitable for [`AsContextMut`] or [`AsContext`] trait bounds on methods if desired.
/// For more information, see [`Store`].
pub struct StoreContextMut<'a, T: 'a, E: WasmEngine> {
    /// The backing implementation.
    inner: E::StoreContextMut<'a, T>,
}

impl<'a, T: 'a, E: WasmEngine> StoreContextMut<'a, T, E> {
    /// Returns the underlying [`Engine`] this store is connected to.
    pub fn engine(&self) -> &Engine<E> {
        Engine::<E>::ref_cast(self.inner.engine())
    }

    /// Access the underlying data owned by this store.
    ///
    /// Same as [`Store::data`].    
    pub fn data(&self) -> &T {
        self.inner.data()
    }

    /// Access the underlying data owned by this store.
    ///
    /// Same as [`Store::data_mut`].
    pub fn data_mut(&mut self) -> &mut T {
        self.inner.data_mut()
    }
}

/// Runtime representation of a value.
///
/// Wasm code manipulate values of the four basic value types:
/// integers and floating-point data of 32 or 64 bit width each, respectively.
///
/// There is no distinction between signed and unsigned integer types. Instead, integers are
/// interpreted by respective operations as either unsigned or signed in two’s complement representation.
#[derive(Clone, Debug)]
pub enum Value {
    /// Value of 32-bit signed or unsigned integer.
    I32(i32),
    /// Value of 64-bit signed or unsigned integer.
    I64(i64),
    /// Value of 32-bit floating point number.
    F32(f32),
    /// Value of 64-bit floating point number.
    F64(f64),
    /// An optional function reference.
    FuncRef(Option<Func>),
    /// An optional external reference.
    ExternRef(Option<ExternRef>),
}

impl<E: WasmEngine> From<&Value> for crate::backend::Value<E> {
    fn from(value: &Value) -> Self {
        match value {
            Value::I32(i32) => Self::I32(*i32),
            Value::I64(i64) => Self::I64(*i64),
            Value::F32(f32) => Self::F32(*f32),
            Value::F64(f64) => Self::F64(*f64),
            Value::FuncRef(None) => Self::FuncRef(None),
            Value::FuncRef(Some(func)) => Self::FuncRef(Some(func.func.cast::<E::Func>().clone())),
            Value::ExternRef(None) => Self::ExternRef(None),
            Value::ExternRef(Some(extern_ref)) => {
                Self::ExternRef(Some(extern_ref.extern_ref.cast::<E::ExternRef>().clone()))
            }
        }
    }
}

impl<E: WasmEngine> From<&backend::Value<E>> for Value {
    fn from(value: &crate::backend::Value<E>) -> Self {
        match value {
            crate::backend::Value::I32(i32) => Self::I32(*i32),
            crate::backend::Value::I64(i64) => Self::I64(*i64),
            crate::backend::Value::F32(f32) => Self::F32(*f32),
            crate::backend::Value::F64(f64) => Self::F64(*f64),
            crate::backend::Value::FuncRef(None) => Self::FuncRef(None),
            crate::backend::Value::FuncRef(Some(func)) => Self::FuncRef(Some(Func {
                func: BackendObject::new(func.clone()),
            })),
            crate::backend::Value::ExternRef(None) => Self::ExternRef(None),
            crate::backend::Value::ExternRef(Some(extern_ref)) => {
                Self::ExternRef(Some(ExternRef {
                    extern_ref: BackendObject::new(extern_ref.clone()),
                }))
            }
        }
    }
}

/// Represents a nullable opaque reference to any data within WebAssembly.
#[derive(Clone, Debug)]
pub struct ExternRef {
    /// The backing implementation.
    extern_ref: BackendObject,
}

impl ExternRef {
    /// Creates a new [`ExternRef`] wrapping the given value.
    pub fn new<T: 'static + Send + Sync, C: AsContextMut>(
        mut ctx: C,
        object: impl Into<Option<T>>,
    ) -> Self {
        Self {
            extern_ref: BackendObject::new(
                <<C::Engine as WasmEngine>::ExternRef as WasmExternRef<C::Engine>>::new(
                    ctx.as_context_mut().inner,
                    object.into(),
                ),
            ),
        }
    }

    /// Returns a shared reference to the underlying data for this [`ExternRef`].
    pub fn downcast<'a, T: 'static, S: 'a, E: WasmEngine>(
        &self,
        ctx: StoreContext<'a, S, E>,
    ) -> Result<Option<&'a T>> {
        self.extern_ref.cast::<E::ExternRef>().downcast(ctx.inner)
    }
}

/// A Wasm or host function reference.
#[derive(Clone, Debug)]
pub struct Func {
    /// The backing implementation.
    func: BackendObject,
}

impl Func {
    /// Creates a new [`Func`] with the given arguments.
    pub fn new<C: AsContextMut>(
        mut ctx: C,
        ty: FuncType,
        func: impl 'static
            + Send
            + Sync
            + Fn(StoreContextMut<'_, C::UserState, C::Engine>, &[Value], &mut [Value]) -> Result<()>,
    ) -> Self {
        let raw_func = <<C::Engine as WasmEngine>::Func as WasmFunc<C::Engine>>::new(
            ctx.as_context_mut().inner,
            ty,
            move |ctx, args, results| {
                let mut input = ArgumentVec::with_capacity(args.len());
                input.extend(args.iter().map(Into::into));

                let mut output = ArgumentVec::with_capacity(results.len());
                output.extend(results.iter().map(Into::into));

                func(StoreContextMut { inner: ctx }, &input, &mut output)?;

                for (i, result) in output.iter().enumerate() {
                    results[i] = result.into();
                }

                std::result::Result::Ok(())
            },
        );

        Self {
            func: BackendObject::new(raw_func),
        }
    }

    /// Returns the function type of the [`Func`].
    pub fn ty<C: AsContext>(&self, ctx: C) -> FuncType {
        self.func
            .cast::<<C::Engine as WasmEngine>::Func>()
            .ty(ctx.as_context().inner)
    }

    /// Calls the Wasm or host function with the given inputs.
    ///
    /// The result is written back into the `outputs` buffer.
    pub fn call<C: AsContextMut>(
        &self,
        mut ctx: C,
        args: &[Value],
        results: &mut [Value],
    ) -> Result<()> {
        let raw_func = self.func.cast::<<C::Engine as WasmEngine>::Func>();

        let mut input = ArgumentVec::with_capacity(args.len());
        input.extend(args.iter().map(Into::into));

        let mut output = ArgumentVec::with_capacity(results.len());
        output.extend(results.iter().map(Into::into));

        raw_func.call::<C::UserState>(ctx.as_context_mut().inner, &input, &mut output)?;

        for (i, result) in output.iter().enumerate() {
            results[i] = result.into();
        }

        std::result::Result::Ok(())
    }
}

/// A Wasm global variable reference.
#[derive(Clone, Debug)]
pub struct Global {
    /// The backing implementation.
    global: BackendObject,
}

impl Global {
    /// Creates a new global variable to the store.
    pub fn new<C: AsContextMut>(mut ctx: C, initial_value: Value, mutable: bool) -> Self {
        Self {
            global: BackendObject::new(<<C::Engine as WasmEngine>::Global as WasmGlobal<
                C::Engine,
            >>::new(
                ctx.as_context_mut().inner,
                (&initial_value).into(),
                mutable,
            )),
        }
    }

    /// Returns the [`GlobalType`] of the global variable.
    pub fn ty<C: AsContext>(&self, ctx: C) -> GlobalType {
        self.global
            .cast::<<C::Engine as WasmEngine>::Global>()
            .ty(ctx.as_context().inner)
    }

    /// Returns the current value of the global variable.
    pub fn get<C: AsContext>(&self, ctx: C) -> Value {
        (&self
            .global
            .cast::<<C::Engine as WasmEngine>::Global>()
            .get(ctx.as_context().inner))
            .into()
    }

    /// Sets a new value to the global variable.
    pub fn set<C: AsContextMut>(&self, mut ctx: C, new_value: Value) -> Result<()> {
        self.global
            .cast::<<C::Engine as WasmEngine>::Global>()
            .set(ctx.as_context_mut().inner, (&new_value).into())
    }
}

/// A parsed and validated WebAssembly module.
#[derive(Clone, Debug)]
pub struct Module {
    /// The backing implementation.
    module: BackendObject,
}

impl Module {
    /// Creates a new Wasm [`Module`] from the given byte stream.
    pub fn new<E: WasmEngine>(engine: &Engine<E>, stream: impl std::io::Read) -> Result<Self> {
        Ok(Self {
            module: BackendObject::new(<E::Module as WasmModule<E>>::new(&engine.backend, stream)?),
        })
    }

    /// Returns an iterator over the exports of the [`Module`].
    pub fn exports<E: WasmEngine>(
        &self,
        #[allow(unused_variables)] engine: &Engine<E>,
    ) -> impl '_ + Iterator<Item = ExportType<'_>> {
        self.module.cast::<E::Module>().exports()
    }

    /// Looks up an export in this [`Module`] by its `name`.
    ///
    /// Returns `None` if no export with the name was found.
    pub fn get_export<E: WasmEngine>(
        &self,
        #[allow(unused_variables)] engine: &Engine<E>,
        name: &str,
    ) -> Option<ExternType> {
        self.module.cast::<E::Module>().get_export(name)
    }

    /// Returns an iterator over the imports of the [`Module`].
    pub fn imports<E: WasmEngine>(
        &self,
        #[allow(unused_variables)] engine: &Engine<E>,
    ) -> impl '_ + Iterator<Item = ImportType<'_>> {
        self.module.cast::<E::Module>().imports()
    }
}

/// An instantiated WebAssembly [`Module`].
///
/// This type represents an instantiation of a [`Module`].
/// It primarily allows to access its [`exports`](Instance::exports)
/// to call functions, get or set globals, read or write memory, etc.
///
/// When interacting with any Wasm code you will want to create an
/// [`Instance`] in order to execute anything.
#[derive(Clone, Debug)]
pub struct Instance {
    /// The backing implementation.
    instance: BackendObject,
}

impl Instance {
    /// Creates a new [`Instance`] which runs code from the provided module against the given import set.
    pub fn new<C: AsContextMut>(mut ctx: C, module: &Module, imports: &Imports) -> Result<Self> {
        let mut backend_imports = crate::backend::Imports::default();
        backend_imports.extend(
            imports
                .into_iter()
                .map(|((host, name), val)| ((host, name), (&val).into())),
        );

        Ok(Self {
            instance: BackendObject::new(<<C::Engine as WasmEngine>::Instance as WasmInstance<
                C::Engine,
            >>::new(
                ctx.as_context_mut().inner,
                module.module.cast(),
                &backend_imports,
            )?),
        })
    }

    /// Returns an iterator over the exports of the [`Instance`].
    pub fn exports<C: AsContext>(&self, ctx: C) -> impl Iterator<Item = Export> {
        self.instance
            .cast::<<C::Engine as WasmEngine>::Instance>()
            .exports(ctx.as_context().inner)
            .map(Into::into)
    }

    /// Returns the value exported to the given `name` if any.
    pub fn get_export<C: AsContext>(&self, ctx: C, name: &str) -> Option<Extern> {
        self.instance
            .cast::<<C::Engine as WasmEngine>::Instance>()
            .get_export(ctx.as_context().inner, name)
            .as_ref()
            .map(Into::into)
    }
}

/// A Wasm linear memory reference.
#[derive(Clone, Debug)]
pub struct Memory {
    /// The backing implementation.
    memory: BackendObject,
}

impl Memory {
    /// Creates a new linear memory to the store.
    pub fn new<C: AsContextMut>(mut ctx: C, ty: MemoryType) -> Result<Self> {
        Ok(Self {
            memory: BackendObject::new(<<C::Engine as WasmEngine>::Memory as WasmMemory<
                C::Engine,
            >>::new(ctx.as_context_mut().inner, ty)?),
        })
    }

    /// Returns the memory type of the linear memory.
    pub fn ty<C: AsContext>(&self, ctx: C) -> MemoryType {
        self.memory
            .cast::<<C::Engine as WasmEngine>::Memory>()
            .ty(ctx.as_context().inner)
    }

    /// Grows the linear memory by the given amount of new pages.
    pub fn grow<C: AsContextMut>(&self, mut ctx: C, additional: u32) -> Result<u32> {
        self.memory
            .cast::<<C::Engine as WasmEngine>::Memory>()
            .grow(ctx.as_context_mut().inner, additional)
    }

    /// Returns the amount of pages in use by the linear memory.
    pub fn current_pages<C: AsContext>(&self, ctx: C) -> u32 {
        self.memory
            .cast::<<C::Engine as WasmEngine>::Memory>()
            .current_pages(ctx.as_context().inner)
    }

    /// Reads `n` bytes from `memory[offset..offset+n]` into `buffer`
    /// where `n` is the length of `buffer`.
    pub fn read<C: AsContext>(&self, ctx: C, offset: usize, buffer: &mut [u8]) -> Result<()> {
        self.memory
            .cast::<<C::Engine as WasmEngine>::Memory>()
            .read(ctx.as_context().inner, offset, buffer)
    }

    /// Writes `n` bytes to `memory[offset..offset+n]` from `buffer`
    /// where `n` if the length of `buffer`.
    pub fn write<C: AsContextMut>(&self, mut ctx: C, offset: usize, buffer: &[u8]) -> Result<()> {
        self.memory
            .cast::<<C::Engine as WasmEngine>::Memory>()
            .write(ctx.as_context_mut().inner, offset, buffer)
    }
}

/// A Wasm table reference.
#[derive(Clone, Debug)]
pub struct Table {
    /// The backing implementation.
    table: BackendObject,
}

impl Table {
    /// Creates a new table to the store.
    pub fn new<C: AsContextMut>(mut ctx: C, ty: TableType, init: Value) -> Result<Self> {
        Ok(Self {
            table: BackendObject::new(<<C::Engine as WasmEngine>::Table as WasmTable<
                C::Engine,
            >>::new(
                ctx.as_context_mut().inner, ty, (&init).into()
            )?),
        })
    }

    /// Returns the type and limits of the table.
    pub fn ty<C: AsContext>(&self, ctx: C) -> TableType {
        self.table
            .cast::<<C::Engine as WasmEngine>::Table>()
            .ty(ctx.as_context().inner)
    }

    /// Returns the current size of the [`Table`].
    pub fn size<C: AsContext>(&self, ctx: C) -> u32 {
        self.table
            .cast::<<C::Engine as WasmEngine>::Table>()
            .size(ctx.as_context().inner)
    }

    /// Grows the table by the given amount of elements.
    ///
    /// Returns the old size of the [`Table`] upon success.
    pub fn grow<C: AsContextMut>(&self, mut ctx: C, delta: u32, init: Value) -> Result<u32> {
        self.table.cast::<<C::Engine as WasmEngine>::Table>().grow(
            ctx.as_context_mut().inner,
            delta,
            (&init).into(),
        )
    }

    /// Returns the [`Table`] element value at `index`.
    ///
    /// Returns `None` if `index` is out of bounds.
    pub fn get<C: AsContext>(&self, ctx: C, index: u32) -> Option<Value> {
        self.table
            .cast::<<C::Engine as WasmEngine>::Table>()
            .get(ctx.as_context().inner, index)
            .as_ref()
            .map(Into::into)
    }

    /// Sets the [`Value`] of this [`Table`] at `index`.
    pub fn set<C: AsContextMut>(&self, mut ctx: C, index: u32, value: Value) -> Result<()> {
        self.table.cast::<<C::Engine as WasmEngine>::Table>().set(
            ctx.as_context_mut().inner,
            index,
            (&value).into(),
        )
    }
}

/// A type-erased object that corresponds to an item from a WebAssembly backend runtime.
struct BackendObject {
    /// The backing implementation.
    inner: Box<dyn AnyCloneBoxed>,
}

impl BackendObject {
    /// Creates a new backend object that wraps the given value.
    pub fn new<T: 'static + Clone + Send + Sync>(value: T) -> Self {
        Self {
            inner: Box::new(value),
        }
    }

    /// Casts this backend object to the provided type, or panics if the types conflicted.
    pub fn cast<T: 'static>(&self) -> &T {
        self.inner
            .as_any()
            .downcast_ref()
            .expect("Attempted to use incorrect context to access function.")
    }
}

impl Clone for BackendObject {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone_boxed(),
        }
    }
}

impl std::fmt::Debug for BackendObject {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BackendObject").finish()
    }
}

/// Marks a type that can be cloned into a box and interpreted as an [`Any`](std::any::Any).
trait AnyCloneBoxed: Any + Send + Sync {
    /// Gets this type as an [`Any`](std::any::Any).
    fn as_any(&self) -> &(dyn Any + Send + Sync);

    /// Creates a clone of this type into a new box.
    fn clone_boxed(&self) -> Box<dyn AnyCloneBoxed>;
}

impl<T: Any + Clone + Send + Sync> AnyCloneBoxed for T {
    fn as_any(&self) -> &(dyn Any + Send + Sync) {
        self
    }

    fn clone_boxed(&self) -> Box<dyn AnyCloneBoxed> {
        Box::new(self.clone())
    }
}

/// A trait used to get shared access to a [`Store`].
pub trait AsContext {
    /// The engine type associated with the context.
    type Engine: WasmEngine;

    /// The user state associated with the [`Store`], aka the `T` in `Store<T>`.
    type UserState;

    /// Returns the store context that this type provides access to.
    fn as_context(&self) -> StoreContext<Self::UserState, Self::Engine>;
}

/// A trait used to get exclusive access to a [`Store`].
pub trait AsContextMut: AsContext {
    /// Returns the store context that this type provides access to.
    fn as_context_mut(&mut self) -> StoreContextMut<Self::UserState, Self::Engine>;
}

impl<'a, T: 'a, E: WasmEngine> AsContext for StoreContext<'a, T, E> {
    type Engine = E;

    type UserState = T;

    fn as_context(&self) -> StoreContext<Self::UserState, Self::Engine> {
        StoreContext {
            inner: crate::backend::AsContext::as_context(&self.inner),
        }
    }
}

impl<'a, T: 'a, E: WasmEngine> AsContext for StoreContextMut<'a, T, E> {
    type Engine = E;

    type UserState = T;

    fn as_context(&self) -> StoreContext<Self::UserState, Self::Engine> {
        StoreContext {
            inner: crate::backend::AsContext::as_context(&self.inner),
        }
    }
}

impl<'a, T: 'a, E: WasmEngine> AsContextMut for StoreContextMut<'a, T, E> {
    fn as_context_mut(&mut self) -> StoreContextMut<Self::UserState, Self::Engine> {
        StoreContextMut {
            inner: crate::backend::AsContextMut::as_context_mut(&mut self.inner),
        }
    }
}
