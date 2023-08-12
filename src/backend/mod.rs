use anyhow::*;
use crate::*;
use std::collections::*;
use std::marker::*;
use std::ops::*;
use std::sync::*;

#[cfg(feature = "backend_wasmi")]
mod backend_wasmi;

pub(crate) struct Backend(Box<dyn BackendHolder>);

impl Backend {
    pub fn new<E: WasmEngine>() -> Self {
        Self(Box::new(BackendImpl::<E>::default()))
    }
}

impl Clone for Backend {
    fn clone(&self) -> Self {
        Self(self.clone_boxed())
    }
}

impl Deref for Backend {
    type Target = dyn BackendHolder;

    fn deref(&self) -> &Self::Target {
        &*self.0
    }
}

pub(crate) trait BackendHolder: 'static + Send + Sync {
    fn clone_boxed(&self) -> Box<dyn BackendHolder>;

    fn module_new(&self, engine: &AnySendSync, stream: &mut dyn std::io::Read) -> Result<Arc<AnySendSync>>;
    fn module_exports<'a>(&self, module: &'a AnySendSync) -> Box<dyn 'a + Iterator<Item = ExportType<'a>>>;
    fn module_get_export(&self, module: &AnySendSync, name: &str) -> Option<ExternType>;
    fn module_imports<'a>(&self, module: &'a AnySendSync) -> Box<dyn 'a + Iterator<Item = ImportType<'a>>>;
}

struct BackendImpl<E: WasmEngine>(PhantomData<fn(E)>);

impl<E: WasmEngine> BackendImpl<E> {
    fn cast<T: 'static>(value: &AnySendSync) -> &T {
        value.downcast_ref::<T>().expect("Backend value was of incorrect type.")
    }
}

impl<E: WasmEngine> Default for BackendImpl<E> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<E: WasmEngine> BackendHolder for BackendImpl<E> {
    fn clone_boxed(&self) -> Box<dyn BackendHolder> {
        Box::new(Self::default())
    }

    fn module_new(&self, engine: &AnySendSync, stream: &mut dyn std::io::Read) -> Result<Arc<AnySendSync>> {
        Ok(Arc::new(<E::Module as WasmModule<E>>::new(Self::cast(engine), stream)?))
    }

    fn module_exports<'a>(&self, module: &'a AnySendSync) -> Box<dyn 'a + Iterator<Item = ExportType<'a>>> {
        Self::cast::<E::Module>(module).exports()
    }

    fn module_get_export(&self, module: &AnySendSync, name: &str) -> Option<ExternType> {
        Self::cast::<E::Module>(module).get_export(name)
    }

    fn module_imports<'a>(&self, module: &'a AnySendSync) -> Box<dyn 'a + Iterator<Item = ImportType<'a>>> {
        Self::cast::<E::Module>(module).imports()
    }
}

#[derive(Clone)]
pub enum Value<E: WasmEngine> {
    I32(i32),
    I64(i64),
    F32(f32),
    F64(f64),
    FuncRef(Option<E::Func>),
    ExternRef(Option<E::ExternRef>),
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ValueType {
    I32,
    I64,
    F32,
    F64,
    FuncRef,
    ExternRef,
}

/// The type of a global variable.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct GlobalType {
    /// The value type of the global variable.
    content: ValueType,
    /// The mutability of the global variable.
    mutable: bool,
}

impl GlobalType {
    /// Creates a new [`GlobalType`] from the given [`ValueType`] and [`Mutability`].
    pub fn new(content: ValueType, mutable: bool) -> Self {
        Self {
            content,
            mutable,
        }
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
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
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
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct MemoryType {
    initial: u32,
    maximum: Option<u32>,
}

impl MemoryType {
    /// Creates a new memory type with initial and optional maximum pages.
    pub fn new(initial: u32, maximum: Option<u32>) -> Self {
        Self {
            initial,
            maximum,
        }
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
#[derive(Debug, Clone)]
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

/// An external item to a WebAssembly module.
///
/// This is returned from [`Instance::exports`](crate::Instance::exports)
/// or [`Instance::get_export`](crate::Instance::get_export).
pub enum Extern<E: WasmEngine> {
    /// A WebAssembly global which acts like a [`Cell<T>`] of sorts, supporting `get` and `set` operations.
    ///
    /// [`Cell<T>`]: https://doc.rust-lang.org/core/cell/struct.Cell.html
    Global(E::Global),
    /// A WebAssembly table which is an array of funtion references.
    Table(E::Table),
    /// A WebAssembly linear memory.
    Memory(E::Memory),
    /// A WebAssembly function which can be called.
    Func(E::Func),
}

impl<E: WasmEngine> Extern<E> {
    /// Returns the underlying global variable if `self` is a global variable.
    ///
    /// Returns `None` otherwise.
    pub fn into_global(self) -> Option<E::Global> {
        if let Self::Global(global) = self {
            return Some(global);
        }
        None
    }

    /// Returns the underlying table if `self` is a table.
    ///
    /// Returns `None` otherwise.
    pub fn into_table(self) -> Option<E::Table> {
        if let Self::Table(table) = self {
            return Some(table);
        }
        None
    }

    /// Returns the underlying linear memory if `self` is a linear memory.
    ///
    /// Returns `None` otherwise.
    pub fn into_memory(self) -> Option<E::Memory> {
        if let Self::Memory(memory) = self {
            return Some(memory);
        }
        None
    }

    /// Returns the underlying function if `self` is a function.
    ///
    /// Returns `None` otherwise.
    pub fn into_func(self) -> Option<E::Func> {
        if let Self::Func(func) = self {
            return Some(func);
        }
        None
    }

    /// Returns the type associated with this [`Extern`].
    ///
    /// # Panics
    ///
    /// If this item does not belong to the `store` provided.
    pub fn ty(&self, ctx: impl AsContext<E>) -> ExternType {
        match self {
            Extern::Global(global) => global.ty(ctx).into(),
            Extern::Table(table) => table.ty(ctx).into(),
            Extern::Memory(memory) => memory.ty(ctx).into(),
            Extern::Func(func) => func.ty(ctx).into(),
        }
    }
}

impl<E: WasmEngine> Clone for Extern<E> {
    fn clone(&self) -> Self {
        match self {
            Self::Global(arg0) => Self::Global(arg0.clone()),
            Self::Table(arg0) => Self::Table(arg0.clone()),
            Self::Memory(arg0) => Self::Memory(arg0.clone()),
            Self::Func(arg0) => Self::Func(arg0.clone()),
        }
    }
}

impl<E: WasmEngine> std::fmt::Debug for Extern<E>
where
    E::Global: std::fmt::Debug,
    E::Func: std::fmt::Debug,
    E::Memory: std::fmt::Debug,
    E::Table: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Global(arg0) => f.debug_tuple("Global").field(arg0).finish(),
            Self::Table(arg0) => f.debug_tuple("Table").field(arg0).finish(),
            Self::Memory(arg0) => f.debug_tuple("Memory").field(arg0).finish(),
            Self::Func(arg0) => f.debug_tuple("Func").field(arg0).finish(),
        }
    }
}

/// A descriptor for an exported WebAssembly value of a [`Module`].
///
/// This type is primarily accessed from the [`Module::exports`] method and describes
/// what names are exported from a Wasm [`Module`] and the type of the item that is exported.
#[derive(Debug)]
pub struct ExportType<'module> {
    /// The name by which the export is known.
    pub name: &'module str,
    /// The type of the exported item.
    pub ty: ExternType,
}

/// A descriptor for an exported WebAssembly value of an [`Instance`].
///
/// This type is primarily accessed from the [`Instance::exports`] method and describes
/// what names are exported from a Wasm [`Instance`] and the type of the item that is exported.
pub struct Export<E: WasmEngine> {
    /// The name by which the export is known.
    pub name: String,
    /// The value of the exported item.
    pub value: Extern<E>,
}

impl<E: WasmEngine> std::fmt::Debug for Export<E> where Extern<E>: std::fmt::Debug
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Export").field("name", &self.name).field("value", &self.value).finish()
    }
}

/// A descriptor for an imported value into a Wasm [`Module`].
///
/// This type is primarily accessed from the [`Module::imports`] method.
/// Each [`ImportType`] describes an import into the Wasm module with the `module/name`
/// that it is imported from as well as the type of item that is being imported.
#[derive(Debug)]
pub struct ImportType<'module> {
    /// The module import name.
    pub module: &'module str,
    /// The name of the imported item.
    pub name: &'module str,
    /// The external item type.
    pub ty: ExternType,
}

/// All of the import data used when instantiating.
/// ```
#[derive(Clone)]
pub struct Imports<E: WasmEngine> {
    pub(crate) map: HashMap<(String, String), Extern<E>>,
}

impl<E: WasmEngine> Imports<E> {
    /// Create a new `Imports`.
    pub fn new() -> Self {
        Self { map: HashMap::default() }
    }

    /// Gets an export given a module and a name
    pub fn get_export(&self, module: &str, name: &str) -> Option<Extern<E>> {
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
        contents: impl IntoIterator<Item = (String, Extern<E>)>,
    ) {
        for (name, extern_) in contents.into_iter() {
            self.map.insert((ns.to_string(), name.clone()), extern_);
        }
    }

    /// Add a single import with a namespace `ns` and name `name`.
    pub fn define(&mut self, ns: &str, name: &str, val: impl Into<Extern<E>>) {
        self.map
            .insert((ns.to_string(), name.to_string()), val.into());
    }

    /// Iterates through all the imports in this structure
    pub fn iter(&self) -> ImportsIterator<E> {
        ImportsIterator::new(self)
    }
}

pub struct ImportsIterator<'a, E: WasmEngine> {
    iter: std::collections::hash_map::Iter<'a, (String, String), Extern<E>>,
}

impl<'a, E: WasmEngine> ImportsIterator<'a, E> {
    fn new(imports: &'a Imports<E>) -> Self {
        let iter = imports.map.iter();
        Self { iter }
    }
}

impl<'a, E: WasmEngine> Iterator for ImportsIterator<'a, E> {
    type Item = (&'a str, &'a str, &'a Extern<E>);

    fn next(&mut self) -> Option<Self::Item> {
        self.iter
            .next()
            .map(|(k, v)| (k.0.as_str(), k.1.as_str(), v))
    }
}

impl<E: WasmEngine> IntoIterator for &Imports<E> {
    type IntoIter = std::collections::hash_map::IntoIter<(String, String), Extern<E>>;
    type Item = ((String, String), Extern<E>);

    fn into_iter(self) -> Self::IntoIter {
        self.map.clone().into_iter()
    }
}

impl<E: WasmEngine> Default for Imports<E> {
    fn default() -> Self {
        Self::new()
    }
}

impl<E: WasmEngine> Extend<((String, String), Extern<E>)> for Imports<E> {
    fn extend<T: IntoIterator<Item = ((String, String), Extern<E>)>>(&mut self, iter: T) {
        for ((ns, name), ext) in iter.into_iter() {
            self.define(&ns, &name, ext);
        }
    }
}

impl<E: WasmEngine> std::fmt::Debug for Imports<E> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        enum SecretMap {
            Empty,
            Some(usize),
        }

        impl SecretMap {
            fn new(len: usize) -> Self {
                if len == 0 {
                    Self::Empty
                } else {
                    Self::Some(len)
                }
            }
        }

        impl std::fmt::Debug for SecretMap {
            fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                match self {
                    Self::Empty => write!(f, "(empty)"),
                    Self::Some(len) => write!(f, "(... {} item(s) ...)", len),
                }
            }
        }

        f.debug_struct("Imports")
            .field("map", &SecretMap::new(self.map.len()))
            .finish()
    }
}

pub trait WasmEngine: 'static + Clone + Sized + Send + Sync {
    type ExternRef: WasmExternRef<Self>;
    type Func: WasmFunc<Self>;
    type Global: WasmGlobal<Self>;
    type Instance: WasmInstance<Self>;
    type Memory: WasmMemory<Self>;
    type Module: WasmModule<Self>;
    type Store<T>: WasmStore<T, Self>;
    type Table: WasmTable<Self>;
}

pub trait WasmExternRef<E: WasmEngine>: Sized + Send + Sync {
    fn new<T: 'static + Send + Sync>(ctx: impl AsContextMut<E>, object: T) -> Self;
    fn downcast<'a, T: 'static>(&self, store: <E::Store<T> as WasmStore<T, E>>::Context<'a>) -> Option<&'a T>;
}

pub trait WasmFunc<E: WasmEngine>: Clone + Sized + Send + Sync {
    fn new<'a, T: 'a>(ctx: impl AsContextMut<E, UserState = T>, ty: FuncType, func: impl HostFunction<'a, T, E>) -> Self;
    fn ty(&self, ctx: impl AsContext<E>) -> FuncType;
    fn call<T>(&self, ctx: impl AsContextMut<E>, args: &[Value<E>], results: &mut [Value<E>]) -> Result<()>;
}

pub trait WasmGlobal<E: WasmEngine>: Clone + Sized + Send + Sync {
    fn new(ctx: impl AsContextMut<E>, value: Value<E>, mutable: bool) -> Self;
    fn ty(&self, ctx: impl AsContext<E>) -> GlobalType;
    fn set(&self, ctx: impl AsContextMut<E>, new_value: Value<E>) -> Result<()>;
    fn get(&self, ctx: impl AsContext<E>) -> Value<E>;
}

pub trait WasmMemory<E: WasmEngine>: Clone + Sized + Send + Sync {
    fn new(ctx: impl AsContextMut<E>, ty: MemoryType) -> Result<Self>;
    fn ty(&self, ctx: impl AsContext<E>) -> MemoryType;
    fn grow(&self, ctx: impl AsContextMut<E>, additional: u32) -> Result<u32>;
    fn current_pages(&self, ctx: impl AsContext<E>) -> u32;
    fn read(&self, ctx: impl AsContext<E>, offset: usize, buffer: &mut [u8]) -> Result<()>;
    fn write(&self, ctx: impl AsContextMut<E>, offset: usize, buffer: &[u8]) -> Result<()>;
}

pub trait WasmTable<E: WasmEngine>: Clone + Sized + Send + Sync {
    fn new(ctx: impl AsContextMut<E>, ty: TableType, init: Value<E>) -> Result<Self>;
    fn ty(&self, ctx: impl AsContext<E>) -> TableType;
    fn size(&self, ctx: impl AsContext<E>) -> u32;
    fn grow(&self, ctx: impl AsContextMut<E>, delta: u32, init: Value<E>) -> Result<u32>;
    fn get(&self, ctx: impl AsContext<E>, index: u32) -> Option<Value<E>>;
    fn set(&self, ctx: impl AsContextMut<E>, index: u32, value: Value<E>) -> Result<()>;
}

pub trait WasmInstance<E: WasmEngine>: Sized + Send + Sync {
    fn new(store: impl AsContextMut<E>, module: &E::Module, imports: &Imports<E>) -> Result<Self>;
    fn exports(&self, store: impl AsContext<E>) -> Box<dyn Iterator<Item = Export<E>>>;
    fn get_export(&self, store: impl AsContext<E>, name: &str) -> Option<Extern<E>>;
}

pub trait WasmModule<E: WasmEngine>: Sized + Send + Sync {
    fn new(engine: &E, stream: impl std::io::Read) -> Result<Self>;
    fn exports(&self) -> Box<dyn '_ + Iterator<Item = ExportType<'_>>>;
    fn get_export(&self, name: &str) -> Option<ExternType>;
    fn imports(&self) -> Box<dyn '_ + Iterator<Item = ImportType<'_>>>;
}

pub trait WasmStore<T, E: WasmEngine> {
    type Context<'a>: WasmStoreContext<'a, T, E> where T: 'a;
    type ContextMut<'a>: WasmStoreContextMut<'a, T, E> where T: 'a;

    fn new(engine: &E, data: T) -> Self;
    fn engine(&self) -> &E;
    fn data(&self) -> &T;
    fn data_mut(&mut self) -> &mut T;
    fn into_data(self) -> T;
}

pub trait WasmStoreContext<'a, T, E: WasmEngine> {
    fn engine(&self) -> &E;
    fn data(&self) -> &T;
}

pub trait WasmStoreContextMut<'a, T, E: WasmEngine> {
    fn data_mut(&mut self) -> &mut T;
}

pub trait AsContext<E: WasmEngine> {
    type UserState;

    fn as_context(&self) -> <E::Store<Self::UserState> as WasmStore<Self::UserState, E>>::Context<'_>;
}

pub trait AsContextMut<E: WasmEngine>: AsContext<E> {
    fn as_context_mut(&mut self) -> <E::Store<Self::UserState> as WasmStore<Self::UserState, E>>::ContextMut<'_>;
}

impl<T: AsContext<E>, E: WasmEngine> AsContext<E> for &T {
    type UserState = T::UserState;

    fn as_context(&self) -> <E::Store<Self::UserState> as WasmStore<Self::UserState, E>>::Context<'_> {
        (*self).as_context()
    }
}

impl<T: AsContext<E>, E: WasmEngine> AsContext<E> for &mut T {
    type UserState = T::UserState;

    fn as_context(&self) -> <E::Store<Self::UserState> as WasmStore<Self::UserState, E>>::Context<'_> {
        (**self).as_context()
    }
}

impl<T: AsContextMut<E>, E: WasmEngine> AsContextMut<E> for &mut T {
    fn as_context_mut(&mut self) -> <E::Store<Self::UserState> as WasmStore<Self::UserState, E>>::ContextMut<'_> {
        (*self).as_context_mut()
    }
}

pub trait HostFunction<'a, T: 'a, E: WasmEngine>: for<'b> private::HostFunctionSealed<'a, 'b, T, E> {}
impl<'a, T: 'a, E: WasmEngine, U: for<'b> private::HostFunctionSealed<'a, 'b, T, E>> HostFunction<'a, T, E> for U {}

mod private {
    use super::*;

    pub trait HostFunctionSealed<'a: 'b, 'b, T: 'a, E: WasmEngine>: 'static + Send + Sync + Fn(<E::Store<T> as WasmStore<T, E>>::ContextMut<'b>, &[Value<E>], &mut [Value<E>]) -> Result<()> {}
    impl<'a: 'b, 'b, T: 'a, E: WasmEngine, U: 'static + Send + Sync + Fn(<E::Store<T> as WasmStore<T, E>>::ContextMut<'b>, &[Value<E>], &mut [Value<E>]) -> Result<()>> HostFunctionSealed<'a, 'b, T, E> for U {}
}
