use crate::*;
use anyhow::*;
use fxhash::*;
use std::marker::*;
use std::ops::*;

#[cfg(feature = "backend_wasmi")]
/// The backend which provides support for the `wasmi` runtime.
mod backend_wasmi;

/// Runtime representation of a value.
#[derive(Clone)]
pub enum Value<E: WasmEngine> {
    /// Value of 32-bit signed or unsigned integer.
    I32(i32),
    /// Value of 64-bit signed or unsigned integer.
    I64(i64),
    /// Value of 32-bit floating point number.
    F32(f32),
    /// Value of 64-bit floating point number.
    F64(f64),
    /// An optional function reference.
    FuncRef(Option<E::Func>),
    /// An optional external reference.
    ExternRef(Option<E::ExternRef>),
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

impl<E: WasmEngine> std::fmt::Debug for Export<E>
where
    Extern<E>: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Export")
            .field("name", &self.name)
            .field("value", &self.value)
            .finish()
    }
}

/// All of the import data used when instantiating.
#[derive(Clone)]
pub struct Imports<E: WasmEngine> {
    /// The inner list of external imports.
    pub(crate) map: FxHashMap<(String, String), Extern<E>>,
}

impl<E: WasmEngine> Imports<E> {
    /// Create a new `Imports`.
    pub fn new() -> Self {
        Self {
            map: FxHashMap::default(),
        }
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

/// An iterator over imports.
pub struct ImportsIterator<'a, E: WasmEngine> {
    /// The inner iterator over external items.
    iter: std::collections::hash_map::Iter<'a, (String, String), Extern<E>>,
}

impl<'a, E: WasmEngine> ImportsIterator<'a, E> {
    /// Creates a new iterator over the imports of an instance.
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
        /// Stores backing debug data.
        enum SecretMap {
            /// The empty variant.
            Empty,
            /// The filled index variant.
            Some(usize),
        }

        impl SecretMap {
            /// Creates a new secret map representation of the given size.
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

/// Provides a backing implementation for a WebAssembly runtime.
pub trait WasmEngine: 'static + Clone + Sized + Send + Sync {
    /// The external reference type.
    type ExternRef: WasmExternRef<Self>;
    /// The function type.
    type Func: WasmFunc<Self>;
    /// The global type.
    type Global: WasmGlobal<Self>;
    /// The instance type.
    type Instance: WasmInstance<Self>;
    /// The memory type.
    type Memory: WasmMemory<Self>;
    /// The module type.
    type Module: WasmModule<Self>;
    /// The store type.
    type Store<T>: WasmStore<T, Self>;
    /// The store context type.
    type StoreContext<'a, T: 'a>: WasmStoreContext<'a, T, Self>;
    /// The mutable store context type.
    type StoreContextMut<'a, T: 'a>: WasmStoreContextMut<'a, T, Self>;
    /// The table type.
    type Table: WasmTable<Self>;
}

/// Provides a nullable opaque reference to any data within WebAssembly.
pub trait WasmExternRef<E: WasmEngine>: Clone + Sized + Send + Sync {
    /// Creates a new reference wrapping the given value.
    fn new<T: 'static + Send + Sync>(ctx: impl AsContextMut<E>, object: Option<T>) -> Self;
    /// Returns a shared reference to the underlying data.
    fn downcast<'a, T: 'static, S: 'a>(
        &self,
        store: E::StoreContext<'a, S>,
    ) -> Result<Option<&'a T>>;
}

/// Provides a Wasm or host function reference.
pub trait WasmFunc<E: WasmEngine>: Clone + Sized + Send + Sync {
    /// Creates a new function with the given arguments.
    fn new<T>(
        ctx: impl AsContextMut<E, UserState = T>,
        ty: FuncType,
        func: impl 'static
            + Send
            + Sync
            + Fn(E::StoreContextMut<'_, T>, &[Value<E>], &mut [Value<E>]) -> Result<()>,
    ) -> Self;
    /// Gets the function type of this object.
    fn ty(&self, ctx: impl AsContext<E>) -> FuncType;
    /// Calls the object with the given arguments.
    fn call<T>(
        &self,
        ctx: impl AsContextMut<E>,
        args: &[Value<E>],
        results: &mut [Value<E>],
    ) -> Result<()>;
}

/// Provides a Wasm global variable reference.
pub trait WasmGlobal<E: WasmEngine>: Clone + Sized + Send + Sync {
    /// Creates a new global variable to the store.
    fn new(ctx: impl AsContextMut<E>, value: Value<E>, mutable: bool) -> Self;
    /// Returns the type of the global variable.
    fn ty(&self, ctx: impl AsContext<E>) -> GlobalType;
    /// Sets the value of the global variable.
    fn set(&self, ctx: impl AsContextMut<E>, new_value: Value<E>) -> Result<()>;
    /// Gets the value of the global variable.
    fn get(&self, ctx: impl AsContext<E>) -> Value<E>;
}

/// Provides a Wasm linear memory reference.
pub trait WasmMemory<E: WasmEngine>: Clone + Sized + Send + Sync {
    /// Creates a new linear memory to the store.
    fn new(ctx: impl AsContextMut<E>, ty: MemoryType) -> Result<Self>;
    /// Returns the memory type of the linear memory.
    fn ty(&self, ctx: impl AsContext<E>) -> MemoryType;
    /// Grows the linear memory by the given amount of new pages.
    fn grow(&self, ctx: impl AsContextMut<E>, additional: u32) -> Result<u32>;
    /// Returns the amount of pages in use by the linear memory.
    fn current_pages(&self, ctx: impl AsContext<E>) -> u32;
    /// Reads `n` bytes from `memory[offset..offset+n]` into `buffer`
    /// where `n` is the length of `buffer`.
    fn read(&self, ctx: impl AsContext<E>, offset: usize, buffer: &mut [u8]) -> Result<()>;
    /// Writes `n` bytes to `memory[offset..offset+n]` from `buffer`
    /// where `n` if the length of `buffer`.
    fn write(&self, ctx: impl AsContextMut<E>, offset: usize, buffer: &[u8]) -> Result<()>;
}

/// Provides a Wasm table reference.
pub trait WasmTable<E: WasmEngine>: Clone + Sized + Send + Sync {
    /// Creates a new table to the store.
    fn new(ctx: impl AsContextMut<E>, ty: TableType, init: Value<E>) -> Result<Self>;
    /// Returns the type and limits of the table.
    fn ty(&self, ctx: impl AsContext<E>) -> TableType;
    /// Returns the current size of the table.
    fn size(&self, ctx: impl AsContext<E>) -> u32;
    /// Grows the table by the given amount of elements.
    fn grow(&self, ctx: impl AsContextMut<E>, delta: u32, init: Value<E>) -> Result<u32>;
    /// Returns the table element value at `index`.
    fn get(&self, ctx: impl AsContext<E>, index: u32) -> Option<Value<E>>;
    /// Sets the value of this table at `index`.
    fn set(&self, ctx: impl AsContextMut<E>, index: u32, value: Value<E>) -> Result<()>;
}

/// Provides an instantiated WASM module.
pub trait WasmInstance<E: WasmEngine>: Clone + Sized + Send + Sync {
    /// Creates a new instance.
    fn new(store: impl AsContextMut<E>, module: &E::Module, imports: &Imports<E>) -> Result<Self>;
    /// Gets the exports of this instance.
    fn exports(&self, store: impl AsContext<E>) -> Box<dyn Iterator<Item = Export<E>>>;
    /// Gets the export of the given name, if any, from this instance.
    fn get_export(&self, store: impl AsContext<E>, name: &str) -> Option<Extern<E>>;
}

/// Provides a parsed and validated WASM module.
pub trait WasmModule<E: WasmEngine>: Clone + Sized + Send + Sync {
    /// Creates a new module from the given byte stream.
    fn new(engine: &E, stream: impl std::io::Read) -> Result<Self>;
    /// Gets the export types of the module.
    fn exports(&self) -> Box<dyn '_ + Iterator<Item = ExportType<'_>>>;
    /// Gets the export type of the given name, if any, from this module.
    fn get_export(&self, name: &str) -> Option<ExternType>;
    /// Gets the import types of the module.
    fn imports(&self) -> Box<dyn '_ + Iterator<Item = ImportType<'_>>>;
}

/// Provides all of the global state that can be manipulated by WASM programs.
pub trait WasmStore<T, E: WasmEngine>:
    AsContext<E, UserState = T> + AsContextMut<E, UserState = T>
{
    /// Creates a new store atop the given engine.
    fn new(engine: &E, data: T) -> Self;
    /// Gets the engine associated with this store.
    fn engine(&self) -> &E;
    /// Gets an immutable reference to the underlying stored data.
    fn data(&self) -> &T;
    /// Gets a mutable reference to the underlying stored data.
    fn data_mut(&mut self) -> &mut T;
    /// Consumes `self` and returns its user provided data.
    fn into_data(self) -> T;
}

/// Provides a temporary immutable handle to a store.
pub trait WasmStoreContext<'a, T, E: WasmEngine>: AsContext<E, UserState = T> {
    /// Gets the engine associated with this store.
    fn engine(&self) -> &E;
    /// Gets an immutable reference to the underlying stored data.
    fn data(&self) -> &T;
}

/// Provides a temporary mutable handle to a store.
pub trait WasmStoreContextMut<'a, T, E: WasmEngine>:
    WasmStoreContext<'a, T, E> + AsContextMut<E, UserState = T>
{
    /// Gets a mutable reference to the underlying stored data.
    fn data_mut(&mut self) -> &mut T;
}

/// A trait used to get shared access to a store.
pub trait AsContext<E: WasmEngine> {
    /// The type of data associated with the store.
    type UserState;

    /// Returns the store context that this type provides access to.
    fn as_context(&self) -> E::StoreContext<'_, Self::UserState>;
}

/// A trait used to get mutable access to a store.
pub trait AsContextMut<E: WasmEngine>: AsContext<E> {
    /// Returns the store context that this type provides access to.
    fn as_context_mut(&mut self) -> E::StoreContextMut<'_, Self::UserState>;
}

impl<T: AsContext<E>, E: WasmEngine> AsContext<E> for &T {
    type UserState = T::UserState;

    fn as_context(&self) -> E::StoreContext<'_, Self::UserState> {
        (*self).as_context()
    }
}

impl<T: AsContext<E>, E: WasmEngine> AsContext<E> for &mut T {
    type UserState = T::UserState;

    fn as_context(&self) -> E::StoreContext<'_, Self::UserState> {
        (**self).as_context()
    }
}

impl<T: AsContextMut<E>, E: WasmEngine> AsContextMut<E> for &mut T {
    fn as_context_mut(&mut self) -> E::StoreContextMut<'_, Self::UserState> {
        (*self).as_context_mut()
    }
}
