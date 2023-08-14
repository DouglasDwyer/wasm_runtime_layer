use anyhow::*;
use crate::*;
use std::collections::*;
use std::marker::*;
use std::ops::*;
use std::sync::*;

//#[cfg(feature = "backend_wasmi")]
//mod backend_wasmi;

#[derive(Clone)]
pub enum Value<E: WasmEngine> {
    I32(i32),
    I64(i64),
    F32(f32),
    F64(f64),
    FuncRef(Option<E::Func>),
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

impl<E: WasmEngine> std::fmt::Debug for Export<E> where Extern<E>: std::fmt::Debug
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Export").field("name", &self.name).field("value", &self.value).finish()
    }
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
    fn new2<'a, T: 'a>(ctx: impl AsContextMut<E, UserState = T>, ty: FuncType, func: impl for<'b> private::HostFunctionSealed<'a, 'b, T, E>) -> Self;
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

pub trait WasmStoreContext<'a, T, E: WasmEngine>: AsContext<E, UserState = T> {
    fn engine(&self) -> &E;
    fn data(&self) -> &T;
}

pub trait WasmStoreContextMut<'a, T, E: WasmEngine>: WasmStoreContext<'a, T, E> + AsContextMut<E, UserState = T> {
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

pub mod private {
    use super::*;

    pub trait HostFunctionSealed<'a: 'b, 'b, T: 'a, E: WasmEngine>: 'static + Send + Sync + Fn(<E::Store<T> as WasmStore<T, E>>::ContextMut<'b>, &[Value<E>], &mut [Value<E>]) -> Result<()> {}
    impl<'a: 'b, 'b, T: 'a, E: WasmEngine, U: 'static + Send + Sync + Fn(<E::Store<T> as WasmStore<T, E>>::ContextMut<'b>, &[Value<E>], &mut [Value<E>]) -> Result<()>> HostFunctionSealed<'a, 'b, T, E> for U {}
}
