#![allow(warnings)]

pub mod backend;

use anyhow::*;
use crate::backend::*;
use ref_cast::*;
use std::any::*;
use std::sync::*;

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

#[derive(RefCast, Clone)]
#[repr(transparent)]
pub struct Engine<E: WasmEngine> {
    backend: E
}

impl<E: WasmEngine> Engine<E> {
    pub fn new(backend: E) -> Self {
        Self {
            backend
        }
    }

    pub fn into_backend(self) -> E {
        self.backend
    }
}

pub struct Store<T, E: WasmEngine> {
    inner: E::Store<T>
}

impl<T, E: WasmEngine> Store<T, E> {
    pub fn new(engine: &Engine<E>, data: T) -> Self {
        Self {
            inner: <E::Store<T> as WasmStore<T, E>>::new(&engine.backend, data)
        }
    }

    pub fn engine(&self) -> &Engine<E> {
        Engine::<E>::ref_cast(self.inner.engine())
    }

    pub fn data(&self) -> &T {
        self.inner.data()
    }

    pub fn data_mut(&mut self) -> &mut T {
        self.inner.data_mut()
    }

    pub fn into_data(self) -> T {
        self.inner.into_data()
    }
}

pub struct StoreContext<'a, T: 'a, E: WasmEngine> {
    inner: <E::Store<T> as WasmStore<T, E>>::Context<'a>
}

impl<'a, T: 'a, E: WasmEngine> StoreContext<'a, T, E> {
    pub fn engine(&self) -> &Engine<E> {
        Engine::<E>::ref_cast(self.inner.engine())
    }

    pub fn data(&self) -> &T {
        self.inner.data()
    }
}

pub struct StoreContextMut<'a, T: 'a, E: WasmEngine> {
    inner: <E::Store<T> as WasmStore<T, E>>::ContextMut<'a>
}

impl<'a, T: 'a, E: WasmEngine> StoreContextMut<'a, T, E> {
    pub fn engine(&self) -> &Engine<E> {
        Engine::<E>::ref_cast(self.inner.engine())
    }

    pub fn data(&self) -> &T {
        self.inner.data()
    }

    pub fn data_mut(&mut self) -> &mut T {
        self.inner.data_mut()
    }
}

#[derive(Clone, Debug)]
pub enum Value {
    I32(i32),
    I64(i64),
    F32(f32),
    F64(f64),
    FuncRef(Option<Func>),
    //ExternRef(Option<E::ExternRef>),
}

impl Value {
    fn into_backend<E: WasmEngine>(&self) -> crate::backend::Value<E> {
        match self {
            Self::I32(i32) => crate::backend::Value::I32(*i32),
            Self::I64(i64) => crate::backend::Value::I64(*i64),
            Self::F32(f32) => crate::backend::Value::F32(*f32),
            Self::F64(f64) => crate::backend::Value::F64(*f64),
            Self::FuncRef(None) => crate::backend::Value::FuncRef(None),
            Self::FuncRef(Some(func)) => crate::backend::Value::FuncRef(Some(func.func.cast::<E::Func>().clone()))
        }
    }
}

fn test_it<T, E: WasmEngine>(mut ctx: StoreContextMut<T, E>) {
    let ff = Func::new::<T, E>(ctx, FuncType::new([], []), |ctx, _, _| { println!("hit it"); Ok(()) });
}

#[derive(Clone, Debug)]
pub struct Func {
    func: BackendObject
}

impl Func {
    pub fn new<'a: 'c, 'c, T: 'a, E: WasmEngine>(ctx: StoreContextMut<'c, T, E>, ty: FuncType, func: impl for<'b> backend::private::HostFunctionSealed<'a, 'b, T, E>) -> Self {
        //let raw_func = <E::Func as WasmFunc<E>>::new2::<T>(ctx.inner, ty, func);

        todo!()
        /*Self {
            func: BackendObject::new::<<C::Engine as WasmEngine>::Func>()),
        }*/
    }

    pub fn ty<C: AsContext>(&self, ctx: C) -> FuncType {
        self.func.cast::<<C::Engine as WasmEngine>::Func>().ty(ctx.as_context().inner)
    }
}

pub struct BackendObject {
    inner: Box<dyn AnyCloneBoxed>
}

impl BackendObject {
    pub fn new<T: 'static + Clone + Send + Sync>(value: T) -> Self {
        Self { inner: Box::new(value) }
    }
    
    pub fn cast<T: 'static>(&self) -> &T {
        self.inner.as_any().downcast_ref().expect("Attempted to use incorrect context to access function.")
    }
}

impl Clone for BackendObject {
    fn clone(&self) -> Self {
        Self { inner: self.inner.clone_boxed() }
    }
}

impl std::fmt::Debug for BackendObject {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BackendObject").finish()
    }
}

trait AnyCloneBoxed: Any + Send + Sync {
    fn as_any(&self) -> &(dyn Any + Send + Sync);
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

pub trait AsContext {
    type Engine: WasmEngine;
    type UserState;

    fn as_context(&self) -> StoreContext<Self::UserState, Self::Engine>;
}

pub trait AsContextMut: AsContext {
    fn as_context_mut(&mut self) -> StoreContextMut<Self::UserState, Self::Engine>;
}

impl<'a, T: 'a, E: WasmEngine> AsContext for StoreContext<'a, T, E> {
    type Engine = E;

    type UserState = T;

    fn as_context(&self) -> StoreContext<Self::UserState, Self::Engine> {
        StoreContext { inner: crate::backend::AsContext::as_context(&self.inner) }
    }
}

impl<'a, T: 'a, E: WasmEngine> AsContext for StoreContextMut<'a, T, E> {
    type Engine = E;

    type UserState = T;

    fn as_context(&self) -> StoreContext<Self::UserState, Self::Engine> {
        StoreContext { inner: crate::backend::AsContext::as_context(&self.inner) }
    }
}

impl<'a, T: 'a, E: WasmEngine> AsContextMut for StoreContextMut<'a, T, E> {
    fn as_context_mut(&mut self) -> StoreContextMut<Self::UserState, Self::Engine> {
        StoreContextMut { inner: crate::backend::AsContextMut::as_context_mut(&mut self.inner) }
    }
}