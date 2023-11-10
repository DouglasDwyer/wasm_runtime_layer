use std::{
    collections::HashMap,
    ops::{Deref, DerefMut},
};

use js_sys::{Function, WebAssembly};
use slab::Slab;
use wasm_bindgen::JsValue;

use crate::backend::Extern;

#[derive(Default, Debug, Clone)]
/// Runtime for WebAssembly
pub struct Engine {}

/// Not Send + Sync
pub(crate) struct InstanceInner {
    pub(crate) instance: WebAssembly::Instance,
    pub(crate) exports: HashMap<String, Extern<Engine>>,
}

#[derive(Debug, Clone)]
pub struct Instance {
    pub(crate) id: usize,
}

pub(crate) struct FuncInner {
    pub(crate) func: Function,
}

#[derive(Debug, Clone)]
pub struct Func {
    id: usize,
}

#[derive(Debug, Clone)]
pub struct Memory {}

pub(crate) struct ModuleInner {
    pub(crate) module: js_sys::WebAssembly::Module,
    pub(crate) exports: js_sys::Object,
}

#[derive(Debug, Clone)]
pub struct Module {
    pub(crate) id: usize,
}

/// Owns all the data for the wasm module
pub struct Store<T> {
    pub(crate) engine: Engine,
    // Instances are not Send + Sync
    pub(crate) instances: Slab<InstanceInner>,
    // Modules are not Send + Sync
    pub(crate) modules: Slab<ModuleInner>,
    pub(crate) funcs: Slab<FuncInner>,
    pub(crate) data: T,
}

impl<T> Store<T> {
    pub(crate) fn create_func(&mut self, func: FuncInner) -> Func {
        Func {
            id: self.funcs.insert(func),
        }
    }
}

/// Immutable context to the store
pub struct StoreContext<'a, T: 'a> {
    /// The store
    pub(crate) store: &'a Store<T>,
}

impl<'a, T: 'a> StoreContext<'a, T> {
    pub fn new(store: &'a Store<T>) -> Self {
        Self { store }
    }
}

impl<'a, T> Deref for StoreContext<'a, T> {
    type Target = Store<T>;

    fn deref(&self) -> &Self::Target {
        self.store
    }
}

/// Mutable context to the store
pub struct StoreContextMut<'a, T: 'a> {
    /// The store
    pub(crate) store: &'a mut Store<T>,
}

impl<'a, T: 'a> StoreContextMut<'a, T> {
    pub fn new(store: &'a mut Store<T>) -> Self {
        Self { store }
    }
}

impl<'a, T> Deref for StoreContextMut<'a, T> {
    type Target = Store<T>;

    fn deref(&self) -> &Self::Target {
        self.store
    }
}

impl<'a, T> DerefMut for StoreContextMut<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.store
    }
}
