mod store;
pub use store::{Store, StoreContext, StoreContextMut, StoreInner};

use std::collections::HashMap;

use js_sys::{Function, WebAssembly};

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
