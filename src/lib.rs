#![allow(warnings)]

pub mod backend;

use anyhow::*;
use crate::backend::*;
use std::any::*;
use std::sync::*;

type AnySendSync = dyn 'static + Any + Send + Sync;

#[derive(Clone)]
pub struct Engine {
    backend: Backend,
    engine: Arc<AnySendSync>
}

impl Engine {
    pub fn new<E: WasmEngine>(backend: E) -> Self {
        Self { backend: Backend::new::<E>(), engine: Arc::new(backend) }
    }
}

#[derive(Clone)]
pub struct Module {
    engine: Engine,
    module: Arc<AnySendSync>
}

impl Module {
    pub fn new(engine: &Engine, mut stream: impl std::io::Read) -> Result<Self> {
        engine.backend.module_new(&*engine.engine, &mut stream).map(|module| Self {
            engine: engine.clone(),
            module
        })
    }

    pub fn engine(&self) -> &Engine {
        &self.engine
    }
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