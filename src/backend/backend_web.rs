use std::{
    cell::{Ref, RefCell, RefMut},
    collections::HashMap,
    error::Error,
    fmt::Display,
    rc::Rc,
};

use anyhow::bail;
use js_sys::{JsString, Object, Reflect, WebAssembly};
use slab::Slab;
use wasm_bindgen::{closure::Closure, JsCast, JsValue};

use super::{
    AsContext, AsContextMut, Export, TableType, Value, WasmEngine, WasmExternRef, WasmFunc,
    WasmGlobal, WasmInstance, WasmMemory, WasmModule, WasmStore, WasmStoreContext,
    WasmStoreContextMut,
};
use crate::{
    backend::Extern,
    web::{
        Engine, Func, FuncInner, Instance, InstanceInner, Memory, Module, ModuleInner, Store,
        StoreContext, StoreContextMut, StoreInner,
    },
};

#[derive(Debug, Clone)]
struct JsErrorMsg {
    message: String,
}

impl Display for JsErrorMsg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.message.fmt(f)
    }
}

impl Error for JsErrorMsg {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        None
    }
}

impl From<&JsValue> for JsErrorMsg {
    fn from(value: &JsValue) -> Self {
        if let Some(v) = value.dyn_ref::<JsString>() {
            Self { message: v.into() }
        } else if let Ok(v) = Reflect::get(value, &"message".into()) {
            Self {
                message: v.as_string().expect("A string object"),
            }
        } else {
            Self {
                message: format!("{value:?}"),
            }
        }
    }
}

impl From<JsValue> for JsErrorMsg {
    fn from(value: JsValue) -> Self {
        Self::from(&value)
    }
}

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

impl WasmInstance<Engine> for Instance {
    fn new(
        mut store: impl super::AsContextMut<Engine>,
        module: &Module,
        imports: &super::Imports<Engine>,
    ) -> anyhow::Result<Self> {
        let _span = tracing::info_span!("Instance::new").entered();
        let mut store: StoreContextMut<_> = store.as_context_mut();

        let module = &mut store.modules[module.id];

        let import_object = js_sys::Object::new();

        for ((module, name), imp) in imports {
            tracing::debug!(module, name, "export");
        }

        tracing::info!("instantiate module");
        // TODO: async instantiation, possibly through a `.ready().await` call on the returned
        // module
        // let instance = WebAssembly::instantiate_module(&module.module, &imports);
        let instance =
            WebAssembly::Instance::new(&module.module, &import_object).map_err(JsErrorMsg::from)?;

        let exports = Reflect::get(&instance, &"exports".into()).expect("exports object");
        let exports = process_exports(&mut store, exports)?;

        let instance = InstanceInner { instance, exports };

        let instance_id = store;
        todo!();
    }

    fn exports(
        &self,
        store: impl super::AsContext<Engine>,
    ) -> Box<dyn Iterator<Item = super::Export<Engine>>> {
        // TODO: modify this trait to make the lifetime of the returned iterator depend on the
        // anonymous lifetime of the store
        let instance: &InstanceInner = &store.as_context().instances[self.id];
        Box::new(
            instance
                .exports
                .iter()
                .map(|(name, value)| Export {
                    name: name.into(),
                    value: value.clone(),
                })
                .collect::<Vec<_>>()
                .into_iter(),
        )
    }

    fn get_export(
        &self,
        store: impl super::AsContext<Engine>,
        name: &str,
    ) -> Option<super::Extern<Engine>> {
        let instance: &InstanceInner = &store.as_context().instances[self.id];
        instance.exports.get(name).cloned()
    }
}

/// Processes a wasm module's exports into a rust side hashmap
fn process_exports<T>(
    store: &mut StoreContextMut<T>,
    js_exports: JsValue,
) -> anyhow::Result<HashMap<String, Extern<Engine>>> {
    if !js_exports.is_object() {
        bail!(
            "WebAssembly exports must be an object, got '{:?}' instead",
            js_exports
        );
    }

    let js_exports: Object = js_exports.into();
    let names = Object::get_own_property_names(&js_exports);
    let len = names.length();

    let mut exports = HashMap::new();
    for i in 0..len {
        let name_js = Reflect::get_u32(&names, i).expect("names is array");
        let name = name_js.as_string().expect("name is string");
        let export = Reflect::get(&js_exports, &name_js).expect("js_exports is object");

        let export = match export.js_typeof().as_string().as_deref() {
            Some("function") => {
                let func = store.create_func(FuncInner {
                    func: export.dyn_into().unwrap(),
                });

                Extern::Func(func)
            }
            _ => unimplemented!(),
        };

        tracing::debug!(?name, ?export, "export");
        exports.insert(name, export);
    }
    Ok(exports)
}

#[derive(Debug, Clone)]
pub struct ExternRef {}

impl WasmExternRef<Engine> for ExternRef {
    fn new<T: 'static + Send + Sync>(ctx: impl AsContextMut<Engine>, object: Option<T>) -> Self {
        todo!()
    }

    fn downcast<'a, T: 'static, S: 'a>(
        &self,
        store: <Engine as WasmEngine>::StoreContext<'a, S>,
    ) -> anyhow::Result<Option<&'a T>> {
        todo!()
    }
}

impl WasmFunc<Engine> for Func {
    fn new<T>(
        ctx: impl AsContextMut<Engine, UserState = T>,
        ty: crate::FuncType,
        func: impl 'static
            + Send
            + Sync
            + Fn(StoreContextMut<T>, &[Value<Engine>], &mut [Value<Engine>]) -> anyhow::Result<()>,
    ) -> Self {
        let _span = tracing::info_span!("Func::new").entered();

        todo!()
        // let mut closure = Closure::new()
    }

    fn ty(&self, ctx: impl AsContext<Engine>) -> crate::FuncType {
        todo!()
    }

    fn call<T>(
        &self,
        ctx: impl AsContextMut<Engine>,
        args: &[Value<Engine>],
        results: &mut [Value<Engine>],
    ) -> anyhow::Result<()> {
        todo!()
    }
}

#[derive(Debug, Clone)]
pub struct Global {}

impl WasmGlobal<Engine> for Global {
    fn new(ctx: impl AsContextMut<Engine>, value: Value<Engine>, mutable: bool) -> Self {
        tracing::info!("Global::new");
        todo!()
    }

    fn ty(&self, ctx: impl AsContext<Engine>) -> crate::GlobalType {
        tracing::info!("Global::ty");
        todo!()
    }

    fn set(&self, ctx: impl AsContextMut<Engine>, new_value: Value<Engine>) -> anyhow::Result<()> {
        todo!()
    }

    fn get(&self, ctx: impl AsContextMut<Engine>) -> Value<Engine> {
        todo!()
    }
}

impl WasmMemory<Engine> for Memory {
    fn new(ctx: impl AsContextMut<Engine>, ty: crate::MemoryType) -> anyhow::Result<Self> {
        todo!()
    }

    fn ty(&self, ctx: impl AsContext<Engine>) -> crate::MemoryType {
        todo!()
    }

    fn grow(&self, ctx: impl AsContextMut<Engine>, additional: u32) -> anyhow::Result<u32> {
        todo!()
    }

    fn current_pages(&self, ctx: impl AsContext<Engine>) -> u32 {
        todo!()
    }

    fn read(
        &self,
        ctx: impl AsContext<Engine>,
        offset: usize,
        buffer: &mut [u8],
    ) -> anyhow::Result<()> {
        todo!()
    }

    fn write(
        &self,
        ctx: impl AsContextMut<Engine>,
        offset: usize,
        buffer: &[u8],
    ) -> anyhow::Result<()> {
        todo!()
    }
}

impl WasmModule<Engine> for Module {
    fn new(engine: &Engine, stream: impl std::io::Read) -> anyhow::Result<Self> {
        todo!()
    }

    fn exports(&self) -> Box<dyn '_ + Iterator<Item = crate::ExportType<'_>>> {
        todo!()
    }

    fn get_export(&self, name: &str) -> Option<crate::ExternType> {
        todo!()
    }

    fn imports(&self) -> Box<dyn '_ + Iterator<Item = crate::ImportType<'_>>> {
        todo!()
    }
}

/// A table of references
#[derive(Debug, Clone)]
pub struct Table {}

impl super::WasmTable<Engine> for Table {
    fn new(
        ctx: impl AsContextMut<Engine>,
        ty: TableType,
        init: Value<Engine>,
    ) -> anyhow::Result<Self> {
        todo!()
    }
    /// Returns the type and limits of the table.
    fn ty(&self, ctx: impl AsContext<Engine>) -> TableType {
        todo!()
    }
    /// Returns the current size of the table.
    fn size(&self, ctx: impl AsContext<Engine>) -> u32 {
        todo!()
    }
    /// Grows the table by the given amount of elements.
    fn grow(
        &self,
        ctx: impl AsContextMut<Engine>,
        delta: u32,
        init: Value<Engine>,
    ) -> anyhow::Result<u32> {
        todo!()
    }
    /// Returns the table element value at `index`.
    fn get(&self, ctx: impl AsContextMut<Engine>, index: u32) -> Option<Value<Engine>> {
        todo!()
    }
    /// Sets the value of this table at `index`.
    fn set(
        &self,
        ctx: impl AsContextMut<Engine>,
        index: u32,
        value: Value<Engine>,
    ) -> anyhow::Result<()> {
        todo!()
    }
}
