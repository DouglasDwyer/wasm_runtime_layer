use std::{collections::HashMap, error::Error, fmt::Display};

use anyhow::bail;
use js_sys::{JsString, Object, Reflect, WebAssembly};
use slab::Slab;
use wasm_bindgen::{JsCast, JsValue};

use super::{
    AsContext, AsContextMut, WasmEngine, WasmInstance, WasmStore, WasmStoreContext,
    WasmStoreContextMut,
};

#[derive(Debug, Clone)]
pub struct JsErrorMsg {
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

#[derive(Debug, Clone)]
pub struct Engine {}

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

#[derive(Debug, Clone)]
struct Instance {
    id: usize,
}

impl WasmInstance<Engine> for Instance {
    fn new(
        store: impl super::AsContextMut<Engine>,
        module: &Module,
        imports: &super::Imports<Engine>,
    ) -> anyhow::Result<Self> {
        let store = store.as_context_mut();

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
        let exports = process_exports(exports)?;

        let instance = InstanceInner { instance, exports };

        let instance_id = store;
        todo!();
    }

    fn exports(
        &self,
        store: impl super::AsContext<Engine>,
    ) -> Box<dyn Iterator<Item = super::Export<Engine>>> {
        todo!()
    }

    fn get_export(
        &self,
        store: impl super::AsContext<Engine>,
        name: &str,
    ) -> Option<super::Extern<Engine>> {
        todo!()
    }
}

fn process_exports(js_exports: JsValue) -> anyhow::Result<HashMap<String, JsValue>> {
    if !js_exports.is_object() {
        bail!(
            "WebAssembly exports must be an object, got '{:?}' instead",
            js_exports
        );
    }

    // TODO: this is duplicated somewhere, but here ...
    let js_exports: Object = js_exports.into();
    let names = Object::get_own_property_names(&js_exports);
    let len = names.length();

    let mut exports = HashMap::new();
    for i in 0..len {
        let name_js = Reflect::get_u32(&names, i).expect("names is array");
        let name = name_js.as_string().expect("name is string");
        let export = Reflect::get(&js_exports, &name_js).expect("js_exports is object");
        exports.insert(name, export);
    }
    Ok(exports)
}

struct ExternRef {}

struct Func {}

struct Global {}

struct Memory {}

struct Module {
    module: js_sys::WebAssembly::Module,
    exports: js_sys::Object,
}

/// Not Send + Sync
struct InstanceInner {
    instance: WebAssembly::Instance,
    exports: HashMap<String, JsValue>,
}

struct Store<T> {
    engine: Engine,
    instances: Slab<InstanceInner>,
    data: T,
}

impl<T> WasmStore<T, Engine> for Store<T> {
    fn new(engine: &Engine, data: T) -> Self {
        Self {
            engine: engine.clone(),
            instances: Slab::new(),
            data,
        }
    }

    fn engine(&self) -> &Engine {
        &self.engine
    }

    fn data(&self) -> &T {
        &self.data
    }

    fn data_mut(&mut self) -> &mut T {
        &mut self.data
    }

    fn into_data(self) -> T {
        todo!()
    }
}

impl<T> AsContext<Engine> for Store<T> {
    type UserState = T;

    fn as_context(&self) -> <Engine as WasmEngine>::StoreContext<'_, Self::UserState> {
        StoreContext { store: self }
    }
}

impl<T> AsContextMut<Engine> for Store<T> {
    fn as_context_mut(&mut self) -> StoreContextMut<T> {
        StoreContextMut { store: self }
    }
}

struct StoreContext<'a, T: 'a> {
    store: &'a Store<T>,
}

impl<'a, T: 'a> WasmStoreContext<'a, T, Engine> for StoreContext<'a, T> {
    fn engine(&self) -> &Engine {
        &self.store.engine
    }

    fn data(&self) -> &T {
        &self.store.data
    }
}

impl<'a, T: 'a> AsContext<Engine> for StoreContext<'a, T> {
    type UserState = T;

    fn as_context(&self) -> StoreContext<'a, T> {
        StoreContext { store: self.store }
    }
}

struct StoreContextMut<'a, T: 'a> {
    store: &'a mut Store<T>,
}

impl<'a, T: 'a> WasmStoreContext<'a, T, Engine> for StoreContextMut<'a, T> {
    fn engine(&self) -> &Engine {
        &self.store.engine
    }

    fn data(&self) -> &T {
        &self.store.data
    }
}

impl<'a, T: 'a> WasmStoreContextMut<'a, T, Engine> for StoreContextMut<'a, T> {
    fn data_mut(&mut self) -> &mut T {
        &mut self.store.data
    }
}

impl<'a, T: 'a> AsContext<Engine> for StoreContextMut<'a, T> {
    type UserState = T;

    fn as_context(&self) -> <Engine as WasmEngine>::StoreContext<'_, Self::UserState> {
        StoreContext { store: self.store }
    }
}

impl<'a, T: 'a> AsContextMut<Engine> for StoreContextMut<'a, T> {
    fn as_context_mut(&mut self) -> StoreContextMut<'a, T> {
        StoreContextMut { store: self.store }
    }
}

struct Table {}
