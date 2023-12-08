/// Conversion to and from JavaScript
pub(crate) mod conversion;
/// Functions
pub(crate) mod func;
/// Instances
mod instance;
/// Memories
pub mod memory;
/// WebAssembly modules
pub(crate) mod module;
/// Stores all the WebAssembly state for a given collection of modules with a similar lifetime
mod store;
/// WebAssembly tables
pub(crate) mod table;
// mod element;

pub use func::Func;
pub use instance::Instance;
pub use memory::Memory;
pub use module::Module;
pub use store::{Store, StoreContext, StoreContextMut, StoreInner};
pub use table::Table;

use wasm_bindgen::{JsCast, JsValue};

use std::{
    cell::{RefCell, RefMut},
    error::Error,
    fmt::Display,
    rc::Rc,
    sync::Arc,
};

use slab::Slab;

use js_sys::{JsString, Object, Reflect, WebAssembly};

use crate::{
    backend::{AsContext, AsContextMut, Value, WasmGlobal},
    GlobalType,
};

use self::{
    conversion::ToStoredJs,
    module::{ModuleInner, ParsedModule},
};

/// Helper to convert a `JsValue` into a proper error, as well as making it `Send` + `Sync`
#[derive(Debug, Clone)]
pub(crate) struct JsErrorMsg {
    /// A string representation of the error message
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

/// Handle used to retain the lifetime of Js passed objects and drop them at an appropriate time.
///
/// Most commonly this is to ensure a closure with captures does not get dropped by Rust while a
/// reference to it exists in the world of Js.
#[derive(Debug)]
pub(crate) struct DropResource(Box<dyn std::fmt::Debug>);

impl DropResource {
    /// Creates a new drop resource from anything that implements `std::fmt::Debug`
    ///
    /// In general, any trait can be used here, but `std::fmt::Debug` is the most common and allows
    /// easy introspection of the values being held on to.
    pub fn new(value: impl 'static + std::fmt::Debug) -> Self {
        Self(Box::new(value))
    }
}

#[derive(Default, Debug, Clone)]
/// Runtime for WebAssembly
pub struct Engine {
    /// Inner state of the engine
    ///
    /// May be accessed at any time, but not recursively
    inner: Rc<RefCell<EngineInner>>,
}

impl Engine {
    // /// Borrow the engine
    // pub(crate) fn borrow(&self) -> Ref<EngineInner> {
    //     self.inner.borrow()
    // }

    /// Mutably borrow the engine
    pub(crate) fn borrow_mut(&self) -> RefMut<EngineInner> {
        self.inner.borrow_mut()
    }
}

/// Holds the inner mutable state of the engine
#[derive(Default, Debug)]
pub(crate) struct EngineInner {
    /// Modules loaded into the engine
    ///
    /// This is a slab since the WasmModule needs to be `Send`, but the WebAssembly::Module is not.
    /// The engine is not `Send` or `Sync` so they are stored here instead.
    pub(crate) modules: Slab<ModuleInner>,
}

impl EngineInner {
    /// Inserts a new module into the engine
    pub fn insert_module(&mut self, module: ModuleInner, parsed: Arc<ParsedModule>) -> Module {
        Module {
            id: self.modules.insert(module),
            parsed,
        }
    }
}

/// A global variable accesible as an import or export in a module
///
/// Stored within the store
#[derive(Debug, Clone)]
pub struct Global {
    /// The id of the global in the store
    pub(crate) id: usize,
}

/// Holds the inner state of the global
#[derive(Debug)]
pub(crate) struct GlobalInner {
    /// The global value
    value: WebAssembly::Global,
    /// The global type
    ty: GlobalType,
}

impl ToStoredJs for Global {
    type Repr = WebAssembly::Global;

    fn to_stored_js<T>(&self, store: &StoreInner<T>) -> WebAssembly::Global {
        let global = &store.globals[self.id];

        global.value.clone()
    }
}

impl Global {
    /// Creates a new global from a JS value
    pub(crate) fn from_exported_global<T>(
        store: &mut StoreInner<T>,
        value: JsValue,
        signature: GlobalType,
    ) -> Option<Self> {
        let global: &WebAssembly::Global = value.dyn_ref()?;

        Some(store.insert_global(GlobalInner {
            value: global.clone(),
            ty: signature,
        }))
    }
}

impl WasmGlobal<Engine> for Global {
    fn new(mut ctx: impl AsContextMut<Engine>, value: Value<Engine>, mutable: bool) -> Self {
        let mut ctx = ctx.as_context_mut();

        let ty = GlobalType::new(value.ty(), mutable);

        let desc = Object::new();

        Reflect::set(
            &desc,
            &"value".into(),
            &value.ty().as_js_descriptor().into(),
        )
        .unwrap();
        Reflect::set(&desc, &"mutable".into(), &mutable.into()).unwrap();

        let value = value.to_stored_js(&ctx);

        let global = GlobalInner {
            value: WebAssembly::Global::new(&desc, &value).unwrap(),
            ty,
        };

        ctx.insert_global(global)
    }

    fn ty(&self, ctx: impl AsContext<Engine>) -> GlobalType {
        ctx.as_context().globals[self.id].ty
    }

    fn set(
        &self,
        mut ctx: impl AsContextMut<Engine>,
        new_value: Value<Engine>,
    ) -> anyhow::Result<()> {
        let store: &mut StoreInner<_> = &mut ctx.as_context_mut();

        let value = &new_value.to_stored_js(store);

        let inner = &mut store.globals[self.id];

        if !inner.ty.mutable {
            return Err(anyhow::anyhow!("Global is not mutable"));
        }

        inner.value.set_value(value);

        Ok(())
    }

    fn get(&self, mut ctx: impl AsContextMut<Engine>) -> Value<Engine> {
        let store: &mut StoreInner<_> = &mut ctx.as_context_mut();
        let inner = &mut store.globals[self.id];

        let ty = inner.ty;
        let value = inner.value.value();

        Value::from_js_typed(store, &ty.content, value).unwrap()
    }
}
