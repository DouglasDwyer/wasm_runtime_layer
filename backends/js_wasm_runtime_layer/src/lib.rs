#![warn(missing_docs)]
#![warn(clippy::missing_docs_in_private_items)]
#![cfg_attr(not(feature = "std"), no_std)]

//! `js_wasm_runtime_layer` implements the `wasm_runtime_layer` abstraction interface over WebAssembly runtimes for your web browser's WebAssembly runtime.

extern crate alloc;

use alloc::{boxed::Box, format, rc::Rc, string::String, sync::Arc};
use core::{
    cell::{RefCell, RefMut},
    error::Error,
    fmt,
};

use js_sys::{JsString, Object, Reflect, WebAssembly};
use slab::Slab;
use wasm_bindgen::{JsCast, JsValue};
use wasm_runtime_layer::{
    backend::{AsContext, AsContextMut, Extern, Value, WasmEngine, WasmExternRef, WasmGlobal},
    GlobalType, ValueType,
};

/// Conversion to and from JavaScript
mod conversion;
/// Functions
mod func;
/// Instances
mod instance;
/// Memories
mod memory;
/// WebAssembly modules
mod module;
/// Stores all the WebAssembly state for a given collection of modules with a similar lifetime
mod store;
/// WebAssembly tables
mod table;

pub use func::Func;
pub use instance::Instance;
pub use memory::Memory;
pub use module::Module;
pub use store::{Store, StoreContext, StoreContextMut, StoreInner};
pub use table::Table;

use self::{
    conversion::{FromJs, FromStoredJs, ToJs, ToStoredJs},
    module::{ModuleInner, ParsedModule},
};

/// Helper to convert a `JsValue` into a proper error, as well as making it `Send` + `Sync`
#[derive(Debug, Clone)]
pub(crate) struct JsErrorMsg {
    /// A string representation of the error message
    message: String,
}

impl fmt::Display for JsErrorMsg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
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

/// Handle used to retain the lifetime of Js passed objects and drop them at an appropriate time.
///
/// Most commonly this is to ensure a closure with captures does not get dropped by Rust while a
/// reference to it exists in the world of Js.
#[derive(Debug)]
#[allow(dead_code)]
pub(crate) struct DropResource(Box<dyn fmt::Debug>);

impl DropResource {
    /// Creates a new drop resource from anything that implements `std::fmt::Debug`
    ///
    /// In general, any trait can be used here, but `std::fmt::Debug` is the most common and allows
    /// easy introspection of the values being held on to.
    pub fn new(value: impl 'static + fmt::Debug) -> Self {
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

        let ty = GlobalType::new(value_ty(&value), mutable);

        let desc = Object::new();

        Reflect::set(&desc, &"value".into(), &value_ty(&value).to_js()).unwrap();
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

        if !inner.ty.mutable() {
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

        value_from_js_typed(store, &ty.content(), value).unwrap()
    }
}

impl ToStoredJs for Value<Engine> {
    type Repr = JsValue;
    /// Convert the value enum to a JavaScript value
    fn to_stored_js<T>(&self, store: &StoreInner<T>) -> JsValue {
        match self {
            &Value::I32(v) => v.into(),
            &Value::I64(v) => v.into(),
            &Value::F32(v) => v.into(),
            &Value::F64(v) => v.into(),
            Value::FuncRef(Some(func)) => {
                let v: &JsValue = store.funcs[func.id].func.as_ref();
                v.clone()
            }
            Value::FuncRef(None) => JsValue::NULL,
            Value::ExternRef(_) => todo!(),
        }
    }
}

impl FromStoredJs for Value<Engine> {
    /// Convert from a JavaScript value.
    ///
    /// Returns `None` if the value can not be represented
    fn from_stored_js<T>(store: &mut StoreInner<T>, value: JsValue) -> Option<Self> {
        let ty = &*value
            .js_typeof()
            .as_string()
            .expect("typeof returns a string");

        let res = match ty {
            "number" => Value::F64(f64::from_stored_js(store, value).unwrap()),
            "bigint" => Value::I64(i64::from_stored_js(store, value).unwrap()),
            "boolean" => Value::I32(bool::from_stored_js(store, value).unwrap() as i32),
            "null" => Value::I32(0),
            "function" => {
                #[cfg(feature = "tracing")]
                tracing::error!("conversion to a function outside of a module not permitted");
                return None;
            }
            // An instance of a WebAssembly.* class or null
            "object" => {
                if value.is_instance_of::<js_sys::Function>() {
                    #[cfg(feature = "tracing")]
                    tracing::error!("conversion to a function outside of a module not permitted");
                    return None;
                } else {
                    #[cfg(feature = "tracing")]
                    tracing::error!(?value, "Unsupported value type");
                    return None;
                }
            }
            _ => {
                #[cfg(feature = "tracing")]
                tracing::error!(?ty, "Unknown value primitive type");
                return None;
            }
        };

        Some(res)
    }
}

#[derive(Debug, Clone)]
/// Extern host reference type
pub struct ExternRef {}

impl WasmExternRef<Engine> for ExternRef {
    fn new<T: 'static + Send + Sync>(_: impl AsContextMut<Engine>, _: T) -> Self {
        unimplemented!("ExternRef is not supported in the js_wasm_runtime_layer backend")
    }

    fn downcast<'a, 's: 'a, T: 'static, S: 's>(
        &self,
        _: <Engine as WasmEngine>::StoreContext<'s, S>,
    ) -> anyhow::Result<&'a T> {
        unimplemented!("ExternRef is not supported in the js_wasm_runtime_layer backend")
    }
}

impl ToStoredJs for Extern<Engine> {
    type Repr = JsValue;
    fn to_stored_js<T>(&self, store: &StoreInner<T>) -> JsValue {
        match self {
            Extern::Global(v) => v.to_stored_js(store).into(),
            Extern::Table(v) => v.to_stored_js(store).into(),
            Extern::Memory(v) => v.to_stored_js(store).into(),
            Extern::Func(v) => v.to_stored_js(store).into(),
        }
    }
}

impl ToJs for ValueType {
    type Repr = JsString;
    /// Convert the value enum to a JavaScript descriptor
    ///
    /// See: <https://developer.mozilla.org/en-US/docs/WebAssembly/JavaScript_interface/Global/Global>
    fn to_js(&self) -> JsString {
        match self {
            ValueType::I32 => "i32",
            ValueType::I64 => "i64",
            ValueType::F32 => "f32",
            ValueType::F64 => "f64",
            ValueType::FuncRef => "anyfunc",
            ValueType::ExternRef => "externref",
        }
        .into()
    }
}

impl FromJs for ValueType {
    fn from_js(value: JsValue) -> Option<Self>
    where
        Self: Sized,
    {
        let s = value.as_string()?;

        let res = match &s[..] {
            "i32" => Self::I32,
            "i64" => Self::I64,
            "f32" => Self::F32,
            "f64" => Self::F64,
            "anyfunc" => Self::FuncRef,
            "externref" => Self::ExternRef,
            _ => {
                #[cfg(feature = "tracing")]
                tracing::error!("Invalid value type {s:?}");
                return None;
            }
        };

        Some(res)
    }
}

/// Convert the JsValue into a Value of the supplied type
pub(crate) fn value_from_js_typed<T>(
    _: &mut StoreInner<T>,
    ty: &ValueType,
    value: JsValue,
) -> Option<Value<Engine>> {
    match ty {
        ValueType::I32 => Some(Value::I32(i32::from_js(value)?)),
        ValueType::I64 => Some(Value::I64(i64::from_js(value)?)),
        ValueType::F32 => Some(Value::F32(f32::from_js(value)?)),
        ValueType::F64 => Some(Value::F64(f64::from_js(value)?)),
        ValueType::FuncRef | ValueType::ExternRef => {
            #[cfg(feature = "tracing")]
            tracing::error!("conversion to a function or extern outside of a module not permitted");
            None
        }
    }
}

/// Convert a value to its type
fn value_ty(value: &Value<Engine>) -> ValueType {
    match value {
        Value::I32(_) => ValueType::I32,
        Value::I64(_) => ValueType::I64,
        Value::F32(_) => ValueType::F32,
        Value::F64(_) => ValueType::F64,
        Value::FuncRef(_) => ValueType::FuncRef,
        Value::ExternRef(_) => ValueType::ExternRef,
    }
}
