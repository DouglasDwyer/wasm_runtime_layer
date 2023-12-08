use js_sys::JsString;
use wasm_bindgen::{JsCast, JsValue};

use super::{AsContextMut, Value, WasmEngine, WasmExternRef};
use crate::{
    backend::Extern,
    web::{
        conversion::{FromJs, FromStoredJs, ToJs, ToStoredJs},
        Engine, Func, Global, Instance, Memory, Module, Store, StoreContext, StoreContextMut,
        StoreInner, Table,
    },
    ValueType,
};

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

/// A table of references
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
                tracing::error!("conversion to a function outside of a module not permitted");
                return None;
            }
            // An instance of a WebAssembly.* class or null
            "object" => {
                if value.is_instance_of::<js_sys::Function>() {
                    tracing::error!("conversion to a function outside of a module not permitted");
                    return None;
                } else {
                    tracing::error!(?value, "Unsupported value type");
                    return None;
                }
            }
            v => {
                tracing::error!(?ty, ?v, "Unknown value primitive type");
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
    fn new<T: 'static + Send + Sync>(_: impl AsContextMut<Engine>, _: Option<T>) -> Self {
        unimplemented!("ExternRef is not supported in the web backend")
    }

    fn downcast<'a, T: 'static, S: 'a>(
        &self,
        _: <Engine as WasmEngine>::StoreContext<'a, S>,
    ) -> anyhow::Result<Option<&'a T>> {
        unimplemented!()
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

impl ValueType {
    /// Converts this type into the canonical ABI kind
    ///
    /// See: <https://webassembly.github.io/spec/js-api/#globals>
    pub(crate) fn as_js_descriptor(&self) -> &str {
        match self {
            Self::I32 => "i32",
            Self::I64 => "i64",
            Self::F32 => "f32",
            Self::F64 => "f64",
            Self::FuncRef => "anyfunc",
            Self::ExternRef => "externref",
        }
    }
}

impl ToJs for ValueType {
    type Repr = JsString;
    /// Convert the value enum to a JavaScript descriptor
    ///
    /// See: <https://developer.mozilla.org/en-US/docs/WebAssembly/JavaScript_interface/Global/Global>
    fn to_js(&self) -> JsString {
        self.as_js_descriptor().into()
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
                tracing::error!("Invalid value type {s:?}");
                return None;
            }
        };

        Some(res)
    }
}

impl Value<Engine> {
    /// Convert the JsValue into a Value of the supplied type
    pub(crate) fn from_js_typed<T>(
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
                tracing::error!(
                    "conversion to a function or extern outside of a module not permitted"
                );
                None
            }
        }
    }

    /// Convert a value to its type
    pub(crate) fn ty(&self) -> ValueType {
        match self {
            Value::I32(_) => ValueType::I32,
            Value::I64(_) => ValueType::I64,
            Value::F32(_) => ValueType::F32,
            Value::F64(_) => ValueType::F64,
            Value::FuncRef(_) => ValueType::FuncRef,
            Value::ExternRef(_) => ValueType::ExternRef,
        }
    }
}
