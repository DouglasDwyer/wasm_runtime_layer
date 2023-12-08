use std::sync::Arc;

use anyhow::Context;
use js_sys::{JsString, Reflect, Uint8Array, WebAssembly};
use wasm_bindgen::{JsCast, JsValue};

use super::{AsContextMut, TableType, Value, WasmEngine, WasmExternRef, WasmModule};
use crate::{
    backend::Extern,
    web::{
        conversion::{FromJs, FromStoredJs, ToJs, ToStoredJs},
        module::{self, ParsedModule},
        Engine, Func, Global, Import, Instance, JsErrorMsg, Memory, Module, ModuleInner, Store,
        StoreContext, StoreContextMut, StoreInner, Table,
    },
    ExternType, GlobalType, ImportType, MemoryType, ValueType,
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

#[derive(Debug, Clone)]
/// Extern host reference type
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

impl WasmModule<Engine> for Module {
    fn new(engine: &Engine, mut stream: impl std::io::Read) -> anyhow::Result<Self> {
        let mut buf = Vec::new();
        stream
            .read_to_end(&mut buf)
            .context("Failed to read module bytes")?;

        let parsed = module::parse_module(&buf)?;

        let module = WebAssembly::Module::new(&Uint8Array::from(buf.as_slice()).into())
            .map_err(JsErrorMsg::from)?;

        let imports = WebAssembly::Module::imports(&module);

        // https://developer.mozilla.org/en-US/docs/WebAssembly/JavaScript_interface/Module/imports
        let imports: Vec<_> = imports
            .into_iter()
            .map(|import| {
                let module = Reflect::get(&import, &"module".into()).unwrap();
                let name = Reflect::get(&import, &"name".into()).unwrap();
                let kind = Reflect::get(&import, &"kind".into()).unwrap();

                Import {
                    module: module.as_string().expect("module is string"),
                    kind: ExternType::from_import(
                        kind.as_string().expect("kind is string").as_str(),
                        name.as_string().expect("name is string").as_str(),
                        &parsed,
                    )
                    .expect("invalid kind"),
                    name: name.as_string().expect("name is string"),
                }
            })
            .collect();

        let module = ModuleInner {
            module,
            parsed: Arc::new(parsed),
        };

        let module = engine.borrow_mut().insert_module(module, imports);

        Ok(module)
    }

    fn exports(&self) -> Box<dyn '_ + Iterator<Item = crate::ExportType<'_>>> {
        todo!()
    }

    fn get_export(&self, name: &str) -> Option<crate::ExternType> {
        todo!()
    }

    fn imports(&self) -> Box<dyn '_ + Iterator<Item = ImportType<'_>>> {
        Box::new(self.imports.iter().map(|v| ImportType {
            module: &v.module,
            name: &v.name,
            ty: v.kind.clone(),
        }))
    }
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
                tracing::error!(?ty, ?value, "Unknown value primitive type");
                return None;
            }
        };

        Some(res)
    }
}

impl ExternType {
    /// See: <https://webassembly.github.io/spec/js-api/#dom-moduleimportdescriptor-kind>
    pub fn to_js_extern_kind(&self) -> &str {
        match self {
            ExternType::Global(_) => "global",
            ExternType::Table(_) => "table",
            ExternType::Memory(_) => "memory",
            ExternType::Func(_) => "function",
        }
    }

    /// Creates an extern type from the given import.
    ///
    /// Uses the parsed module to infer the signature of an inferred function
    pub(crate) fn from_import(kind: &str, name: &str, parsed: &ParsedModule) -> Option<Self> {
        let signature = parsed.imports.get(name)?;

        match kind {
            "global" => Some(ExternType::Global(GlobalType {
                content: crate::ValueType::I32,
                mutable: true,
            })),
            "table" => Some(ExternType::Table(TableType {
                element: ValueType::I32,
                min: 0,
                max: None,
            })),
            "memory" => Some(ExternType::Memory(MemoryType {
                initial: 0,
                maximum: None,
            })),
            "function" => Some(ExternType::Func(signature.clone().try_into_func().ok()?)),
            _ => {
                tracing::error!(?kind, "unknown import kind");
                None
            }
        }
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
        store: &mut StoreInner<T>,
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
