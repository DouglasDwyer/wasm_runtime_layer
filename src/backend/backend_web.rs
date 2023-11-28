use std::{
    collections::{BTreeMap, HashMap},
    sync::Arc,
};

use anyhow::{bail, Context};
use js_sys::{DataView, JsString, Object, Reflect, Uint8Array, WebAssembly};
use wasm_bindgen::{JsCast, JsValue};

use super::{
    AsContext, AsContextMut, Export, TableType, Value, WasmEngine, WasmExternRef, WasmInstance,
    WasmMemory, WasmModule,
};
use crate::{
    backend::Extern,
    web::{
        conversion::{FromJs, FromStoredJs, ToJs, ToStoredJs},
        module::{self, ParsedModule},
        Engine, Func, Global, Import, Instance, InstanceInner, JsErrorMsg, Memory, Module,
        ModuleInner, Store, StoreContext, StoreContextMut, StoreInner, Table,
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

impl WasmInstance<Engine> for Instance {
    fn new(
        mut store: impl super::AsContextMut<Engine>,
        module: &Module,
        imports: &super::Imports<Engine>,
    ) -> anyhow::Result<Self> {
        let _span = tracing::info_span!("Instance::new").entered();
        tracing::info!("Instance::new");
        let mut store: &mut StoreInner<_> = &mut *store.as_context_mut();

        let instance;
        let parsed;
        {
            let mut engine = store.engine.borrow_mut();
            tracing::info!(?module.id, "get module");
            let module = &mut engine.modules[module.id];
            parsed = module.parsed.clone();

            let imports_object = create_imports_object(store, imports);

            tracing::info!(?imports_object, "instantiate module");
            // TODO: async instantiation, possibly through a `.ready().await` call on the returned
            // module
            // let instance = WebAssembly::instantiate_module(&module.module, &imports);
            instance = WebAssembly::Instance::new(&module.module, &imports_object)
                .map_err(JsErrorMsg::from)
                .with_context(|| format!("Failed to instantiate module"))?;

            tracing::info!(?instance, "created instance");
        };

        let js_exports = Reflect::get(&instance, &"exports".into()).expect("exports object");
        let exports = process_exports(&mut store, js_exports, &parsed)?;

        let instance = InstanceInner { instance, exports };

        let instance = store.insert_instance(instance);

        Ok(instance)
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

fn create_imports_object<T>(store: &StoreInner<T>, imports: &super::Imports<Engine>) -> Object {
    let _span = tracing::info_span!("process_imports").entered();

    let imports = imports
        .into_iter()
        .map(|((module, name), imp)| {
            tracing::debug!(?module, ?name, ?imp, "import");
            let js = imp.to_stored_js(store);

            tracing::debug!(module, name, "export");

            (module, (JsString::from(&*name), js))
        })
        .fold(BTreeMap::<String, Vec<_>>::new(), |mut acc, (m, value)| {
            acc.entry(m).or_default().push(value);
            acc
        });

    imports
        .iter()
        .map(|(module, imports)| {
            let obj = Object::new();

            for (name, value) in imports {
                Reflect::set(&obj, &name, value).unwrap();
            }

            (module, obj)
        })
        .fold(Object::new(), |acc, (m, imports)| {
            Reflect::set(&acc, &(&*m).into(), &imports).unwrap();
            acc
        })
}

/// Processes a wasm module's exports into a rust side hashmap
fn process_exports<T>(
    store: &mut StoreInner<T>,
    exports: JsValue,
    parsed: &ParsedModule,
) -> anyhow::Result<HashMap<String, Extern<Engine>>> {
    let _span = tracing::info_span!("process_exports", ?exports).entered();
    if !exports.is_object() {
        bail!(
            "WebAssembly exports must be an object, got '{:?}' instead",
            exports
        );
    }

    let exports: Object = exports.into();
    let names = Object::get_own_property_names(&exports);
    let len = names.length();

    tracing::debug!(?names, ?exports);

    Object::entries(&exports)
        .into_iter()
        .map(|entry| {
            let name = Reflect::get_u32(&entry, 0)
                .unwrap()
                .as_string()
                .expect("name is string");

            let value: JsValue = Reflect::get_u32(&entry, 1).unwrap().into();

            let _span = tracing::info_span!("process_export", ?name, ?value).entered();

            let ty = value.js_typeof();

            let signature = parsed.exports.get(&name).expect("export signature").clone();

            let ext = match &value
                .js_typeof()
                .as_string()
                .expect("typeof returns a string")[..]
            {
                "function" => {
                    tracing::info!("function");

                    let func = Func::from_exported_function(
                        &name,
                        store,
                        value,
                        signature.try_into_func().unwrap(),
                    )
                    .unwrap();

                    Extern::Func(func)
                }
                "object" => {
                    if value.is_instance_of::<js_sys::Function>() {
                        tracing::info!("function");
                        let func = Func::from_exported_function(
                            &name,
                            store,
                            value,
                            signature.try_into_func().unwrap(),
                        )
                        .unwrap();

                        Extern::Func(func)
                    } else if value.is_instance_of::<WebAssembly::Table>() {
                        tracing::info!(?value, "export table");
                        let table = Table::from_stored_js(store, value).unwrap();

                        Extern::Table(table)
                    } else if value.is_instance_of::<WebAssembly::Memory>() {
                        let memory = Memory::from_stored_js(store, value).unwrap();

                        Extern::Memory(memory)
                    } else if value.is_instance_of::<WebAssembly::Global>() {
                        let global = Global::from_stored_js(store, value).unwrap();

                        Extern::Global(global)
                    } else {
                        tracing::error!("Unsupported export type {value:?}");
                        panic!("Unsupported export type {value:?}")
                    }
                }
                _ => panic!("Unsupported export type {value:?}"),
            };

            Ok((name, ext))
            // tracing::debug!(?name, ?value, ?ty, "entry");
        })
        .collect()
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
                tracing::info!(?import);
                let module = Reflect::get(&import, &"module".into()).unwrap();
                let name = Reflect::get(&import, &"name".into()).unwrap();
                let kind = Reflect::get(&import, &"kind".into()).unwrap();

                Import {
                    module: module.as_string().expect("module is string").into(),
                    kind: ExternType::from_import(
                        kind.as_string().expect("kind is string").as_str(),
                        name.as_string().expect("name is string").as_str(),
                        &parsed,
                    )
                    .expect("invalid kind"),
                    name: name.as_string().expect("name is string").into(),
                }
            })
            .collect();

        tracing::info!(?imports, "module imports");

        let module = ModuleInner {
            module,
            parsed: Arc::new(parsed),
        };

        let module = engine.borrow_mut().insert_module(module, imports);

        tracing::info!(?module, "created module");

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
        tracing::info!(?value, "from_js_value");
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
                    tracing::info!("function");
                    if value.is_null() {
                        Value::FuncRef(None)
                    } else {
                        tracing::error!(
                            "conversion to a function outside of a module not permitted"
                        );
                        return None;
                    }
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

    pub fn from_import(kind: &str, name: &str, parsed: &ParsedModule) -> Option<Self> {
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
        let _span = tracing::info_span!("Extern::to_stored_js", ?self).entered();

        match self {
            Extern::Global(v) => v.to_stored_js(store).into(),
            Extern::Table(v) => v.to_stored_js(store).into(),
            Extern::Memory(v) => v.to_stored_js(store).into(),
            Extern::Func(v) => v.to_stored_js(store).into(),
        }
    }
}

impl ValueType {
    pub(crate) fn to_js_descriptor(&self) -> &str {
        match self {
            Self::I32 => "i32",
            Self::I64 => "i64",
            Self::F32 => "f32",
            Self::F64 => "f64",
            Self::FuncRef => "anyfunc",
            Self::ExternRef => "externref",
        }
        .into()
    }
}

impl ToJs for ValueType {
    type Repr = JsString;
    /// Convert the value enum to a JavaScript descriptor
    ///
    /// See: <https://developer.mozilla.org/en-US/docs/WebAssembly/JavaScript_interface/Global/Global>
    fn to_js(&self) -> JsString {
        self.to_js_descriptor().into()
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
    pub(crate) fn from_js_typed<T>(
        store: &mut StoreInner<T>,
        ty: &ValueType,
        value: JsValue,
    ) -> Option<Value<Engine>> {
        match ty {
            ValueType::I32 => Some(Value::I32(i32::from_js(value)?)),
            ValueType::I64 => todo!(),
            ValueType::F32 => Some(Value::F32(f32::from_js(value)?)),
            ValueType::F64 => todo!(),
            ValueType::FuncRef => todo!(),
            ValueType::ExternRef => todo!(),
        }
    }
}
