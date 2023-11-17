use std::{
    collections::{BTreeMap, HashMap},
    sync::Arc,
};

use anyhow::{bail, Context};
use js_sys::{JsString, Object, Reflect, Uint8Array, WebAssembly};
use wasm_bindgen::{JsCast, JsValue};

use super::{
    AsContext, AsContextMut, Export, TableType, Value, WasmEngine, WasmExternRef, WasmInstance,
    WasmMemory, WasmModule,
};
use crate::{
    backend::Extern,
    web::{
        conversion::JsConvert, Engine, Func, Global, Import, Instance, InstanceInner, JsErrorMsg,
        Memory, Module, ModuleInner, Store, StoreContext, StoreContextMut, StoreInner, Table,
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

        let instance = {
            let mut engine = store.engine.borrow_mut();
            tracing::info!(?module.id, "get module");
            let module = &mut engine.modules[module.id];

            let imports_object = process_imports(store, imports);

            tracing::info!(?imports_object, "instantiate module");
            // TODO: async instantiation, possibly through a `.ready().await` call on the returned
            // module
            // let instance = WebAssembly::instantiate_module(&module.module, &imports);
            WebAssembly::Instance::new(&module.module, &imports_object)
                .map_err(JsErrorMsg::from)
                .with_context(|| format!("Failed to instantiate module"))?
        };

        tracing::info!(?instance, "created instance");

        let exports = Reflect::get(&instance, &"exports".into()).expect("exports object");
        let exports = process_exports(&mut store, exports)?;

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

fn process_imports<T>(store: &StoreInner<T>, imports: &super::Imports<Engine>) -> Object {
    let _span = tracing::info_span!("process_imports").entered();

    let imports = imports
        .into_iter()
        .map(|((module, name), imp)| {
            tracing::debug!(?module, ?name, ?imp, "import");
            let js = imp.to_js(store);

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

            let ext = match &value
                .js_typeof()
                .as_string()
                .expect("typeof returns a string")[..]
            {
                "function" => {
                    tracing::info!("function");
                    let func = Func::from_js(store, value).unwrap();

                    Extern::Func(func)
                }
                "object" => {
                    if value.is_instance_of::<js_sys::Function>() {
                        tracing::info!("function");
                        let func = Func::from_js(store, value).unwrap();

                        Extern::Func(func)
                    } else if value.is_instance_of::<WebAssembly::Table>() {
                        tracing::info!(?value, "export table");
                        let table = Table::from_js(store, value).unwrap();

                        Extern::Table(table)
                    } else if value.is_instance_of::<WebAssembly::Memory>() {
                        let memory = Memory::from_js(store, value).unwrap();

                        Extern::Memory(memory)
                    } else if value.is_instance_of::<WebAssembly::Global>() {
                        let global = Global::from_js(store, value).unwrap();

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
    fn new(engine: &Engine, mut stream: impl std::io::Read) -> anyhow::Result<Self> {
        let mut buf = Vec::new();
        stream
            .read_to_end(&mut buf)
            .context("Failed to read module bytes")?;

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
                    module: module.as_string().expect("module is string").into(),
                    name: name.as_string().expect("name is string").into(),
                    kind: ExternType::from_js_extern_kind(
                        kind.as_string().expect("kind is string").as_str(),
                    )
                    .expect("invalid kind"),
                }
            })
            .collect();

        let module = ModuleInner { module };

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
impl JsConvert for Value<Engine> {
    /// Convert the value enum to a JavaScript value
    fn to_js<T>(&self, store: &StoreInner<T>) -> JsValue {
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

    /// Convert from a JavaScript value.
    ///
    /// Returns `None` if the value can not be represented
    fn from_js<T>(store: &mut StoreInner<T>, value: JsValue) -> Option<Self> {
        tracing::info!(?value, "from_js_value");
        let ty = &*value
            .js_typeof()
            .as_string()
            .expect("typeof returns a string");

        let res = match ty {
            "number" => Value::F64(value.try_into().unwrap()),
            "bigint" => Value::I64(value.clone().try_into().unwrap()),
            "boolean" => Value::I32(value.as_bool().unwrap() as i32),
            "null" => Value::I32(0),
            "function" => {
                // TODO: this will not depuplicate function definitions
                let func = Func::from_js(store, value.clone()).unwrap();

                Value::FuncRef(Some(func))
            }
            // An instance of a WebAssembly.* class or null
            "object" => {
                if value.is_instance_of::<js_sys::Function>() {
                    tracing::info!("function");
                    if value.is_null() {
                        Value::FuncRef(None)
                    } else {
                        let func = Func::from_js(store, value.clone())
                            .expect("failed to convert to a function");

                        Value::FuncRef(Some(func))
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

    pub fn from_js_extern_kind(kind: &str) -> Option<Self> {
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
            "function" => Some(ExternType::Func(crate::FuncType {
                len_params: 0,
                params_results: Arc::from([]),
            })),
            _ => {
                tracing::error!(?kind, "unknown import kind");
                None
            }
        }
    }
}

impl JsConvert for Extern<Engine> {
    fn to_js<T>(&self, store: &StoreInner<T>) -> JsValue {
        let _span = tracing::info_span!("Extern::to_js", ?self).entered();

        match self {
            Extern::Global(v) => v.to_js(store),
            Extern::Table(v) => v.to_js(store),
            Extern::Memory(v) => v.to_js(store),
            Extern::Func(v) => v.to_js(store),
        }
    }

    fn from_js<T>(store: &mut StoreInner<T>, value: JsValue) -> Option<Self>
    where
        Self: Sized,
    {
        todo!()
    }
}

impl ValueType {
    pub(crate) fn to_js_descriptor(&self) -> JsValue {
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

impl JsConvert for ValueType {
    /// Convert the value enum to a JavaScript descriptor
    ///
    /// See: <https://developer.mozilla.org/en-US/docs/WebAssembly/JavaScript_interface/Global/Global>
    fn to_js<T>(&self, store: &StoreInner<T>) -> JsValue {
        self.to_js_descriptor()
    }

    fn from_js<T>(store: &mut StoreInner<T>, value: JsValue) -> Option<Self>
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
