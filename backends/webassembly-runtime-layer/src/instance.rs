use std::collections::BTreeMap;

use anyhow::{bail, Context};
use fxhash::FxHashMap;
use js_sys::{JsString, Object, Reflect, WebAssembly};
use wasm_bindgen::{JsCast, JsValue};

use wasm_runtime_layer::backend::{Export, Extern, Imports, WasmInstance};

use crate::{
    conversion::ToStoredJs, module::ParsedModule, Engine, Func, Global, JsErrorMsg, Memory, Module,
    StoreInner, Table,
};

/// A WebAssembly Instance.
#[derive(Debug, Clone)]
pub struct Instance {
    /// The id of the instance
    pub(crate) id: usize,
}

/// Holds the inner state of the instance
///
/// Not *Send* + *Sync*, as all other Js values.
#[derive(Debug)]
pub(crate) struct InstanceInner {
    /// The inner instance
    #[allow(dead_code)]
    pub(crate) instance: WebAssembly::Instance,
    /// The exports of the instance
    pub(crate) exports: FxHashMap<String, Extern<Engine>>,
}

impl WasmInstance<Engine> for Instance {
    fn new(
        mut store: impl super::AsContextMut<Engine>,
        module: &Module,
        imports: &Imports<Engine>,
    ) -> anyhow::Result<Self> {
        #[cfg(feature = "tracing")]
        let _span = tracing::debug_span!("Instance::new").entered();
        let store: &mut StoreInner<_> = &mut *store.as_context_mut();

        let instance;
        let parsed;
        let imports_object;

        {
            let mut engine = store.engine.borrow_mut();
            let module = &mut engine.modules[module.id];
            parsed = module.parsed.clone();

            imports_object = create_imports_object(store, imports);

            // TODO: async instantiation, possibly through a `.ready().await` call on the returned
            // module
            // let instance = WebAssembly::instantiate_module(&module.module, &imports);
            instance = WebAssembly::Instance::new(&module.module, &imports_object)
                .map_err(JsErrorMsg::from)
                .with_context(|| "Failed to instantiate module")?;
        };

        #[cfg(feature = "tracing")]
        let _span = tracing::debug_span!("get_exports").entered();

        let js_exports = Reflect::get(&instance, &"exports".into()).expect("exports object");
        let exports = process_exports(store, js_exports, &parsed)?;

        let instance = InstanceInner { instance, exports };

        let instance = store.insert_instance(instance);

        Ok(instance)
    }

    fn exports(
        &self,
        store: impl super::AsContext<Engine>,
    ) -> Box<dyn Iterator<Item = Export<Engine>>> {
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
    ) -> Option<Extern<Engine>> {
        let instance: &InstanceInner = &store.as_context().instances[self.id];
        instance.exports.get(name).cloned()
    }
}

/// Creates the js import map
fn create_imports_object<T>(store: &StoreInner<T>, imports: &Imports<Engine>) -> Object {
    #[cfg(feature = "tracing")]
    let _span = tracing::debug_span!("process_imports").entered();

    let imports = imports
        .into_iter()
        .map(|((module, name), imp)| {
            #[cfg(feature = "tracing")]
            tracing::trace!(?module, ?name, ?imp, "import");
            let js = imp.to_stored_js(store);

            #[cfg(feature = "tracing")]
            tracing::trace!(module, name, "export");

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
                Reflect::set(&obj, name, value).unwrap();
            }

            (module, obj)
        })
        .fold(Object::new(), |acc, (m, imports)| {
            Reflect::set(&acc, &(m).into(), &imports).unwrap();
            acc
        })
}

/// Processes a wasm module's exports into a hashmap
fn process_exports<T>(
    store: &mut StoreInner<T>,
    exports: JsValue,
    parsed: &ParsedModule,
) -> anyhow::Result<FxHashMap<String, Extern<Engine>>> {
    #[cfg(feature = "tracing")]
    let _span = tracing::debug_span!("process_exports", ?exports).entered();
    if !exports.is_object() {
        bail!(
            "WebAssembly exports must be an object, got '{:?}' instead",
            exports
        );
    }

    let exports: Object = exports.into();

    Object::entries(&exports)
        .into_iter()
        .map(|entry| {
            let name = Reflect::get_u32(&entry, 0)
                .unwrap()
                .as_string()
                .expect("name is string");

            let value: JsValue = Reflect::get_u32(&entry, 1).unwrap();

            #[cfg(feature = "tracing")]
            let _span = tracing::trace_span!("process_export", ?name, ?value).entered();

            let signature = parsed.exports.get(&name).expect("export signature").clone();

            let ext = match &value
                .js_typeof()
                .as_string()
                .expect("typeof returns a string")[..]
            {
                "function" => {
                    let func = Func::from_exported_function(
                        store,
                        value,
                        signature.try_into_func().unwrap(),
                    )
                    .unwrap();

                    Extern::Func(func)
                }
                "object" => {
                    if value.is_instance_of::<js_sys::Function>() {
                        let func = Func::from_exported_function(
                            store,
                            value,
                            signature.try_into_func().unwrap(),
                        )
                        .unwrap();

                        Extern::Func(func)
                    } else if value.is_instance_of::<WebAssembly::Table>() {
                        let table = Table::from_stored_js(
                            store,
                            value,
                            signature.try_into_table().unwrap(),
                        )
                        .unwrap();

                        Extern::Table(table)
                    } else if value.is_instance_of::<WebAssembly::Memory>() {
                        let memory = Memory::from_exported_memory(
                            store,
                            value,
                            signature.try_into_memory().unwrap(),
                        )
                        .unwrap();

                        Extern::Memory(memory)
                    } else if value.is_instance_of::<WebAssembly::Global>() {
                        let global = Global::from_exported_global(
                            store,
                            value,
                            signature.try_into_global().unwrap(),
                        )
                        .unwrap();

                        Extern::Global(global)
                    } else {
                        #[cfg(feature = "tracing")]
                        tracing::error!("Unsupported export type {value:?}");
                        panic!("Unsupported export type {value:?}")
                    }
                }
                _ => panic!("Unsupported export type {value:?}"),
            };

            Ok((name, ext))
        })
        .collect()
}
