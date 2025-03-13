use alloc::{
    boxed::Box,
    string::{String, ToString},
    sync::Arc,
    vec::Vec,
};

use fxhash::FxHashMap;
use js_sys::{Uint8Array, WebAssembly};
use wasm_runtime_layer::{
    backend::WasmModule, ExportType, ExternType, FuncType, GlobalType, ImportType, MemoryType,
    TableType, ValueType,
};
use wasmparser::RefType;

use crate::{Engine, JsErrorMsg};

#[derive(Debug, Clone)]
/// A WebAssembly Module.
pub struct Module {
    /// The id of the module
    pub(crate) id: usize,
    /// The imports of the module
    pub(crate) parsed: Arc<ParsedModule>,
}

/// A WebAssembly Module.
#[derive(Debug)]
pub(crate) struct ModuleInner {
    /// The inner module
    pub(crate) module: js_sys::WebAssembly::Module,
    /// The parsed module, containing import and export signatures
    pub(crate) parsed: Arc<ParsedModule>,
}

impl WasmModule<Engine> for Module {
    fn new(engine: &Engine, bytes: &[u8]) -> anyhow::Result<Self> {
        let parsed = parse_module(bytes)?;

        let module =
            WebAssembly::Module::new(&Uint8Array::from(bytes).into()).map_err(JsErrorMsg::from)?;

        let parsed = Arc::new(parsed);

        let module = ModuleInner {
            module,
            parsed: parsed.clone(),
        };

        let module = engine.borrow_mut().insert_module(module, parsed);

        Ok(module)
    }

    fn exports(&self) -> Box<dyn '_ + Iterator<Item = ExportType<'_>>> {
        Box::new(self.parsed.exports.iter().map(|(name, ty)| ExportType {
            name: name.as_str(),
            ty: ty.clone(),
        }))
    }

    fn get_export(&self, name: &str) -> Option<ExternType> {
        self.parsed.exports.get(name).cloned()
    }

    fn imports(&self) -> Box<dyn '_ + Iterator<Item = ImportType<'_>>> {
        Box::new(
            self.parsed
                .imports
                .iter()
                .map(|((module, name), kind)| ImportType {
                    module,
                    name,
                    ty: kind.clone(),
                }),
        )
    }
}

/// Convert a [`wasmparser::ValType`] to a [`ValueType`].
fn value_type_from(ty: &wasmparser::ValType) -> ValueType {
    match ty {
        wasmparser::ValType::I32 => ValueType::I32,
        wasmparser::ValType::I64 => ValueType::I64,
        wasmparser::ValType::F32 => ValueType::F32,
        wasmparser::ValType::F64 => ValueType::F64,
        wasmparser::ValType::V128 => unimplemented!("v128 is not supported"),
        wasmparser::ValType::Ref(ty) => value_type_from_ref_type(ty),
    }
}

/// Convert a [`RefType`] to a [`ValueType`].
fn value_type_from_ref_type(ty: &RefType) -> ValueType {
    if ty.is_func_ref() {
        ValueType::FuncRef
    } else if ty.is_extern_ref() {
        ValueType::ExternRef
    } else {
        unimplemented!("unsupported reference type {ty:?}")
    }
}

/// Convert a [`wasmparser::TableType`] to a [`TableType`].
fn table_type_from(ty: &wasmparser::TableType) -> TableType {
    TableType::new(
        value_type_from_ref_type(&ty.element_type),
        ty.initial.try_into().expect("table size"),
        ty.maximum.map(|v| v.try_into().expect("table size")),
    )
}

#[derive(Debug)]
/// A parsed core module with imports and exports
pub(crate) struct ParsedModule {
    /// Import signatures
    pub(crate) imports: FxHashMap<(String, String), ExternType>,
    /// Export signatures
    pub(crate) exports: FxHashMap<String, ExternType>,
}

/// Parses a module from bytes and extracts import and export signatures
pub(crate) fn parse_module(bytes: &[u8]) -> anyhow::Result<ParsedModule> {
    let parser = wasmparser::Parser::new(0);

    let mut imports = FxHashMap::default();
    let mut exports = FxHashMap::default();

    let mut types = Vec::new();

    let mut functions = Vec::new();
    let mut memories = Vec::new();
    let mut tables = Vec::new();
    let mut globals = Vec::new();

    parser.parse_all(bytes).try_for_each(|payload| {
        match payload? {
            wasmparser::Payload::TypeSection(section) => {
                for ty in section {
                    let ty = ty?;

                    let mut subtypes = ty.types();
                    let subtype = subtypes.next();

                    let ty = match (subtype, subtypes.next()) {
                        (Some(subtype), None) => match &subtype.composite_type {
                            wasmparser::CompositeType {
                                inner: wasmparser::CompositeInnerType::Func(func_type),
                                shared: false,
                            } => FuncType::new(
                                func_type.params().iter().map(value_type_from),
                                func_type.results().iter().map(value_type_from),
                            ),
                            _ => unreachable!(),
                        },
                        _ => unimplemented!(),
                    };

                    types.push(ty);
                }
            }
            wasmparser::Payload::FunctionSection(section) => {
                for type_index in section {
                    let type_index = type_index?;

                    let ty = &types[type_index as usize];

                    functions.push(ty.clone());
                }
            }
            wasmparser::Payload::TableSection(section) => {
                for table in section {
                    let table = table?;

                    tables.push(table_type_from(&table.ty));
                }
            }
            wasmparser::Payload::MemorySection(section) => {
                for memory in section {
                    let memory = memory?;

                    memories.push(MemoryType::new(
                        memory.initial.try_into().expect("memory size"),
                        memory.maximum.map(|v| v.try_into().expect("memory size")),
                    ))
                }
            }
            wasmparser::Payload::GlobalSection(section) => {
                for global in section {
                    let global = global?;

                    let ty = value_type_from(&global.ty.content_type);
                    let mutable = global.ty.mutable;

                    globals.push(GlobalType::new(ty, mutable));
                }
            }
            wasmparser::Payload::TagSection(_section) =>
            {
                #[cfg(feature = "tracing")]
                for tag in _section {
                    let tag = tag?;

                    tracing::trace!(?tag, "tag");
                }
            }
            wasmparser::Payload::ImportSection(section) => {
                for import in section {
                    let import = import?;
                    let ty = match import.ty {
                        wasmparser::TypeRef::Func(index) => {
                            let sig = types[index as usize].clone().with_name(import.name);
                            functions.push(sig.clone());
                            ExternType::Func(sig)
                        }
                        wasmparser::TypeRef::Table(ty) => {
                            // functions.push(sig.clone());
                            tables.push(table_type_from(&ty));
                            ExternType::Table(table_type_from(&ty))
                        }
                        wasmparser::TypeRef::Memory(_) => todo!(),
                        wasmparser::TypeRef::Global(_) => todo!(),
                        wasmparser::TypeRef::Tag(_) => todo!(),
                    };

                    imports.insert((import.module.to_string(), import.name.to_string()), ty);
                }
            }
            wasmparser::Payload::ExportSection(section) => {
                for export in section {
                    let export = export?;
                    let index = export.index as usize;
                    let ty = match export.kind {
                        wasmparser::ExternalKind::Func => {
                            ExternType::Func(functions[index].clone().with_name(export.name))
                        }
                        wasmparser::ExternalKind::Table => ExternType::Table(tables[index]),
                        wasmparser::ExternalKind::Memory => ExternType::Memory(memories[index]),
                        wasmparser::ExternalKind::Global => ExternType::Global(globals[index]),
                        wasmparser::ExternalKind::Tag => todo!(),
                    };

                    exports.insert(export.name.to_string(), ty);
                }
            }
            wasmparser::Payload::ElementSection(_section) =>
            {
                #[cfg(feature = "tracing")]
                for element in _section {
                    let element = element?;
                    match element.kind {
                        wasmparser::ElementKind::Passive => tracing::debug!("passive"),
                        wasmparser::ElementKind::Active { .. } => tracing::debug!("active"),
                        wasmparser::ElementKind::Declared => tracing::debug!("declared"),
                    }
                }
            }
            _ => {}
        }

        anyhow::Ok(())
    })?;

    Ok(ParsedModule { imports, exports })
}
