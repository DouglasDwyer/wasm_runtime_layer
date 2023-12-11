use std::sync::Arc;

use anyhow::Context;
use fxhash::FxHashMap;
use js_sys::{Uint8Array, WebAssembly};
use wasmparser::RefType;

use crate::{
    backend::WasmModule, ExternType, FuncType, GlobalType, ImportType, MemoryType, TableType,
    ValueType,
};

use super::{Engine, JsErrorMsg};

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
    fn new(engine: &Engine, mut stream: impl std::io::Read) -> anyhow::Result<Self> {
        let mut buf = Vec::new();
        stream
            .read_to_end(&mut buf)
            .context("Failed to read module bytes")?;

        let parsed = parse_module(&buf)?;

        let module = WebAssembly::Module::new(&Uint8Array::from(buf.as_slice()).into())
            .map_err(JsErrorMsg::from)?;

        let parsed = Arc::new(parsed);

        let module = ModuleInner {
            module,
            parsed: parsed.clone(),
        };

        let module = engine.borrow_mut().insert_module(module, parsed);

        Ok(module)
    }

    fn exports(&self) -> Box<dyn '_ + Iterator<Item = crate::ExportType<'_>>> {
        Box::new(
            self.parsed
                .exports
                .iter()
                .map(|(name, ty)| crate::ExportType {
                    name: name.as_str(),
                    ty: ty.clone(),
                }),
        )
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

impl From<&wasmparser::ValType> for ValueType {
    fn from(value: &wasmparser::ValType) -> Self {
        match value {
            wasmparser::ValType::I32 => Self::I32,
            wasmparser::ValType::I64 => Self::I64,
            wasmparser::ValType::F32 => Self::F32,
            wasmparser::ValType::F64 => Self::F64,
            wasmparser::ValType::V128 => unimplemented!("v128 is not supported"),
            wasmparser::ValType::Ref(ty) => ty.into(),
        }
    }
}

impl From<&RefType> for ValueType {
    fn from(value: &RefType) -> Self {
        if value.is_func_ref() {
            Self::FuncRef
        } else if value.is_extern_ref() {
            Self::ExternRef
        } else {
            unimplemented!("unsupported reference type {value:?}")
        }
    }
}

impl From<&wasmparser::TableType> for TableType {
    fn from(value: &wasmparser::TableType) -> Self {
        TableType {
            element: (&value.element_type).into(),
            min: value.initial,
            max: value.maximum,
        }
    }
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
            wasmparser::Payload::Version { .. } => {}
            wasmparser::Payload::TypeSection(section) => {
                for ty in section {
                    let ty = ty?;

                    let ty = match ty.types() {
                        [subtype] => match &subtype.composite_type {
                            wasmparser::CompositeType::Func(func_type) => FuncType::new(
                                func_type.params().iter().map(ValueType::from),
                                func_type.results().iter().map(ValueType::from),
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

                    tables.push((&table.ty).into());
                }
            }
            wasmparser::Payload::MemorySection(section) => {
                for memory in section {
                    let memory = memory?;

                    memories.push(MemoryType {
                        initial: memory.initial.try_into().expect("memory size"),
                        maximum: memory.maximum.map(|v| v.try_into().expect("memory size")),
                    })
                }
            }
            wasmparser::Payload::GlobalSection(section) => {
                for global in section {
                    let global = global?;

                    let ty: ValueType = (&global.ty.content_type).into();
                    let mutable = global.ty.mutable;

                    globals.push(GlobalType {
                        content: ty,
                        mutable,
                    });
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
                            tables.push((&ty).into());
                            ExternType::Table((&ty).into())
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
            wasmparser::Payload::StartSection { .. } => {}
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
            wasmparser::Payload::DataCountSection { .. } => {}
            wasmparser::Payload::DataSection(_) => {}
            wasmparser::Payload::CodeSectionStart { .. } => {}
            wasmparser::Payload::CodeSectionEntry(_) => {}
            wasmparser::Payload::ModuleSection { .. } => {}
            wasmparser::Payload::InstanceSection(_) => {}
            wasmparser::Payload::CoreTypeSection(_) => {}
            wasmparser::Payload::ComponentSection { .. } => {}
            wasmparser::Payload::ComponentInstanceSection(_) => {}
            wasmparser::Payload::ComponentAliasSection(_) => {}
            wasmparser::Payload::ComponentTypeSection(_) => {}
            wasmparser::Payload::ComponentCanonicalSection(_) => {}
            wasmparser::Payload::ComponentStartSection { .. } => {}
            wasmparser::Payload::ComponentImportSection(_) => {}
            wasmparser::Payload::ComponentExportSection(_) => {}
            wasmparser::Payload::CustomSection(_) => {}
            wasmparser::Payload::UnknownSection { .. } => {}
            wasmparser::Payload::End(_) => {}
        }

        anyhow::Ok(())
    })?;

    Ok(ParsedModule { imports, exports })
}
