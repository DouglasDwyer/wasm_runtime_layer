use std::collections::HashMap;

use wasmparser::RefType;

use crate::{ExternType, FuncType, GlobalType, MemoryType, TableType, ValueType};

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
pub struct ParsedModule {
    pub imports: HashMap<String, ExternType>,
    pub exports: HashMap<String, ExternType>,
    types: Vec<FuncType>,
    functions: Vec<FuncType>,
    memories: Vec<MemoryType>,
    tables: Vec<TableType>,
    globals: Vec<GlobalType>,
}

pub fn fmt_items<T: std::fmt::Debug>(items: impl IntoIterator<Item = T>) -> String {
    items
        .into_iter()
        .enumerate()
        .map(|(i, v)| format!("{i:>4}: {v:?}"))
        .collect::<Vec<_>>()
        .join("\n")
}

pub fn parse_module(bytes: &[u8]) -> eyre::Result<ParsedModule> {
    tracing::debug!("parsing module\n{bytes:?}");
    let parser = wasmparser::Parser::new(0);

    let mut imports = HashMap::new();
    let mut exports = HashMap::new();

    let mut types = Vec::new();

    let mut functions = Vec::new();
    let mut memories = Vec::new();
    let mut tables = Vec::new();
    // let mut tags = Vec::new();
    let mut globals = Vec::new();
    // let mut elements = Vec::new();

    parser.parse_all(bytes).try_for_each(|payload| {
        match payload? {
            wasmparser::Payload::Version {
                num,
                encoding,
                range,
            } => {}
            wasmparser::Payload::TypeSection(section) => {
                for ty in section {
                    let ty = ty?;

                    let ty = match ty.types() {
                        [subtype] => match &subtype.composite_type {
                            wasmparser::CompositeType::Func(func_type) => {
                                FuncType::new(
                                    func_type.params().iter().map(ValueType::from),
                                    func_type.results().iter().map(ValueType::from),
                                )
                            }
                            _ => unreachable!(),
                        },
                        _ => unimplemented!(),
                    };

                    types.push(ty);
                }
                // tracing::info!("pushed {} types", types.len());
                tracing::debug!("\n{}", fmt_items(&types));
            }
            wasmparser::Payload::FunctionSection(section) => {
                for type_index in section {
                    let type_index = type_index?;

                    let ty = &types[type_index as usize];

                    functions.push(ty.clone());
                }

                // tracing::info!("pushed {} functions", functions.len());
                tracing::debug!("\n{}", fmt_items(&functions));
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
            wasmparser::Payload::TagSection(section) => {
                for tag in section {
                    let tag = tag?;

                    tracing::info!(?tag, "tag");
                }
            }
            wasmparser::Payload::ImportSection(section) => {
                for import in section {
                    let import = import?;
                    let ty = match import.ty {
                        wasmparser::TypeRef::Func(index) => {
                            let sig = types[index as usize].clone();
                            functions.push(sig.clone());
                            ExternType::Func(sig)
                        }
                        wasmparser::TypeRef::Table(ty) => {
                            // functions.push(sig.clone());
                            tables.push((&ty).into());
                            ExternType::Table((&ty).into())
                        },
                        wasmparser::TypeRef::Memory(_) => todo!(),
                        wasmparser::TypeRef::Global(_) => todo!(),
                        wasmparser::TypeRef::Tag(_) => todo!(),
                    };

                    // tracing::info!(module = import.module, name = import.name, ?ty, "imports");
                    imports.insert(import.name.to_string(), ty);
                }
                tracing::debug!("imports: \n{}", fmt_items(&imports));
            }
            wasmparser::Payload::ExportSection(section) => {
                for export in section {
                    let export = export?;
                    let index = export.index as usize;
                    let ty = match export.kind {
                        wasmparser::ExternalKind::Func => {
                            tracing::info!(?export.name, ?index, f=?functions[index], "found exported function index");
                            ExternType::Func(functions[index].clone())
                        }
                        wasmparser::ExternalKind::Table => ExternType::Table(tables[index]),
                        wasmparser::ExternalKind::Memory => ExternType::Memory(memories[index]),
                        wasmparser::ExternalKind::Global => ExternType::Global(globals[index]),
                        wasmparser::ExternalKind::Tag => todo!(),
                    };

                    // tracing::info!(name = export.name, ?ty, "export");
                    exports.insert(export.name.to_string(), ty);
                }
                tracing::debug!("exports \n{}", fmt_items(&exports));
            }
            wasmparser::Payload::StartSection { func, range } => {}
            wasmparser::Payload::ElementSection(section) => {
                for (element) in section {
                    let element  = element?;
                    match element.kind {
                        wasmparser::ElementKind::Passive => tracing::debug!("passive"),
                        wasmparser::ElementKind::Active { table_index, offset_expr } => tracing::debug!("active" ),
                        wasmparser::ElementKind::Declared => tracing::debug!("declared"),
                    }

                    // tracing::debug!("element");
                    // match element.items {
                    //     wasmparser::ElementItems::Functions(v) => {
                    //         for v in v {}

                    //         todo!("element functions")
                    //     },
                    //     wasmparser::ElementItems::Expressions(_, _) => todo!("element expr"),
                    // }
                }
            }
            wasmparser::Payload::DataCountSection { count, range } => {}
            wasmparser::Payload::DataSection(_) => {}
            wasmparser::Payload::CodeSectionStart { count, range, size } => {}
            wasmparser::Payload::CodeSectionEntry(_) => {}
            wasmparser::Payload::ModuleSection { parser, range } => {}
            wasmparser::Payload::InstanceSection(_) => {}
            wasmparser::Payload::CoreTypeSection(_) => {}
            wasmparser::Payload::ComponentSection { parser, range } => {}
            wasmparser::Payload::ComponentInstanceSection(_) => {}
            wasmparser::Payload::ComponentAliasSection(_) => {}
            wasmparser::Payload::ComponentTypeSection(_) => {}
            wasmparser::Payload::ComponentCanonicalSection(_) => {}
            wasmparser::Payload::ComponentStartSection { start, range } => {}
            wasmparser::Payload::ComponentImportSection(_) => {}
            wasmparser::Payload::ComponentExportSection(_) => {}
            wasmparser::Payload::CustomSection(_) => {}
            wasmparser::Payload::UnknownSection {
                id,
                contents,
                range,
            } => {}
            wasmparser::Payload::End(_) => {}
        }

        Ok(()) as eyre::Result<_>
    })?;

    tracing::info!(?imports, ?exports, "imports");

    Ok(ParsedModule {
        imports,
        exports,
        types,
        functions,
        memories,
        tables,
        globals,
    })
}
