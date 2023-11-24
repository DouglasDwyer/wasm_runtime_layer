use crate::backend::*;
use crate::ArgumentVec;

impl WasmEngine for wasmtime::Engine {
    type ExternRef = wasmtime::ExternRef;

    type Func = wasmtime::Func;

    type Global = wasmtime::Global;

    type Instance = InstanceData;

    type Memory = wasmtime::Memory;

    type Module = wasmtime::Module;

    type Store<T> = wasmtime::Store<T>;

    type StoreContext<'a, T: 'a> = wasmtime::StoreContext<'a, T>;

    type StoreContextMut<'a, T: 'a> = wasmtime::StoreContextMut<'a, T>;

    type Table = wasmtime::Table;
}

impl WasmExternRef<wasmtime::Engine> for wasmtime::ExternRef {
    fn new<T: 'static + Send + Sync>(
        _: impl AsContextMut<wasmtime::Engine>,
        object: Option<T>,
    ) -> Self {
        Self::new::<Option<T>>(object)
    }

    fn downcast<'a, T: 'static, S: 'a>(
        &self,
        _: <wasmtime::Engine as WasmEngine>::StoreContext<'a, S>,
    ) -> Result<Option<&'a T>> {
        Ok(self
            .data()
            .downcast_ref::<&Option<T>>()
            .ok_or_else(|| Error::msg("Incorrect extern ref type."))?
            .as_ref())
    }
}

impl WasmFunc<wasmtime::Engine> for wasmtime::Func {
    fn new<T>(
        mut ctx: impl AsContextMut<wasmtime::Engine, UserState = T>,
        ty: FuncType,
        func: impl 'static
            + Send
            + Sync
            + Fn(
                wasmtime::StoreContextMut<T>,
                &[Value<wasmtime::Engine>],
                &mut [Value<wasmtime::Engine>],
            ) -> Result<()>,
    ) -> Self {
        tracing::info!(?ty, "Func::new");
        wasmtime::Func::new(
            ctx.as_context_mut(),
            ty.into(),
            move |mut caller, args, results| {
                let mut input = ArgumentVec::with_capacity(args.len());
                input.extend(args.iter().map(Into::into));

                let mut output = ArgumentVec::with_capacity(results.len());
                output.extend(results.iter().map(Into::into));

                func(
                    wasmtime::AsContextMut::as_context_mut(&mut caller),
                    &input,
                    &mut output,
                )?;

                for (i, result) in output.iter().enumerate() {
                    results[i] = result.into();
                }

                std::result::Result::Ok(())
            },
        )
    }

    fn ty(&self, ctx: impl AsContext<wasmtime::Engine>) -> FuncType {
        self.ty(ctx.as_context()).into()
    }

    fn call<T>(
        &self,
        mut ctx: impl AsContextMut<wasmtime::Engine>,
        args: &[Value<wasmtime::Engine>],
        results: &mut [Value<wasmtime::Engine>],
    ) -> Result<()> {
        let mut input = ArgumentVec::with_capacity(args.len());
        input.extend(args.iter().map(Into::into));

        let mut output = ArgumentVec::with_capacity(results.len());
        output.extend(results.iter().map(Into::into));

        self.call(ctx.as_context_mut(), &input[..], &mut output[..])?;

        for (i, result) in output.iter().enumerate() {
            results[i] = result.into();
        }

        Ok(())
    }
}

impl WasmGlobal<wasmtime::Engine> for wasmtime::Global {
    fn new(
        mut ctx: impl AsContextMut<wasmtime::Engine>,
        value: Value<wasmtime::Engine>,
        mutable: bool,
    ) -> Self {
        let value = wasmtime::Val::from(&value);
        Self::new(
            ctx.as_context_mut(),
            wasmtime::GlobalType::new(
                value.ty(),
                if mutable {
                    wasmtime::Mutability::Var
                } else {
                    wasmtime::Mutability::Const
                },
            ),
            value,
        )
        .expect("Could not create global.")
    }

    fn ty(&self, ctx: impl AsContext<wasmtime::Engine>) -> GlobalType {
        self.ty(ctx.as_context()).into()
    }

    fn set(
        &self,
        mut ctx: impl AsContextMut<wasmtime::Engine>,
        new_value: Value<wasmtime::Engine>,
    ) -> Result<()> {
        self.set(ctx.as_context_mut(), (&new_value).into())
            .map_err(Error::msg)
    }

    fn get(&self, mut ctx: impl AsContextMut<wasmtime::Engine>) -> Value<wasmtime::Engine> {
        (&self.get(ctx.as_context_mut())).into()
    }
}

/// Holds data about an instance for temporary immutable access.
#[derive(Clone, Debug)]
pub struct InstanceData {
    /// The instance itself.
    pub instance: wasmtime::Instance,
    /// The instance exports.
    pub exports: Arc<FxHashMap<String, backend::Export<wasmtime::Engine>>>,
}

impl WasmInstance<wasmtime::Engine> for InstanceData {
    fn new(
        mut store: impl AsContextMut<wasmtime::Engine>,
        module: &<wasmtime::Engine as WasmEngine>::Module,
        imports: &Imports<wasmtime::Engine>,
    ) -> Result<Self> {
        let mut linker = wasmtime::Linker::new(store.as_context().engine());

        for ((module, name), imp) in imports {
            linker.define(store.as_context(), &module, &name, imp)?;
        }

        let res = linker.instantiate(store.as_context_mut(), module)?;
        let exports = Arc::new(
            res.exports(store.as_context_mut())
                .map(|x| {
                    (
                        x.name().to_string(),
                        backend::Export {
                            name: x.name().to_string(),
                            value: x.into_extern().into(),
                        },
                    )
                })
                .collect(),
        );

        Ok(Self {
            instance: res,
            exports,
        })
    }

    fn exports<'a>(
        &self,
        _: impl AsContext<wasmtime::Engine>,
    ) -> Box<dyn Iterator<Item = backend::Export<wasmtime::Engine>>> {
        Box::new(
            self.exports
                .values()
                .cloned()
                .collect::<Vec<_>>()
                .into_iter(),
        )
    }

    fn get_export(
        &self,
        _: impl AsContext<wasmtime::Engine>,
        name: &str,
    ) -> Option<backend::Extern<wasmtime::Engine>> {
        self.exports.get(name).map(|x| x.value.clone())
    }
}

impl WasmMemory<wasmtime::Engine> for wasmtime::Memory {
    fn new(mut ctx: impl AsContextMut<wasmtime::Engine>, ty: MemoryType) -> Result<Self> {
        Self::new(ctx.as_context_mut(), ty.into()).map_err(Error::msg)
    }

    fn ty(&self, ctx: impl AsContext<wasmtime::Engine>) -> MemoryType {
        self.ty(ctx.as_context()).into()
    }

    fn grow(&self, mut ctx: impl AsContextMut<wasmtime::Engine>, additional: u32) -> Result<u32> {
        self.grow(ctx.as_context_mut(), additional as u64)
            .map(|x| x as u32)
            .map_err(Error::msg)
    }

    fn current_pages(&self, ctx: impl AsContext<wasmtime::Engine>) -> u32 {
        self.size(ctx.as_context()) as u32
    }

    fn read(
        &self,
        ctx: impl AsContext<wasmtime::Engine>,
        offset: usize,
        buffer: &mut [u8],
    ) -> Result<()> {
        self.read(ctx.as_context(), offset, buffer)
            .map_err(Error::msg)
    }

    fn write(
        &self,
        mut ctx: impl AsContextMut<wasmtime::Engine>,
        offset: usize,
        buffer: &[u8],
    ) -> Result<()> {
        self.write(ctx.as_context_mut(), offset, buffer)
            .map_err(Error::msg)
    }
}

impl WasmModule<wasmtime::Engine> for wasmtime::Module {
    fn new(engine: &wasmtime::Engine, mut stream: impl std::io::Read) -> Result<Self> {
        let mut buf = Vec::default();
        stream.read_to_end(&mut buf)?;
        Ok(wasmtime::Module::from_binary(engine, &buf)?)
    }

    fn exports(&self) -> Box<dyn '_ + Iterator<Item = ExportType<'_>>> {
        Box::new(self.exports().map(|x| ExportType {
            name: x.name(),
            ty: x.ty().into(),
        }))
    }

    fn get_export(&self, name: &str) -> Option<ExternType> {
        self.get_export(name).map(Into::into)
    }

    fn imports(&self) -> Box<dyn '_ + Iterator<Item = ImportType<'_>>> {
        Box::new(self.imports().map(|x| ImportType {
            module: x.module(),
            name: x.name(),
            ty: x.ty().into(),
        }))
    }
}

impl<T> WasmStore<T, wasmtime::Engine> for wasmtime::Store<T> {
    fn new(engine: &wasmtime::Engine, data: T) -> Self {
        Self::new(engine, data)
    }

    fn engine(&self) -> &wasmtime::Engine {
        self.engine()
    }

    fn data(&self) -> &T {
        self.data()
    }

    fn data_mut(&mut self) -> &mut T {
        self.data_mut()
    }

    fn into_data(self) -> T {
        self.into_data()
    }
}

impl<'a, T> WasmStoreContext<'a, T, wasmtime::Engine> for wasmtime::StoreContext<'a, T> {
    fn engine(&self) -> &wasmtime::Engine {
        self.engine()
    }

    fn data(&self) -> &T {
        self.data()
    }
}

impl<'a, T> AsContext<wasmtime::Engine> for wasmtime::StoreContext<'a, T> {
    type UserState = T;

    fn as_context(&self) -> wasmtime::StoreContext<T> {
        wasmtime::AsContext::as_context(self)
    }
}

impl<'a, T> AsContext<wasmtime::Engine> for wasmtime::StoreContextMut<'a, T> {
    type UserState = T;

    fn as_context(&self) -> wasmtime::StoreContext<T> {
        wasmtime::AsContext::as_context(self)
    }
}

impl<'a, T> AsContextMut<wasmtime::Engine> for wasmtime::StoreContextMut<'a, T> {
    fn as_context_mut(&mut self) -> wasmtime::StoreContextMut<T> {
        wasmtime::AsContextMut::as_context_mut(self)
    }
}

impl<'a, T> WasmStoreContext<'a, T, wasmtime::Engine> for wasmtime::StoreContextMut<'a, T> {
    fn engine(&self) -> &wasmtime::Engine {
        self.engine()
    }

    fn data(&self) -> &T {
        self.data()
    }
}

impl<'a, T> WasmStoreContextMut<'a, T, wasmtime::Engine> for wasmtime::StoreContextMut<'a, T> {
    fn data_mut(&mut self) -> &mut T {
        self.data_mut()
    }
}

impl WasmTable<wasmtime::Engine> for wasmtime::Table {
    fn new(
        mut ctx: impl AsContextMut<wasmtime::Engine>,
        ty: TableType,
        init: Value<wasmtime::Engine>,
    ) -> Result<Self> {
        Self::new(ctx.as_context_mut(), ty.into(), (&init).into()).map_err(Error::msg)
    }

    fn ty(&self, ctx: impl AsContext<wasmtime::Engine>) -> TableType {
        self.ty(ctx.as_context()).into()
    }

    fn size(&self, ctx: impl AsContext<wasmtime::Engine>) -> u32 {
        self.size(ctx.as_context())
    }

    fn grow(
        &self,
        mut ctx: impl AsContextMut<wasmtime::Engine>,
        delta: u32,
        init: Value<wasmtime::Engine>,
    ) -> Result<u32> {
        self.grow(ctx.as_context_mut(), delta, (&init).into())
            .map_err(Error::msg)
    }

    fn get(
        &self,
        mut ctx: impl AsContextMut<wasmtime::Engine>,
        index: u32,
    ) -> Option<Value<wasmtime::Engine>> {
        self.get(ctx.as_context_mut(), index)
            .as_ref()
            .map(Into::into)
    }

    fn set(
        &self,
        mut ctx: impl AsContextMut<wasmtime::Engine>,
        index: u32,
        value: Value<wasmtime::Engine>,
    ) -> Result<()> {
        self.set(ctx.as_context_mut(), index, (&value).into())
            .map_err(Error::msg)
    }
}

impl<T> AsContext<wasmtime::Engine> for wasmtime::Store<T> {
    type UserState = T;

    fn as_context(&self) -> <wasmtime::Engine as WasmEngine>::StoreContext<'_, Self::UserState> {
        wasmtime::AsContext::as_context(self)
    }
}

impl<T> AsContextMut<wasmtime::Engine> for wasmtime::Store<T> {
    fn as_context_mut(
        &mut self,
    ) -> <wasmtime::Engine as WasmEngine>::StoreContextMut<'_, Self::UserState> {
        wasmtime::AsContextMut::as_context_mut(self)
    }
}

impl From<wasmtime::ValType> for ValueType {
    fn from(value: wasmtime::ValType) -> Self {
        match value {
            wasmtime::ValType::I32 => Self::I32,
            wasmtime::ValType::I64 => Self::I64,
            wasmtime::ValType::F32 => Self::F32,
            wasmtime::ValType::F64 => Self::F64,
            wasmtime::ValType::FuncRef => Self::FuncRef,
            wasmtime::ValType::ExternRef => Self::ExternRef,
            _ => unimplemented!(),
        }
    }
}

impl From<ValueType> for wasmtime::ValType {
    fn from(value: ValueType) -> Self {
        match value {
            ValueType::I32 => Self::I32,
            ValueType::I64 => Self::I64,
            ValueType::F32 => Self::F32,
            ValueType::F64 => Self::F64,
            ValueType::FuncRef => Self::FuncRef,
            ValueType::ExternRef => Self::ExternRef,
        }
    }
}

impl From<&wasmtime::Val> for Value<wasmtime::Engine> {
    fn from(value: &wasmtime::Val) -> Self {
        match value {
            wasmtime::Val::I32(x) => Self::I32(*x),
            wasmtime::Val::I64(x) => Self::I64(*x),
            wasmtime::Val::F32(x) => Self::F32(f32::from_bits(*x)),
            wasmtime::Val::F64(x) => Self::F64(f64::from_bits(*x)),
            wasmtime::Val::FuncRef(x) => Self::FuncRef(*x),
            wasmtime::Val::ExternRef(x) => Self::ExternRef(x.clone()),
            _ => unimplemented!(),
        }
    }
}

impl From<&Value<wasmtime::Engine>> for wasmtime::Val {
    fn from(value: &Value<wasmtime::Engine>) -> Self {
        match value {
            Value::I32(x) => Self::I32(*x),
            Value::I64(x) => Self::I64(*x),
            Value::F32(x) => Self::F32(x.to_bits()),
            Value::F64(x) => Self::F64(x.to_bits()),
            Value::FuncRef(x) => Self::FuncRef(*x),
            Value::ExternRef(x) => Self::ExternRef(x.clone()),
        }
    }
}

impl From<wasmtime::FuncType> for FuncType {
    fn from(value: wasmtime::FuncType) -> Self {
        Self::new(
            value.params().map(Into::into),
            value.results().map(Into::into),
        )
    }
}

impl From<FuncType> for wasmtime::FuncType {
    fn from(value: FuncType) -> Self {
        Self::new(
            value.params().iter().map(|&x| x.into()),
            value.results().iter().map(|&x| x.into()),
        )
    }
}

impl From<wasmtime::GlobalType> for GlobalType {
    fn from(value: wasmtime::GlobalType) -> Self {
        Self::new(
            value.content().clone().into(),
            matches!(value.mutability(), wasmtime::Mutability::Var),
        )
    }
}

impl From<GlobalType> for wasmtime::GlobalType {
    fn from(value: GlobalType) -> Self {
        Self::new(
            value.content().into(),
            if value.mutable() {
                wasmtime::Mutability::Var
            } else {
                wasmtime::Mutability::Const
            },
        )
    }
}

impl From<Extern<wasmtime::Engine>> for wasmtime::Extern {
    fn from(value: Extern<wasmtime::Engine>) -> Self {
        match value {
            Extern::Func(x) => wasmtime::Extern::Func(x),
            Extern::Global(x) => wasmtime::Extern::Global(x),
            Extern::Memory(x) => wasmtime::Extern::Memory(x),
            Extern::Table(x) => wasmtime::Extern::Table(x),
        }
    }
}

impl From<wasmtime::Extern> for Extern<wasmtime::Engine> {
    fn from(value: wasmtime::Extern) -> Self {
        match value {
            wasmtime::Extern::Func(x) => Extern::Func(x),
            wasmtime::Extern::Global(x) => Extern::Global(x),
            wasmtime::Extern::Memory(x) => Extern::Memory(x),
            wasmtime::Extern::Table(x) => Extern::Table(x),
            _ => unimplemented!(),
        }
    }
}

impl From<wasmtime::MemoryType> for MemoryType {
    fn from(value: wasmtime::MemoryType) -> Self {
        Self::new(value.minimum() as u32, value.maximum().map(|x| x as u32))
    }
}

impl From<MemoryType> for wasmtime::MemoryType {
    fn from(value: MemoryType) -> Self {
        Self::new(value.initial_pages(), value.maximum_pages())
    }
}

impl From<wasmtime::TableType> for TableType {
    fn from(value: wasmtime::TableType) -> Self {
        Self::new(value.element().into(), value.minimum(), value.maximum())
    }
}

impl From<TableType> for wasmtime::TableType {
    fn from(value: TableType) -> Self {
        Self::new(value.element().into(), value.minimum(), value.maximum())
    }
}

impl From<wasmtime::ExternType> for ExternType {
    fn from(value: wasmtime::ExternType) -> Self {
        match value {
            wasmtime::ExternType::Func(x) => Self::Func(x.into()),
            wasmtime::ExternType::Global(x) => Self::Global(x.into()),
            wasmtime::ExternType::Memory(x) => Self::Memory(x.into()),
            wasmtime::ExternType::Table(x) => Self::Table(x.into()),
        }
    }
}

impl From<ExternType> for wasmtime::ExternType {
    fn from(value: ExternType) -> Self {
        match value {
            ExternType::Func(x) => Self::Func(x.into()),
            ExternType::Global(x) => Self::Global(x.into()),
            ExternType::Memory(x) => Self::Memory(x.into()),
            ExternType::Table(x) => Self::Table(x.into()),
        }
    }
}
