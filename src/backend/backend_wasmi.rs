use crate::backend::*;
use crate::ArgumentVec;
use std::sync::*;

impl WasmEngine for wasmi::Engine {
    type ExternRef = wasmi::ExternRef;

    type Func = wasmi::Func;

    type Global = wasmi::Global;

    type Instance = wasmi::Instance;

    type Memory = wasmi::Memory;

    type Module = Arc<wasmi::Module>;

    type Store<T> = wasmi::Store<T>;

    type StoreContext<'a, T: 'a> = wasmi::StoreContext<'a, T>;

    type StoreContextMut<'a, T: 'a> = wasmi::StoreContextMut<'a, T>;

    type Table = wasmi::Table;
}

impl WasmExternRef<wasmi::Engine> for wasmi::ExternRef {
    fn new<T: 'static + Send + Sync>(
        mut ctx: impl AsContextMut<wasmi::Engine>,
        object: Option<T>,
    ) -> Self {
        Self::new::<T>(ctx.as_context_mut(), object)
    }

    fn downcast<'a, T: 'static, S: 'a>(
        &self,
        store: <wasmi::Engine as WasmEngine>::StoreContext<'a, S>,
    ) -> Result<Option<&'a T>> {
        if let Some(data) = self.data(store) {
            data.downcast_ref()
                .ok_or_else(|| Error::msg("Incorrect extern ref type."))
                .map(Some)
        } else {
            Ok(None)
        }
    }
}

impl WasmFunc<wasmi::Engine> for wasmi::Func {
    fn new<T>(
        mut ctx: impl AsContextMut<wasmi::Engine, UserState = T>,
        ty: FuncType,
        func: impl 'static
            + Send
            + Sync
            + Fn(
                wasmi::StoreContextMut<T>,
                &[Value<wasmi::Engine>],
                &mut [Value<wasmi::Engine>],
            ) -> Result<()>,
    ) -> Self {
        wasmi::Func::new(
            ctx.as_context_mut(),
            ty.into(),
            move |mut caller, args, results| {
                let mut input = ArgumentVec::with_capacity(args.len());
                input.extend(args.iter().map(Into::into));

                let mut output = ArgumentVec::with_capacity(results.len());
                output.extend(results.iter().map(Into::into));

                func(
                    wasmi::AsContextMut::as_context_mut(&mut caller),
                    &input,
                    &mut output,
                )
                .map_err(HostError)?;

                for (i, result) in output.iter().enumerate() {
                    results[i] = result.into();
                }

                std::result::Result::Ok(())
            },
        )
    }

    fn ty(&self, ctx: impl AsContext<wasmi::Engine>) -> FuncType {
        self.ty(ctx.as_context()).into()
    }

    fn call<T>(
        &self,
        mut ctx: impl AsContextMut<wasmi::Engine>,
        args: &[Value<wasmi::Engine>],
        results: &mut [Value<wasmi::Engine>],
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

impl WasmGlobal<wasmi::Engine> for wasmi::Global {
    fn new(
        mut ctx: impl AsContextMut<wasmi::Engine>,
        value: Value<wasmi::Engine>,
        mutable: bool,
    ) -> Self {
        Self::new(
            ctx.as_context_mut(),
            (&value).into(),
            if mutable {
                wasmi::Mutability::Var
            } else {
                wasmi::Mutability::Const
            },
        )
    }

    fn ty(&self, ctx: impl AsContext<wasmi::Engine>) -> GlobalType {
        self.ty(ctx.as_context()).into()
    }

    fn set(
        &self,
        mut ctx: impl AsContextMut<wasmi::Engine>,
        new_value: Value<wasmi::Engine>,
    ) -> Result<()> {
        self.set(ctx.as_context_mut(), (&new_value).into())
            .map_err(Error::msg)
    }

    fn get(&self, ctx: impl AsContextMut<wasmi::Engine>) -> Value<wasmi::Engine> {
        (&self.get(ctx.as_context())).into()
    }
}

impl WasmInstance<wasmi::Engine> for wasmi::Instance {
    fn new(
        mut store: impl AsContextMut<wasmi::Engine>,
        module: &<wasmi::Engine as WasmEngine>::Module,
        imports: &Imports<wasmi::Engine>,
    ) -> Result<Self> {
        let mut linker = wasmi::Linker::new(store.as_context().engine());

        for ((module, name), imp) in imports {
            linker.define(&module, &name, imp)?;
        }

        let pre = linker.instantiate(store.as_context_mut(), module)?;
        Ok(pre.start(store.as_context_mut())?)
    }

    fn exports<'a>(
        &self,
        store: impl AsContext<wasmi::Engine>,
    ) -> Box<dyn Iterator<Item = backend::Export<wasmi::Engine>>> {
        Box::new(
            wasmi::Instance::exports(self, store.as_context())
                .map(|x| backend::Export {
                    name: x.name().to_string(),
                    value: x.into_extern().into(),
                })
                .collect::<Vec<_>>()
                .into_iter(),
        )
    }

    fn get_export(
        &self,
        store: impl AsContext<wasmi::Engine>,
        name: &str,
    ) -> Option<backend::Extern<wasmi::Engine>> {
        wasmi::Instance::get_export(self, store.as_context(), name).map(Into::into)
    }
}

impl WasmMemory<wasmi::Engine> for wasmi::Memory {
    fn new(mut ctx: impl AsContextMut<wasmi::Engine>, ty: MemoryType) -> Result<Self> {
        Self::new(ctx.as_context_mut(), ty.into()).map_err(Error::msg)
    }

    fn ty(&self, ctx: impl AsContext<wasmi::Engine>) -> MemoryType {
        self.ty(ctx.as_context()).into()
    }

    fn grow(&self, mut ctx: impl AsContextMut<wasmi::Engine>, additional: u32) -> Result<u32> {
        self.grow(
            ctx.as_context_mut(),
            wasmi::core::Pages::new(additional).context("Could not create additional pages.")?,
        )
        .map(Into::into)
        .map_err(Error::msg)
    }

    fn current_pages(&self, ctx: impl AsContext<wasmi::Engine>) -> u32 {
        self.current_pages(ctx.as_context()).into()
    }

    fn read(
        &self,
        ctx: impl AsContext<wasmi::Engine>,
        offset: usize,
        buffer: &mut [u8],
    ) -> Result<()> {
        self.read(ctx.as_context(), offset, buffer)
            .map_err(Error::msg)
    }

    fn write(
        &self,
        mut ctx: impl AsContextMut<wasmi::Engine>,
        offset: usize,
        buffer: &[u8],
    ) -> Result<()> {
        self.write(ctx.as_context_mut(), offset, buffer)
            .map_err(Error::msg)
    }
}

impl WasmModule<wasmi::Engine> for Arc<wasmi::Module> {
    fn new(engine: &wasmi::Engine, stream: impl std::io::Read) -> Result<Self> {
        Ok(Self::new(wasmi::Module::new(engine, stream)?))
    }

    fn exports(&self) -> Box<dyn '_ + Iterator<Item = ExportType<'_>>> {
        Box::new((**self).exports().map(|x| ExportType {
            name: x.name(),
            ty: x.ty().clone().into(),
        }))
    }

    fn get_export(&self, name: &str) -> Option<ExternType> {
        (**self).get_export(name).map(Into::into)
    }

    fn imports(&self) -> Box<dyn '_ + Iterator<Item = ImportType<'_>>> {
        Box::new((**self).imports().map(|x| ImportType {
            module: x.module(),
            name: x.name(),
            ty: x.ty().clone().into(),
        }))
    }
}

impl<T> WasmStore<T, wasmi::Engine> for wasmi::Store<T> {
    fn new(engine: &wasmi::Engine, data: T) -> Self {
        Self::new(engine, data)
    }

    fn engine(&self) -> &wasmi::Engine {
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

impl<'a, T> WasmStoreContext<'a, T, wasmi::Engine> for wasmi::StoreContext<'a, T> {
    fn engine(&self) -> &wasmi::Engine {
        self.engine()
    }

    fn data(&self) -> &T {
        self.data()
    }
}

impl<'a, T> AsContext<wasmi::Engine> for wasmi::StoreContext<'a, T> {
    type UserState = T;

    fn as_context(&self) -> wasmi::StoreContext<T> {
        wasmi::AsContext::as_context(self)
    }
}

impl<'a, T> AsContext<wasmi::Engine> for wasmi::StoreContextMut<'a, T> {
    type UserState = T;

    fn as_context(&self) -> wasmi::StoreContext<T> {
        wasmi::AsContext::as_context(self)
    }
}

impl<'a, T> AsContextMut<wasmi::Engine> for wasmi::StoreContextMut<'a, T> {
    fn as_context_mut(&mut self) -> wasmi::StoreContextMut<T> {
        wasmi::AsContextMut::as_context_mut(self)
    }
}

impl<'a, T> WasmStoreContext<'a, T, wasmi::Engine> for wasmi::StoreContextMut<'a, T> {
    fn engine(&self) -> &wasmi::Engine {
        self.engine()
    }

    fn data(&self) -> &T {
        self.data()
    }
}

impl<'a, T> WasmStoreContextMut<'a, T, wasmi::Engine> for wasmi::StoreContextMut<'a, T> {
    fn data_mut(&mut self) -> &mut T {
        self.data_mut()
    }
}

impl WasmTable<wasmi::Engine> for wasmi::Table {
    fn new(
        mut ctx: impl AsContextMut<wasmi::Engine>,
        ty: TableType,
        init: Value<wasmi::Engine>,
    ) -> Result<Self> {
        Self::new(ctx.as_context_mut(), ty.into(), (&init).into()).map_err(Error::msg)
    }

    fn ty(&self, ctx: impl AsContext<wasmi::Engine>) -> TableType {
        self.ty(ctx.as_context()).into()
    }

    fn size(&self, ctx: impl AsContext<wasmi::Engine>) -> u32 {
        self.size(ctx.as_context())
    }

    fn grow(
        &self,
        mut ctx: impl AsContextMut<wasmi::Engine>,
        delta: u32,
        init: Value<wasmi::Engine>,
    ) -> Result<u32> {
        self.grow(ctx.as_context_mut(), delta, (&init).into())
            .map_err(Error::msg)
    }

    fn get(
        &self,
        ctx: impl AsContextMut<wasmi::Engine>,
        index: u32,
    ) -> Option<Value<wasmi::Engine>> {
        self.get(ctx.as_context(), index).as_ref().map(Into::into)
    }

    fn set(
        &self,
        mut ctx: impl AsContextMut<wasmi::Engine>,
        index: u32,
        value: Value<wasmi::Engine>,
    ) -> Result<()> {
        self.set(ctx.as_context_mut(), index, (&value).into())
            .map_err(Error::msg)
    }
}

impl<T> AsContext<wasmi::Engine> for wasmi::Store<T> {
    type UserState = T;

    fn as_context(&self) -> <wasmi::Engine as WasmEngine>::StoreContext<'_, Self::UserState> {
        wasmi::AsContext::as_context(self)
    }
}

impl<T> AsContextMut<wasmi::Engine> for wasmi::Store<T> {
    fn as_context_mut(
        &mut self,
    ) -> <wasmi::Engine as WasmEngine>::StoreContextMut<'_, Self::UserState> {
        wasmi::AsContextMut::as_context_mut(self)
    }
}

impl From<wasmi::core::ValueType> for ValueType {
    fn from(value: wasmi::core::ValueType) -> Self {
        match value {
            wasmi::core::ValueType::I32 => Self::I32,
            wasmi::core::ValueType::I64 => Self::I64,
            wasmi::core::ValueType::F32 => Self::F32,
            wasmi::core::ValueType::F64 => Self::F64,
            wasmi::core::ValueType::FuncRef => Self::FuncRef,
            wasmi::core::ValueType::ExternRef => Self::ExternRef,
        }
    }
}

impl From<ValueType> for wasmi::core::ValueType {
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

impl From<&wasmi::Value> for Value<wasmi::Engine> {
    fn from(value: &wasmi::Value) -> Self {
        match value {
            wasmi::Value::I32(x) => Self::I32(*x),
            wasmi::Value::I64(x) => Self::I64(*x),
            wasmi::Value::F32(x) => Self::F32(x.to_float()),
            wasmi::Value::F64(x) => Self::F64(x.to_float()),
            wasmi::Value::FuncRef(x) => Self::FuncRef(x.func().copied()),
            wasmi::Value::ExternRef(x) => {
                Self::ExternRef(if x.is_null() { None } else { Some(*x) })
            }
        }
    }
}

impl From<&Value<wasmi::Engine>> for wasmi::Value {
    fn from(value: &Value<wasmi::Engine>) -> Self {
        match value {
            Value::I32(x) => Self::I32(*x),
            Value::I64(x) => Self::I64(*x),
            Value::F32(x) => Self::F32(wasmi::core::F32::from_float(*x)),
            Value::F64(x) => Self::F64(wasmi::core::F64::from_float(*x)),
            Value::FuncRef(x) => Self::FuncRef(wasmi::FuncRef::new(*x)),
            Value::ExternRef(x) => Self::ExternRef(x.unwrap_or_default()),
        }
    }
}

impl From<wasmi::FuncType> for FuncType {
    fn from(value: wasmi::FuncType) -> Self {
        Self::new(
            value.params().iter().map(|&x| x.into()),
            value.results().iter().map(|&x| x.into()),
        )
    }
}

impl From<FuncType> for wasmi::FuncType {
    fn from(value: FuncType) -> Self {
        Self::new(
            value.params().iter().map(|&x| x.into()),
            value.results().iter().map(|&x| x.into()),
        )
    }
}

impl From<wasmi::GlobalType> for GlobalType {
    fn from(value: wasmi::GlobalType) -> Self {
        Self::new(value.content().into(), value.mutability().is_mut())
    }
}

impl From<GlobalType> for wasmi::GlobalType {
    fn from(value: GlobalType) -> Self {
        Self::new(
            value.content().into(),
            if value.mutable() {
                wasmi::Mutability::Var
            } else {
                wasmi::Mutability::Const
            },
        )
    }
}

impl From<Extern<wasmi::Engine>> for wasmi::Extern {
    fn from(value: Extern<wasmi::Engine>) -> Self {
        match value {
            Extern::Func(x) => wasmi::Extern::Func(x),
            Extern::Global(x) => wasmi::Extern::Global(x),
            Extern::Memory(x) => wasmi::Extern::Memory(x),
            Extern::Table(x) => wasmi::Extern::Table(x),
        }
    }
}

impl From<wasmi::Extern> for Extern<wasmi::Engine> {
    fn from(value: wasmi::Extern) -> Self {
        match value {
            wasmi::Extern::Func(x) => Extern::Func(x),
            wasmi::Extern::Global(x) => Extern::Global(x),
            wasmi::Extern::Memory(x) => Extern::Memory(x),
            wasmi::Extern::Table(x) => Extern::Table(x),
        }
    }
}

impl From<wasmi::MemoryType> for MemoryType {
    fn from(value: wasmi::MemoryType) -> Self {
        Self::new(
            value.initial_pages().into(),
            value.maximum_pages().map(Into::into),
        )
    }
}

impl From<MemoryType> for wasmi::MemoryType {
    fn from(value: MemoryType) -> Self {
        Self::new(value.initial_pages(), value.maximum_pages()).expect("Could not convert memory.")
    }
}

impl From<wasmi::TableType> for TableType {
    fn from(value: wasmi::TableType) -> Self {
        Self::new(value.element().into(), value.minimum(), value.maximum())
    }
}

impl From<TableType> for wasmi::TableType {
    fn from(value: TableType) -> Self {
        Self::new(value.element().into(), value.minimum(), value.maximum())
    }
}

impl From<wasmi::ExternType> for ExternType {
    fn from(value: wasmi::ExternType) -> Self {
        match value {
            wasmi::ExternType::Func(x) => Self::Func(x.into()),
            wasmi::ExternType::Global(x) => Self::Global(x.into()),
            wasmi::ExternType::Memory(x) => Self::Memory(x.into()),
            wasmi::ExternType::Table(x) => Self::Table(x.into()),
        }
    }
}

impl From<ExternType> for wasmi::ExternType {
    fn from(value: ExternType) -> Self {
        match value {
            ExternType::Func(x) => Self::Func(x.into()),
            ExternType::Global(x) => Self::Global(x.into()),
            ExternType::Memory(x) => Self::Memory(x.into()),
            ExternType::Table(x) => Self::Table(x.into()),
        }
    }
}

/// Represents a `wasmi` error derived from `anyhow`.
#[derive(Debug)]
struct HostError(anyhow::Error);

impl std::fmt::Display for HostError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

impl wasmi::core::HostError for HostError {}
