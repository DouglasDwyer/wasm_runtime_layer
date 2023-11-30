use eyre::Context;
use js_sys::{Object, Reflect, WebAssembly};
use wasm_bindgen::{JsCast, JsValue};
use web_sys::console;

use crate::{
    backend::{AsContext, AsContextMut, Value, WasmTable},
    web::Func,
    TableType, ValueType,
};

use super::{
    conversion::{FromStoredJs, ToJs, ToStoredJs},
    Engine, JsErrorMsg, StoreContextMut, StoreInner,
};

#[derive(Debug, Clone)]
pub struct Table {
    pub(crate) id: usize,
}

pub(crate) struct TableInner {
    values: Vec<Value<Engine>>,
    ty: TableType,
}

impl std::fmt::Debug for TableInner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TableInner")
            .field("ty", &self.ty)
            .field("values", &self.values)
            .finish_non_exhaustive()
    }
}

impl FromStoredJs for Table {
    fn from_stored_js<T>(store: &mut StoreInner<T>, value: JsValue) -> Option<Self> {
        let _span = tracing::info_span!("Table::from_js", ?value).entered();
        console::log_1(&value);
        let table = value.dyn_ref::<WebAssembly::Table>()?;

        // let mut ty = crate::ValueType::FuncRef;

        let values: Vec<_> = (0..table.length())
            .map(|i| {
                let value = table.get(i).unwrap();

                // TODO: allow the api to accept table of ExternRef and not just Function.
                //
                // See: <https://github.com/rustwasm/wasm-bindgen/issues/3708>
                if value.is_null() {
                    Value::FuncRef(None)
                } else {
                    Value::FuncRef(Some(
                        todo!(),
                        // Func::from_stored_js(store, value.into())
                        //     .expect("table value is not a function"),
                    ))
                }
            })
            .collect();

        let inner = TableInner {
            ty: TableType {
                element: ValueType::FuncRef,
                min: values.len() as _,
                max: Some(values.len() as _),
            },
            values,
        };

        tracing::info!(?inner, "created table from js");

        Some(store.insert_table(inner))
    }
}

impl ToStoredJs for Table {
    type Repr = WebAssembly::Table;

    fn to_stored_js<T>(&self, store: &StoreInner<T>) -> WebAssembly::Table {
        let inner = &store.tables[self.id];

        assert_eq!(inner.ty.min, inner.values.len() as u32);
        let _span = tracing::info_span!("Table::to_stored_js", id=?self.id, ?inner).entered();
        let desc = Object::new();

        Reflect::set(&desc, &"element".into(), &inner.ty.element.to_js().into()).unwrap();

        Reflect::set(&desc, &"initial".into(), &inner.ty.min.into()).unwrap();

        if let Some(max) = inner.ty.max {
            Reflect::set(&desc, &"maximum".into(), &max.into()).unwrap();
        }

        tracing::info!(?inner, ?desc, "Table::to_js");
        let res = WebAssembly::Table::new(&desc)
            .map_err(JsErrorMsg::from)
            .context("Failed to create inner")
            .unwrap();

        for (i, value) in inner.values.iter().enumerate() {
            let value = value.to_stored_js(&store);
            res.set(
                i as u32,
                &value
                    // Unchecked here to allow null functions.
                    // `Table::set` should accept any JsValue according to MDN, but it currently
                    // only accepts `Function`.
                    //
                    // See: <https://github.com/rustwasm/wasm-bindgen/issues/3708>
                    .unchecked_into(),
            )
            .unwrap();
        }

        res
    }
}

impl WasmTable<Engine> for Table {
    fn new(
        mut ctx: impl AsContextMut<Engine>,
        ty: TableType,
        init: Value<Engine>,
    ) -> eyre::Result<Self> {
        let _span = tracing::info_span!("Table::new", ?ty, ?init).entered();
        let mut ctx: StoreContextMut<_> = ctx.as_context_mut();

        let table = TableInner {
            values: std::iter::repeat(init).take(ty.min as usize).collect(),
            ty,
        };

        let table = ctx.insert_table(table);

        Ok(table)
    }
    /// Returns the type and limits of the table.
    fn ty(&self, ctx: impl AsContext<Engine>) -> TableType {
        ctx.as_context().tables[self.id].ty
    }
    /// Returns the current size of the table.
    fn size(&self, ctx: impl AsContext<Engine>) -> u32 {
        ctx.as_context().tables[self.id].values.len() as _
    }
    /// Grows the table by the given amount of elements.
    fn grow(
        &self,
        mut ctx: impl AsContextMut<Engine>,
        delta: u32,
        init: Value<Engine>,
    ) -> eyre::Result<u32> {
        let ctx: &mut StoreInner<_> = &mut *ctx.as_context_mut();
        let table = &mut ctx.tables[self.id];

        let old_size = table.values.len() as _;
        table
            .values
            .extend(std::iter::repeat(init).take(delta as _));

        Ok(old_size)
    }
    /// Returns the table element value at `index`.
    fn get(&self, ctx: impl AsContextMut<Engine>, index: u32) -> Option<Value<Engine>> {
        let ctx: &StoreInner<_> = &*ctx.as_context();
        // RA breaks on this and sees the wrong impl of `value.get`
        //
        // Explicitely telling it that this is a slice of Value<Engine> causes it to see the
        // slice::get method rather than the WasmTable::get function, which shouldn't happen and is
        // a bug since &[]` does not implement `WasmTable`, but alas...
        let values: &[Value<Engine>] = &ctx.tables[self.id].values;

        values.get(index as usize).cloned()
    }
    /// Sets the value of this table at `index`.
    fn set(
        &self,
        mut ctx: impl AsContextMut<Engine>,
        index: u32,
        value: Value<Engine>,
    ) -> eyre::Result<()> {
        let ctx: &mut StoreInner<_> = &mut *ctx.as_context_mut();
        // RA breaks on this and sees the wrong impl of `value.get`
        //
        // Explicitely telling it that this is a slice of Value<Engine> causes it to see the
        // slice::get method rather than the WasmTable::get function, which shouldn't happen and is
        // a bug since &[]` does not implement `WasmTable`, but alas...
        let values: &mut [Value<Engine>] = &mut ctx.tables[self.id].values;

        *values
            .get_mut(index as usize)
            .ok_or_else(|| eyre::eyre!("invalid index"))? = value;

        Ok(())
    }
}
