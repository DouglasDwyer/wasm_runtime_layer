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
    table: WebAssembly::Table,
    // values: Vec<Value<Engine>>,
    ty: TableType,
}

impl std::fmt::Debug for TableInner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TableInner")
            .field("ty", &self.ty)
            .field("inner", &self.table)
            .finish_non_exhaustive()
    }
}

impl Table {
    pub(crate) fn from_stored_js<T>(
        store: &mut StoreInner<T>,
        value: JsValue,
        ty: TableType,
    ) -> Option<Self> {
        let _span = tracing::info_span!("Table::from_js", ?value).entered();
        console::log_1(&value);
        let table = value.dyn_into::<WebAssembly::Table>().ok()?;

        // let mut ty = crate::ValueType::FuncRef;
        assert!(table.length() >= ty.min);
        assert_eq!(ty.element, ValueType::FuncRef);

        // let values: Vec<_> = (0..table.length())
        //     .map(|i| {
        //         let value = table.get(i).unwrap();

        //         // TODO: allow the api to accept table of ExternRef and not just Function.
        //         //
        //         // See: <https://github.com/rustwasm/wasm-bindgen/issues/3708>
        //         if value.is_null() {
        //             Value::FuncRef(None)
        //         } else {
        //             Value::FuncRef(Some(
        //                 todo!(),
        //                 // Func::from_stored_js(store, value.into())
        //                 //     .expect("table value is not a function"),
        //             ))
        //         }
        //     })
        //     .collect();

        let inner = TableInner { ty, table };

        tracing::info!(?inner, "created table from js");

        Some(store.insert_table(inner))
    }
}

impl ToStoredJs for Table {
    type Repr = WebAssembly::Table;

    fn to_stored_js<T>(&self, store: &StoreInner<T>) -> WebAssembly::Table {
        let inner = &store.tables[self.id];
        inner.table.clone()
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

        let desc = Object::new();
        Reflect::set(
            &desc,
            &"element".into(),
            &ty.element.to_js_descriptor().into(),
        );
        Reflect::set(&desc, &"initial".into(), &ty.min.into());
        if let Some(max) = ty.max {
            Reflect::set(&desc, &"initial".into(), &max.into());
        }

        let table = WebAssembly::Table::new(&desc).map_err(JsErrorMsg::from)?;

        for i in 0..ty.min {
            table.set(i, init.to_stored_js(&ctx).unchecked_ref());
        }

        let table = TableInner {
            // values: std::iter::repeat(init).take(ty.min as usize).collect(),
            ty,
            table,
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
        ctx.as_context().tables[self.id].table.length()
    }
    /// Grows the table by the given amount of elements.
    fn grow(
        &self,
        mut ctx: impl AsContextMut<Engine>,
        delta: u32,
        init: Value<Engine>,
    ) -> eyre::Result<u32> {
        let ctx: &mut StoreInner<_> = &mut *ctx.as_context_mut();
        let init = init.to_stored_js(&ctx);
        let init = init.unchecked_ref();

        let inner = &mut ctx.tables[self.id];
        let old_len = inner.table.grow(delta).map_err(JsErrorMsg::from)?;

        for i in old_len..(old_len + delta) {
            inner.table.set(i, init);
        }

        // let old_size = table.values.len() as _;
        // table
        //     .values
        //     .extend(std::iter::repeat(init).take(delta as _));

        Ok(old_len)
    }
    /// Returns the table element value at `index`.
    fn get(&self, ctx: impl AsContextMut<Engine>, index: u32) -> Option<Value<Engine>> {
        let ctx: &StoreInner<_> = &*ctx.as_context();
        todo!()
        // RA breaks on this and sees the wrong impl of `value.get`
        //
        // Explicitely telling it that this is a slice of Value<Engine> causes it to see the
        // slice::get method rather than the WasmTable::get function, which shouldn't happen and is
        // a bug since &[]` does not implement `WasmTable`, but alas...
        // let values: &[Value<Engine>] = &ctx.tables[self.id].table;

        // values.get(index as usize).cloned()
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
        let value = value.to_stored_js(ctx);

        let inner: &mut TableInner = &mut ctx.tables[self.id];

        inner
            .table
            .set(index, value.unchecked_ref())
            .map_err(JsErrorMsg::from)
            .wrap_err("Invalid index")?;

        Ok(())
    }
}
