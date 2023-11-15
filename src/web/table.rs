use std::{io::repeat, ops::Deref};

use anyhow::Context;
use js_sys::{Function, Object, Reflect, WebAssembly};
use wasm_bindgen::{JsCast, JsValue};

use crate::{
    backend::{AsContext, AsContextMut, Value, WasmModule, WasmTable},
    TableType, ValueType,
};

use super::{Engine, JsErrorMsg, StoreContextMut, StoreInner};

#[derive(Debug, Clone)]
pub struct Table {
    pub(crate) id: usize,
}

pub(crate) struct TableInner {
    /// NOTE: these two needs to be kept in sync
    ///
    /// When getting a value from the table we can not naively get it from the js side table, as we
    /// thus need to allocate a new store entry to keep it in rust.
    ///
    /// As such, every value in the table needs to have a equivalent Rust side counterpart that can
    /// be returned and not duplicated.
    ///
    /// We could also solve this by interning all Js functions, memories, and other references when
    /// creating this in the store to ensure that we do not create multiple instances in the slab
    /// referencing the same resource, causing unbounded memory growth. This however, is more
    /// difficult as we can not reliable get a stable hash of a JavaScript value.
    value: WebAssembly::Table,
    values: Vec<Value<Engine>>,
    ty: TableType,
}

impl std::fmt::Debug for TableInner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TableInner")
            .field("ty", &self.ty)
            .field("values", &self.value)
            .finish_non_exhaustive()
    }
}

impl TableInner {
    pub fn from_js<T>(store: &mut StoreInner<T>, table: &JsValue) -> Self {
        let table = table.dyn_ref::<WebAssembly::Table>().unwrap();

        let mut ty = table
            .get(0)
            .map(|v| Value::from_js_value(store, &*v).ty())
            .unwrap_or(ValueType::FuncRef);

        let values: Vec<_> = (0..table.length())
            .map(|i| {
                let value = table.get(i).unwrap();

                let v = Value::from_js_value(store, &*value);

                ty = v.ty();

                v
            })
            .collect();

        TableInner {
            value: table.clone(),
            values,
            ty: TableType {
                element: ty,
                min: 0,
                max: None,
            },
        }
    }
}

impl WasmTable<Engine> for Table {
    fn new(
        mut ctx: impl AsContextMut<Engine>,
        ty: TableType,
        init: Value<Engine>,
    ) -> anyhow::Result<Self> {
        let mut ctx: StoreContextMut<_> = ctx.as_context_mut();

        let desc = Object::new();

        Reflect::set(
            &desc,
            &"element".into(),
            &ty.element.to_js_descriptor().into(),
        );
        Reflect::set(&desc, &"initial".into(), &ty.min.into());
        if let Some(max) = ty.max {
            Reflect::set(&desc, &"maximum".into(), &max.into());
        }

        let table = WebAssembly::Table::new(&desc)
            .map_err(JsErrorMsg::from)
            .context("Failed to create table")
            .unwrap();

        let init = init.to_js_value(&ctx);

        for i in 0..ty.min {}

        let table = TableInner {
            value: table,
            ty,
            values,
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
        ctx.as_context().tables[self.id].value.length()
    }

    /// Grows the table by the given amount of elements.
    fn grow(
        &self,
        mut ctx: impl AsContextMut<Engine>,
        delta: u32,
        init: Value<Engine>,
    ) -> anyhow::Result<u32> {
        let ctx: &mut StoreInner<_> = &mut *ctx.as_context_mut();
        let table = &mut ctx.tables[self.id];

        // NOTE: <https://docs.rs/js-sys/0.3.65/js_sys/WebAssembly/struct.Table.html#method.grow> is
        // not correct, as it should accept a second optional argument.
        //
        // We need to get this function manually.
        //
        // Neither can we use <https://docs.rs/js-sys/0.3.65/js_sys/WebAssembly/struct.Table.html#method.set>
        // since it incorrectly uses `Function` values and not the more generic JsValue :|
        let grow_func: Function = Reflect::get(&*table.value, &"grow".into())
            .expect("Table has method `grow`")
            .into();

        let old_size = grow_func
            .call2(&*table.value, &delta.into(), &init.to_js_value(ctx))
            .unwrap()
            .as_f64()
            .expect("grow returns a number") as u32;

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
        let value = &ctx.tables[self.id].value.get(index).ok()?;

        Value::from_js_value(ctx, &*value)
    }
    /// Sets the value of this table at `index`.
    fn set(
        &self,
        mut ctx: impl AsContextMut<Engine>,
        index: u32,
        value: Value<Engine>,
    ) -> anyhow::Result<()> {
        let ctx: &mut StoreInner<_> = &mut *ctx.as_context_mut();
        // RA breaks on this and sees the wrong impl of `value.get`
        //
        // Explicitely telling it that this is a slice of Value<Engine> causes it to see the
        // slice::get method rather than the WasmTable::get function, which shouldn't happen and is
        // a bug since &[]` does not implement `WasmTable`, but alas...
        let values: &mut [Value<Engine>] = &mut ctx.tables[self.id].values;

        *values
            .get_mut(index as usize)
            .ok_or_else(|| anyhow::anyhow!("invalid index"))? = value;

        Ok(())
    }
}
