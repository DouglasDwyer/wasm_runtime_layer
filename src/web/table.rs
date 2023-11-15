use js_sys::WebAssembly;
use wasm_bindgen::{JsCast, JsValue};

use crate::{
    backend::{AsContext, AsContextMut, Value, WasmTable},
    TableType,
};

use super::{Engine, StoreContextMut, StoreInner};

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
            .field("values", &self.values.len())
            .finish_non_exhaustive()
    }
}

impl TableInner {
    pub fn from_js<T>(store: &mut StoreInner<T>, table: &JsValue) -> Self {
        let table = table.dyn_ref::<WebAssembly::Table>().unwrap();

        let mut ty = crate::ValueType::FuncRef;

        let values: Vec<_> = (0..table.length())
            .map(|i| {
                let value = table.get(i).unwrap();

                let v = Value::from_js_value(store, &*value);

                ty = v.ty();

                v
            })
            .collect();

        TableInner {
            ty: TableType {
                element: ty,
                min: values.len() as _,
                max: None,
            },
            values,
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
    ) -> anyhow::Result<u32> {
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
