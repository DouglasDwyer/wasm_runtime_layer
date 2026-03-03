use core::fmt;

use anyhow::Context;
use js_sys::{Object, Reflect, WebAssembly};
use wasm_bindgen::{JsCast, JsValue};
use wasm_runtime_layer::{
    backend::{AsContext, AsContextMut, Ref, WasmTable},
    RefType, TableType,
};

use crate::{
    conversion::{ToJs, ToStoredJs},
    Engine, JsErrorMsg, StoreContextMut, StoreInner,
};

#[derive(Debug, Clone)]
/// WebAssembly table
pub struct Table {
    /// The id of the table in the store
    pub(crate) id: usize,
}

/// Holds the inner WebAssembly table
pub(crate) struct TableInner {
    /// Table reference
    table: WebAssembly::Table,
    /// The table signature
    ty: TableType,
}

impl fmt::Debug for TableInner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TableInner")
            .field("ty", &self.ty)
            .field("inner", &self.table)
            .finish_non_exhaustive()
    }
}

impl Table {
    /// Creates a new table from a JS value
    pub(crate) fn from_stored_js<T>(
        store: &mut StoreInner<T>,
        value: JsValue,
        ty: TableType,
    ) -> Option<Self> {
        #[cfg(feature = "tracing")]
        let _span = tracing::trace_span!("Table::from_js", ?value).entered();
        let table = value.dyn_into::<WebAssembly::Table>().ok()?;

        assert!(table.length() >= ty.minimum());
        assert_eq!(ty.element(), RefType::FuncRef);

        let inner = TableInner { ty, table };

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
        init: Ref<Engine>,
    ) -> anyhow::Result<Self> {
        #[cfg(feature = "tracing")]
        let _span = tracing::debug_span!("Table::new", ?ty, ?init).entered();
        let mut ctx: StoreContextMut<_> = ctx.as_context_mut();

        let desc = Object::new();
        Reflect::set(&desc, &"element".into(), &ty.element().to_js()).unwrap();

        Reflect::set(&desc, &"initial".into(), &ty.minimum().into()).unwrap();
        if let Some(max) = ty.maximum() {
            Reflect::set(&desc, &"initial".into(), &max.into()).unwrap();
        }

        let table = WebAssembly::Table::new(&desc).map_err(JsErrorMsg::from)?;

        for i in 0..ty.minimum() {
            table
                .set(i, init.to_stored_js(&ctx).unchecked_ref())
                .unwrap();
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
        init: Ref<Engine>,
    ) -> anyhow::Result<u32> {
        let ctx: &mut StoreInner<_> = &mut *ctx.as_context_mut();
        let init = init.to_stored_js(ctx);
        let init = init.unchecked_ref();

        let inner = &mut ctx.tables[self.id];
        let old_len = inner.table.grow(delta).map_err(JsErrorMsg::from)?;

        for i in old_len..(old_len + delta) {
            inner.table.set(i, init).unwrap();
        }

        Ok(old_len)
    }

    /// Returns the table element at `index`.
    fn get(&self, _: impl AsContextMut<Engine>, _: u32) -> Option<Ref<Engine>> {
        // It is not possible to determine the type or signature of the element.
        //
        // To enable this we would need to cache and intern a unique id for each element to be able
        // to reconcile the signature and existing Store element. This would also avoid duplicating
        // elements
        #[cfg(feature = "tracing")]
        tracing::error!("get is not implemented");
        None
    }

    /// Sets the value of this table at `index`.
    fn set(
        &self,
        mut ctx: impl AsContextMut<Engine>,
        index: u32,
        elem: Ref<Engine>,
    ) -> anyhow::Result<()> {
        let ctx: &mut StoreInner<_> = &mut *ctx.as_context_mut();
        // RA breaks on this and sees the wrong impl of `elem.get`
        //
        // Explicitely telling it that this is a slice of Ref<Engine> causes it to see the
        // slice::get method rather than the WasmTable::get function, which shouldn't happen and is
        // a bug since &[]` does not implement `WasmTable`, but alas...
        let elem = elem.to_stored_js(ctx);

        let inner: &mut TableInner = &mut ctx.tables[self.id];

        inner
            .table
            .set(index, elem.unchecked_ref())
            .map_err(JsErrorMsg::from)
            .context("Invalid index")?;

        Ok(())
    }
}
