use anyhow::Context;
use js_sys::{Array, Function};
use wasm_bindgen::{closure::Closure, JsCast, JsValue};
use web_sys::console;

use crate::{
    backend::{AsContext, AsContextMut, Value, WasmFunc},
    web::{DropResource, JsErrorMsg},
};

use super::{
    conversion::{FromStoredJs, ToStoredJs},
    Engine, StoreContextMut, StoreInner,
};

/// A bound function
#[derive(Debug, Clone)]
pub struct Func {
    pub(crate) id: usize,
}

/// Internal represenation of [`Func`]
#[derive(Debug)]
pub(crate) struct FuncInner {
    pub(crate) func: Function,
}

impl ToStoredJs for Func {
    type Repr = Function;
    fn to_stored_js<T>(&self, store: &StoreInner<T>) -> Function {
        let func = &store.funcs[self.id];
        func.func.clone()
    }
}

impl FromStoredJs for Func {
    fn from_stored_js<T>(store: &mut StoreInner<T>, value: JsValue) -> Option<Self> {
        let func: Function = value.dyn_into().ok()?;

        Some(store.insert_func(FuncInner { func }))
    }
}

impl WasmFunc<Engine> for Func {
    fn new<T>(
        mut ctx: impl AsContextMut<Engine, UserState = T>,
        ty: crate::FuncType,
        func: impl 'static
            + Send
            + Sync
            + Fn(StoreContextMut<T>, &[Value<Engine>], &mut [Value<Engine>]) -> anyhow::Result<()>,
    ) -> Self {
        let _span = tracing::info_span!("Func::new").entered();

        let mut ctx: StoreContextMut<_> = ctx.as_context_mut();

        let mut store = ctx.store();
        // let mut host_args = Vec::new();

        let closure: Closure<dyn FnMut(Array) -> JsValue> = Closure::new(move |args: Array| {
            // let store: &mut StoreInner<T> = &mut *store.as_context_mut();

            let (arg_types, ret_types) = ty.params_results.split_at(ty.len_params);

            tracing::info!(?args, "called import");

            // host_args.clear();

            // host_args.extend(
            //     args.iter()
            //         .zip(arg_types)
            //         .map(|(value, ty)| Value::from_js_typed(store, ty, value)),
            // );

            // assert_eq!(host_args.len(), ty.len_params);

            JsValue::UNDEFINED
        });

        let func = ctx.insert_func(FuncInner {
            func: closure.as_ref().unchecked_ref::<Function>().clone(),
        });

        tracing::debug!(id = func.id, "func");
        ctx.insert_drop_resource(DropResource::new(closure));

        func
    }

    fn ty(&self, ctx: impl AsContext<Engine>) -> crate::FuncType {
        todo!()
    }

    fn call<T>(
        &self,
        mut ctx: impl AsContextMut<Engine>,
        args: &[Value<Engine>],
        results: &mut [Value<Engine>],
    ) -> anyhow::Result<()> {
        tracing::info!(id = self.id, ?args, ?results, "call");

        let ctx: &mut StoreInner<_> = &mut *ctx.as_context_mut();
        let inner = &mut ctx.funcs[self.id];

        tracing::info!(?inner, "function");

        console::log_1(&inner.func);

        inner
            .func
            .apply(&JsValue::UNDEFINED, &Array::new())
            .map_err(JsErrorMsg::from)
            .context("Guest function threw an error")?;

        Ok(())
    }
}
