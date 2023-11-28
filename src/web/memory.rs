use js_sys::{ArrayBuffer, Object, Reflect, Uint8Array, WebAssembly};
use wasm_bindgen::{JsCast as _, JsValue};

use crate::backend::{AsContext, AsContextMut, WasmMemory};

use super::{
    conversion::{FromStoredJs, ToStoredJs},
    Engine, JsErrorMsg, StoreInner,
};

#[derive(Debug, Clone)]
pub struct Memory {
    pub id: usize,
}

#[derive(Debug)]
pub struct MemoryInner {
    pub value: WebAssembly::Memory,
}

impl MemoryInner {
    pub fn as_uint8array(&self, offset: u32, len: u32) -> Uint8Array {
        let buffer = self.value.buffer();
        let buffer = buffer.dyn_ref::<ArrayBuffer>().unwrap();

        Uint8Array::new_with_byte_offset_and_length(buffer, offset, len)
    }
}

impl ToStoredJs for Memory {
    type Repr = WebAssembly::Memory;
    fn to_stored_js<T>(&self, store: &StoreInner<T>) -> WebAssembly::Memory {
        let memory = &store.memories[self.id];

        memory.value.clone()
    }
}

impl FromStoredJs for Memory {
    fn from_stored_js<T>(store: &mut StoreInner<T>, value: JsValue) -> Option<Self> {
        let memory: &WebAssembly::Memory = value.dyn_ref()?;

        Some(store.insert_memory(MemoryInner {
            value: memory.clone(),
        }))
    }
}

impl WasmMemory<Engine> for Memory {
    fn new(mut ctx: impl AsContextMut<Engine>, ty: crate::MemoryType) -> anyhow::Result<Self> {
        let desc = Object::new();
        Reflect::set(&desc, &"intial".into(), &ty.initial.into()).unwrap();
        if let Some(maximum) = ty.maximum {
            Reflect::set(&desc, &"maximum".into(), &maximum.into()).unwrap();
        }

        let memory = WebAssembly::Memory::new(&desc).map_err(JsErrorMsg::from)?;

        let ctx: &mut StoreInner<_> = &mut *ctx.as_context_mut();

        Ok(ctx.insert_memory(MemoryInner { value: memory }))
    }

    fn ty(&self, ctx: impl AsContext<Engine>) -> crate::MemoryType {
        todo!()
    }

    fn grow(&self, mut ctx: impl AsContextMut<Engine>, additional: u32) -> anyhow::Result<u32> {
        let ctx: &mut StoreInner<_> = &mut *ctx.as_context_mut();

        let inner = &mut ctx.memories[self.id];
        Ok(inner.value.grow(additional))
    }

    fn current_pages(&self, ctx: impl AsContext<Engine>) -> u32 {
        todo!()
    }

    fn read(
        &self,
        ctx: impl AsContext<Engine>,
        offset: usize,
        buffer: &mut [u8],
    ) -> anyhow::Result<()> {
        let ctx: &StoreInner<_> = &*ctx.as_context();
        let memory = &ctx.memories[self.id];

        memory
            .as_uint8array(offset as _, buffer.len() as _)
            .copy_to(buffer);

        Ok(())
    }

    fn write(
        &self,
        mut ctx: impl AsContextMut<Engine>,
        offset: usize,
        buffer: &[u8],
    ) -> anyhow::Result<()> {
        let ctx: &mut StoreInner<_> = &mut *ctx.as_context_mut();

        let inner = &mut ctx.memories[self.id];
        let host_mem = wasm_bindgen::memory()
            .dyn_into::<WebAssembly::Memory>()
            .unwrap()
            .buffer()
            .dyn_into::<ArrayBuffer>()
            .unwrap();

        let dst = inner.as_uint8array(offset as _, buffer.len() as _);

        tracing::debug!("writing {buffer:?} into guest");
        dst.copy_from(buffer);

        Ok(())
    }
}
