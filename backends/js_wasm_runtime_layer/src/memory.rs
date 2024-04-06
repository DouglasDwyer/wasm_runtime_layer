use js_sys::{ArrayBuffer, Object, Reflect, Uint8Array, WebAssembly};
use wasm_bindgen::{JsCast as _, JsValue};
use wasm_runtime_layer::{
    backend::{AsContext, AsContextMut, WasmMemory},
    MemoryType,
};

use crate::{conversion::ToStoredJs, Engine, JsErrorMsg, StoreInner};

#[derive(Debug, Clone)]
/// WebAssembly memory
pub struct Memory {
    /// The id of the memory in the store
    pub id: usize,
}

#[derive(Debug)]
/// Holds the inner state of the memory
pub(crate) struct MemoryInner {
    /// The memory value
    pub value: WebAssembly::Memory,
    /// The memory type
    pub ty: MemoryType,
}

impl MemoryInner {
    /// Returns a `Uint8Array` view of the memory
    pub(crate) fn as_uint8array(&self, offset: u32, len: u32) -> Uint8Array {
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

impl Memory {
    /// Construct a memory from an exported memory object
    pub(crate) fn from_exported_memory<T>(
        store: &mut StoreInner<T>,
        value: JsValue,
        ty: MemoryType,
    ) -> Option<Self> {
        let memory: &WebAssembly::Memory = value.dyn_ref()?;

        Some(store.insert_memory(MemoryInner {
            value: memory.clone(),
            ty,
        }))
    }
}

impl WasmMemory<Engine> for Memory {
    fn new(mut ctx: impl AsContextMut<Engine>, ty: MemoryType) -> anyhow::Result<Self> {
        let desc = Object::new();
        Reflect::set(&desc, &"initial".into(), &ty.initial_pages().into()).unwrap();
        if let Some(maximum) = ty.maximum_pages() {
            Reflect::set(&desc, &"maximum".into(), &maximum.into()).unwrap();
        }

        let memory = WebAssembly::Memory::new(&desc).map_err(JsErrorMsg::from)?;

        let ctx: &mut StoreInner<_> = &mut *ctx.as_context_mut();

        Ok(ctx.insert_memory(MemoryInner { value: memory, ty }))
    }

    fn ty(&self, ctx: impl AsContext<Engine>) -> MemoryType {
        ctx.as_context().memories[self.id].ty
    }

    fn grow(&self, mut ctx: impl AsContextMut<Engine>, additional: u32) -> anyhow::Result<u32> {
        let ctx: &mut StoreInner<_> = &mut *ctx.as_context_mut();

        let inner = &mut ctx.memories[self.id];
        Ok(inner.value.grow(additional))
    }

    fn current_pages(&self, _: impl AsContext<Engine>) -> u32 {
        unimplemented!()
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
        let dst = inner.as_uint8array(offset as _, buffer.len() as _);

        #[cfg(feature = "tracing")]
        tracing::debug!("writing {buffer:?} into guest");
        dst.copy_from(buffer);

        Ok(())
    }
}
