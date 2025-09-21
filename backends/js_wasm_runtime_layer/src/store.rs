use alloc::{boxed::Box, vec::Vec};
use core::{
    fmt, mem,
    ops::{Deref, DerefMut},
};

use slab::Slab;
use wasm_runtime_layer::backend::{
    AsContext, AsContextMut, WasmEngine, WasmStore, WasmStoreContext, WasmStoreContextMut,
};

use crate::{
    func::FuncInner, instance::InstanceInner, memory::MemoryInner, table::TableInner, DropResource,
    Engine, Func, Global, GlobalInner, Instance, Memory, Table,
};

/// Owns all the data for the wasm module
///
/// Can be cheaply cloned
///
/// The data is retained through the lifetime of the store, and no GC will collect data from
/// no-longer used modules. It is as such recommended to have the stores lifetime correspond to its
/// modules, and not repeatedly create and drop modules within an existing store, but rather create
/// a new store for it, to avoid unbounded memory use.
pub struct Store<T: 'static> {
    /// The internal store is kept behind a pointer.
    ///
    /// This is to allow referencing and reconstructing a calling context in exported functions,
    /// where it is not possible to prove the correct lifetime and borrowing rules statically nor
    /// dynamically using RefCells. This is because functions can be re-entrant with exclusive but
    /// stacked calling contexts. [`std::cell::RefCell`] and [`std::cell::RefMut`] do not allow
    /// for recursive usage by design (and it would be nigh impossible and quite expensive to enforce at runtime).
    ///
    /// The store is stored through a raw pointer, as using a `Pin<Box<T>>` would not be possible,
    /// despite the memory location of the Box contents technically being pinned in memory. This is
    /// because of the stacked borrows model.
    ///
    /// When the outer box is moved, it invalidates all tags in its borrow stack, even
    /// though the memory location remains. This invalidates all references and raw pointers to `T`
    /// created from the Box.
    ///
    /// See: <https://blog.nilstrieb.dev/posts/box-is-a-unique-type/> for more details.
    ///
    /// By using a box here, we would leave invalid pointers with revoked access permissions to the
    /// memory location of `T`.
    ///
    /// This creates undefined behavior as the Rust compiler will incorrectly optimize register
    /// accesses and memory loading and incorrect no-alias attributes.
    ///
    /// To circumvent this we can use a raw pointer obtained from unwrapping a Box.
    ///
    /// # Playground
    ///
    /// - `Pin<Box<T>>` solution (UB): <https://play.rust-lang.org/?version=stable&mode=debug&edition=2021&gist=685c984584bc0ca1faa780ca292f406c>
    /// - raw pointer solution (sound): <https://play.rust-lang.org/?version=stable&mode=release&edition=2021&gist=257841cb1675106d55c756ad59fde2fb>
    ///
    /// You can use `Tools > Miri` to test the validity
    inner: *mut StoreInner<T>,
}

impl<T: 'static> Store<T> {
    /// Creates a new store from the inner box
    fn from_inner(inner: Box<StoreInner<T>>) -> Self {
        Self {
            inner: Box::into_raw(inner),
        }
    }

    /// Returns a borrow of the store
    pub(crate) fn get(&self) -> StoreContext<'_, T> {
        // Safety:
        //
        // A shared reference to the store signifies a non-mutable ownership, and is thus safe.
        let inner = unsafe { &*self.inner };
        StoreContext::from_ref(inner)
    }

    /// Returns a mutable borrow of the store
    pub(crate) fn get_mut(&mut self) -> StoreContextMut<'_, T> {
        // Safety:
        //
        // &mut self
        let inner = unsafe { &mut *self.inner };
        StoreContextMut::from_ref(inner)
    }
}

impl<T: 'static> Drop for Store<T> {
    fn drop(&mut self) {
        unsafe { drop(Box::from_raw(self.inner)) }
    }
}

impl<T: 'static> WasmStore<T, Engine> for Store<T> {
    fn new(engine: &Engine, data: T) -> Self {
        #[cfg(feature = "tracing")]
        let _span = tracing::debug_span!("Store::new").entered();
        Self::from_inner(Box::new(StoreInner {
            engine: engine.clone(),
            instances: Slab::new(),
            funcs: Slab::new(),
            globals: Slab::new(),
            tables: Slab::new(),
            memories: Slab::new(),
            drop_resources: Vec::new(),
            data,
        }))
    }

    fn engine(&self) -> &Engine {
        &self.get().store.engine
    }

    fn data(&self) -> &T {
        &self.get().store.data
    }

    fn data_mut(&mut self) -> &mut T {
        &mut self.get_mut().store.data
    }

    fn into_data(self) -> T {
        // Safety:
        //
        // Ownership of `self` signifies that no guest stack is currently active
        let ptr = unsafe { Box::from_raw(self.inner) };

        // Don't execute drop for `Store`. This impl deallocates the whole box, which we don't
        // want.
        //
        // The box will be deallocated at the end of this scope
        mem::forget(self);

        ptr.data
    }
}

impl<T: 'static> AsContext<Engine> for Store<T> {
    type UserState = T;

    fn as_context(&self) -> <Engine as WasmEngine>::StoreContext<'_, Self::UserState> {
        self.get()
    }
}

impl<T: 'static> AsContextMut<Engine> for Store<T> {
    fn as_context_mut(&mut self) -> StoreContextMut<'_, T> {
        self.get_mut()
    }
}

impl<T: 'static + fmt::Debug> fmt::Debug for Store<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.inner.fmt(f)
    }
}

#[derive(Debug)]
/// Holds the inner state of the store
pub struct StoreInner<T: 'static> {
    /// The engine used
    pub(crate) engine: Engine,
    /// Instances are not Send + Sync
    pub(crate) instances: Slab<InstanceInner>,
    /// Modules are not Send + Sync
    pub(crate) funcs: Slab<FuncInner>,
    /// Globals
    pub(crate) globals: Slab<GlobalInner>,
    /// Tables
    pub(crate) tables: Slab<TableInner>,
    /// Guest memories
    pub(crate) memories: Slab<MemoryInner>,
    /// The user data
    pub(crate) data: T,

    /// **Note**: append ONLY. No resource must be dropped or removed from this vector as long as
    /// the store is still alive.
    ///
    /// Dropping a resource too early is safe, but the resulting behavior is not specifed and may
    /// include incorrect results, memory leaks or panics, etc.
    drop_resources: Vec<DropResource>,
}

impl<T: 'static> StoreInner<T> {
    /// Inserts a new function and returns its id
    pub(crate) fn insert_func(&mut self, func: FuncInner) -> Func {
        Func {
            id: self.funcs.insert(func),
        }
    }

    /// Inserts a new global and returns its id
    pub(crate) fn insert_global(&mut self, global: GlobalInner) -> Global {
        Global {
            id: self.globals.insert(global),
        }
    }

    /// Inserts a new table and returns its id
    pub(crate) fn insert_table(&mut self, table: TableInner) -> Table {
        Table {
            id: self.tables.insert(table),
        }
    }

    /// Inserts a new instance and returns its id
    pub(crate) fn insert_instance(&mut self, instance: InstanceInner) -> Instance {
        Instance {
            id: self.instances.insert(instance),
        }
    }

    /// Inserts a new guest memory and returns its id
    pub(crate) fn insert_memory(&mut self, memory: MemoryInner) -> Memory {
        Memory {
            id: self.memories.insert(memory),
        }
    }

    /// Tie the lifetime of a reference or other value to the lifetime of the store using
    /// [`DropResource`].
    pub(crate) fn insert_drop_resource(&mut self, value: DropResource) {
        self.drop_resources.push(value)
    }
}

/// Immutable context to the store
pub struct StoreContext<'a, T: 'static> {
    /// The store
    store: &'a StoreInner<T>,
}

impl<'a, T: 'static> StoreContext<'a, T> {
    /// Provides a store context from a reference
    pub fn from_ref(store: &'a StoreInner<T>) -> Self {
        Self { store }
    }
}

impl<T: 'static> Deref for StoreContext<'_, T> {
    type Target = StoreInner<T>;

    fn deref(&self) -> &Self::Target {
        self.store
    }
}

/// Mutable context to the store
pub struct StoreContextMut<'a, T: 'static> {
    /// The store
    store: &'a mut StoreInner<T>,
}

impl<'a, T: 'static> StoreContextMut<'a, T> {
    /// Returns a pointer to the inner store
    pub(crate) fn as_ptr(&mut self) -> *mut StoreInner<T> {
        self.store as *mut _
    }

    /// Provides a mutable store context from a reference
    pub(crate) fn from_ref(store: &'a mut StoreInner<T>) -> Self {
        Self { store }
    }
}

impl<T: 'static> Deref for StoreContextMut<'_, T> {
    type Target = StoreInner<T>;

    fn deref(&self) -> &Self::Target {
        &*self.store
    }
}

impl<T: 'static> DerefMut for StoreContextMut<'_, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut *self.store
    }
}

impl<'a, T: 'static> WasmStoreContext<'a, T, Engine> for StoreContext<'a, T> {
    fn engine(&self) -> &Engine {
        &self.engine
    }

    fn data(&self) -> &T {
        &self.data
    }
}

impl<'a, T: 'static> AsContext<Engine> for StoreContext<'a, T> {
    type UserState = T;

    fn as_context(&self) -> StoreContext<'_, T> {
        StoreContext { store: self.store }
    }
}

impl<'a, T: 'static> WasmStoreContext<'a, T, Engine> for StoreContextMut<'a, T> {
    fn engine(&self) -> &Engine {
        &self.engine
    }

    fn data(&self) -> &T {
        &self.data
    }
}

impl<'a, T: 'static> WasmStoreContextMut<'a, T, Engine> for StoreContextMut<'a, T> {
    fn data_mut(&mut self) -> &mut T {
        &mut self.data
    }
}

impl<'a, T: 'static> AsContext<Engine> for StoreContextMut<'a, T> {
    type UserState = T;

    fn as_context(&self) -> <Engine as WasmEngine>::StoreContext<'_, T> {
        StoreContext { store: self.store }
    }
}

impl<'a, T: 'static> AsContextMut<Engine> for StoreContextMut<'a, T> {
    fn as_context_mut(&mut self) -> StoreContextMut<'_, T> {
        StoreContextMut { store: self.store }
    }
}
