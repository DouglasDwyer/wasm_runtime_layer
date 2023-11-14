use std::{
    cell::{Ref, RefCell, RefMut},
    ops::{Deref, DerefMut},
    rc::Rc,
};

use slab::Slab;

use crate::backend::{
    AsContext, AsContextMut, WasmEngine, WasmStore, WasmStoreContext, WasmStoreContextMut,
};

use super::{DropResource, Engine, Func, FuncInner, InstanceInner, ModuleInner};

/// Owns all the data for the wasm module
///
/// Can be cheaply cloned
pub struct Store<T> {
    inner: Rc<RefCell<StoreInner<T>>>,
}

impl<T> Store<T> {
    pub(crate) fn from_inner(inner: Rc<RefCell<StoreInner<T>>>) -> Self {
        Self { inner }
    }

    pub(crate) fn borrow(&mut self) -> Ref<'_, StoreInner<T>> {
        self.inner.borrow()
    }

    pub(crate) fn borrow_mut(&mut self) -> RefMut<'_, StoreInner<T>> {
        self.inner.borrow_mut()
    }

    pub(crate) fn get_mut(&mut self) -> StoreContextMut<T> {
        StoreContextMut::from_store(self)
    }

    pub(crate) fn get(&self) -> StoreContext<T> {
        StoreContext::from_store(self)
    }
}

impl<T: std::fmt::Debug> std::fmt::Debug for Store<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.inner.fmt(f)
    }
}

impl<T> Clone for Store<T> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

/// TODO: consider moving this to compile time by modifying the AsContext(Mut) traits to return an
/// associated type instead of a singular StoreContext
///
/// Alternatively, this could be solved using unsafe by storing a pointer to the data along
/// with an Option<Ref> for keeping the guard alive.
enum RcOrRef<'a, T> {
    Rc(Ref<'a, T>),
    Ref(&'a T),
}

impl<'a, T> Deref for RcOrRef<'a, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        match self {
            Self::Rc(v) => &*v,
            Self::Ref(v) => v,
        }
    }
}

enum RcOrRefMut<'a, T> {
    Rc(RefMut<'a, T>),
    Ref(&'a mut T),
}

impl<'a, T> Deref for RcOrRefMut<'a, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        match self {
            Self::Rc(v) => &*v,
            Self::Ref(v) => v,
        }
    }
}

impl<'a, T> DerefMut for RcOrRefMut<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        match self {
            Self::Rc(v) => &mut *v,
            Self::Ref(v) => v,
        }
    }
}

#[derive(Debug)]
pub struct StoreInner<T> {
    pub(crate) engine: Engine,
    // Instances are not Send + Sync
    pub(crate) instances: Slab<InstanceInner>,
    // Modules are not Send + Sync
    pub(crate) modules: Slab<ModuleInner>,
    pub(crate) funcs: Slab<FuncInner>,
    pub(crate) data: T,

    /// **Note**: append ONLY. No resource must be dropped or removed from this vector as long as
    /// the store is still alive.
    ///
    /// Dropping a resource too early is safe, but the resulting behavior is not specifed and may
    /// include incorrect results, memory leaks or panics, etc.
    drop_resources: Vec<DropResource>,
}

impl<T> StoreInner<T> {
    pub(crate) fn create_func(&mut self, func: FuncInner) -> Func {
        Func {
            id: self.funcs.insert(func),
        }
    }

    /// Tie the lifetime of a reference or other value to the lifetime of the store using
    /// [`DropResource`].
    pub(crate) fn push_drop_resource(&mut self, value: DropResource) {
        self.drop_resources.push(value)
    }
}

/// Immutable context to the store
pub struct StoreContext<'a, T: 'a> {
    /// The store
    store: RcOrRef<'a, StoreInner<T>>,
    orig: &'a Rc<RefCell<StoreInner<T>>>,
}

impl<'a, T: 'a> StoreContext<'a, T> {
    pub fn from_store(store: &'a Store<T>) -> Self {
        Self {
            store: RcOrRef::Rc(store.inner.borrow()),
            orig: &store.inner,
        }
    }
}

impl<'a, T> Deref for StoreContext<'a, T> {
    type Target = StoreInner<T>;

    fn deref(&self) -> &Self::Target {
        &*self.store
    }
}

/// Mutable context to the store
pub struct StoreContextMut<'a, T: 'a> {
    /// The store
    store: RcOrRefMut<'a, StoreInner<T>>,
    orig: &'a Rc<RefCell<StoreInner<T>>>,
}

impl<'a, T: 'a> StoreContextMut<'a, T> {
    pub fn from_store(store: &'a Store<T>) -> Self {
        Self {
            store: RcOrRefMut::Rc(store.inner.borrow_mut()),
            orig: &store.inner,
        }
    }
}

impl<'a, T> Deref for StoreContextMut<'a, T> {
    type Target = StoreInner<T>;

    fn deref(&self) -> &Self::Target {
        &*self.store
    }
}

impl<'a, T> DerefMut for StoreContextMut<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut *self.store
    }
}

impl<T> WasmStore<T, Engine> for Store<T> {
    fn new(engine: &Engine, data: T) -> Self {
        let _span = tracing::info_span!("Store::new").entered();
        Self::from_inner(Rc::new(RefCell::new(StoreInner {
            engine: engine.clone(),
            instances: Slab::new(),
            modules: Slab::new(),
            funcs: Slab::new(),
            drop_resources: Vec::new(),
            data,
        })))
    }

    fn engine(&self) -> &Engine {
        unimplemented!()
    }

    // fn data(&self) -> &T {
    //     unimplemented!()
    // }

    // fn data_mut(&mut self) -> &mut T {
    //     unimplemented!()
    // }

    // fn into_data(self) -> T {
    //     todo!()
    // }
}

impl<T> AsContext<Engine> for Store<T> {
    type UserState = T;

    fn as_context(&self) -> <Engine as WasmEngine>::StoreContext<'_, Self::UserState> {
        self.get()
    }
}

impl<T> AsContextMut<Engine> for Store<T> {
    fn as_context_mut(&mut self) -> StoreContextMut<T> {
        StoreContextMut::from_store(&mut *self)
    }
}

impl<'a, T: 'a> WasmStoreContext<'a, T, Engine> for StoreContext<'a, T> {
    fn engine(&self) -> &Engine {
        &self.engine
    }

    fn data(&self) -> &T {
        &self.data
    }
}

impl<'a, T: 'a> AsContext<Engine> for StoreContext<'a, T> {
    type UserState = T;

    fn as_context(&self) -> StoreContext<'_, T> {
        StoreContext {
            store: RcOrRef::Ref(&*self),
            orig: self.orig,
        }
    }
}

impl<'a, T: 'a> WasmStoreContext<'a, T, Engine> for StoreContextMut<'a, T> {
    fn engine(&self) -> &Engine {
        &self.engine
    }

    fn data(&self) -> &T {
        &self.data
    }
}

impl<'a, T: 'a> WasmStoreContextMut<'a, T, Engine> for StoreContextMut<'a, T> {
    fn data_mut(&mut self) -> &mut T {
        &mut self.data
    }
}

impl<'a, T: 'a> AsContext<Engine> for StoreContextMut<'a, T> {
    type UserState = T;

    fn as_context(&self) -> <Engine as WasmEngine>::StoreContext<'_, T> {
        match &self.store {
            RcOrRefMut::Rc(v) => StoreContext {
                store: RcOrRef::Ref(&*v),
                orig: self.orig,
            },
            RcOrRefMut::Ref(v) => StoreContext {
                store: RcOrRef::Ref(&*v),
                orig: self.orig,
            },
        }
    }
}

impl<'a, T: 'a> AsContextMut<Engine> for StoreContextMut<'a, T> {
    fn as_context_mut(&mut self) -> StoreContextMut<'_, T> {
        StoreContextMut {
            store: RcOrRefMut::Ref(&mut *self.store),
            orig: self.orig,
        }
    }
}

impl<'a, T> StoreContext<'a, T> {
    pub fn store(&self) -> Store<T> {
        Store {
            inner: self.orig.clone(),
        }
    }
}

impl<'a, T> StoreContextMut<'a, T> {
    pub fn store(&self) -> Store<T> {
        Store {
            inner: self.orig.clone(),
        }
    }
}
