use super::WasmEngine;

#[derive(Debug, Clone)]
pub struct Engine {}

impl WasmEngine for Engine {
    type ExternRef = ExternRef;

    type Func = Func;

    type Global = Global;

    type Instance = Instance;

    type Memory = Memory;

    type Module = Module;

    type Store<T> = Store<T>;

    type StoreContext<'a, T: 'a> = StoreContext<'a, T>;

    type StoreContextMut<'a, T: 'a> = StoreContextMut<'a, T>;

    type Table = Table;
}

struct ExternRef {}

struct Func {}

struct Global {}

struct Instance {}

struct Memory {}

struct Module {}

struct Store<T> {
    data: T,
}

struct StoreContext<'a, T: 'a> {
    data: &'a T,
}

struct StoreContextMut<'a, T: 'a> {
    data: &'a mut T,
}

struct Table {}
