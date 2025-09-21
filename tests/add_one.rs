use wasm_runtime_layer::{backend::WasmEngine, Engine, Imports, Instance, Module, Store, Value};

#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_wasmtime() {
    // 1. Instantiate a runtime
    let engine = Engine::new(wasmtime_runtime_layer::Engine::default());
    add_one(&engine)
}

#[test]
#[wasm_bindgen_test::wasm_bindgen_test]
fn test_wasmi() {
    // 1. Instantiate a runtime
    let engine = Engine::new(wasmi_runtime_layer::Engine::default());
    add_one(&engine)
}

#[wasm_bindgen_test::wasm_bindgen_test]
#[cfg(target_arch = "wasm32")]
fn test_js_wasm() {
    // 1. Instantiate a runtime
    let engine = Engine::new(js_wasm_runtime_layer::Engine::default());
    add_one(&engine)
}

// #[test]
// #[cfg(not(target_arch = "wasm32"))]
// fn test_wasmer() {
//     // 1. Instantiate a runtime
//     let engine = Engine::new(wasmer_runtime_layer::Engine::default());
//     add_one(&engine)
// }

#[allow(unused)]
fn add_one(engine: &Engine<impl WasmEngine>) {
    let mut store = Store::new(engine, ());

    // 2. Create modules and instances, similar to other runtimes
    let module_bin = wat::parse_str(
        r#"
        (module
        (type $t0 (func (param i32) (result i32)))
        (func $add_one (export "add_one") (type $t0) (param $p0 i32) (result i32)
            local.get $p0
            i32.const 1
            i32.add))
        "#,
    )
    .unwrap();

    let module = Module::new(engine, &module_bin).unwrap();
    let instance = Instance::new(&mut store, &module, &Imports::default()).unwrap();

    let add_one = instance
        .get_export(&store, "add_one")
        .unwrap()
        .into_func()
        .unwrap();

    let mut result = [crate::Value::I32(0)];
    add_one
        .call(&mut store, &[crate::Value::I32(42)], &mut result)
        .unwrap();

    assert_eq!(result[0], crate::Value::I32(43));
}
