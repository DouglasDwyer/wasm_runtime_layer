use std::{collections::BTreeMap, io::Cursor};

use wasm_runtime_layer::{
    backend::WasmEngine, Engine, Extern, Func, FuncType, Imports, Instance, Module, Store, Value,
    ValueType,
};

#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_wasmtime() {
    // 1. Instantiate a runtime
    let engine = Engine::new(wasmtime_runtime_layer::Engine::default());
    multi_value(&engine)
}

#[test]
#[wasm_bindgen_test::wasm_bindgen_test]
fn test_wasmi() {
    // 1. Instantiate a runtime
    let engine = Engine::new(wasmi_runtime_layer::Engine::default());
    multi_value(&engine)
}

#[wasm_bindgen_test::wasm_bindgen_test]
#[cfg(target_arch = "wasm32")]
fn test_js_wasm() {
    // 1. Instantiate a runtime

    let engine = Engine::new(js_wasm_runtime_layer::Engine::default());
    multi_value(&engine)
}

#[allow(unused)]
fn multi_value(engine: &Engine<impl WasmEngine>) {
    let mut store = Store::new(engine, ());

    // 2. Create modules and instances, similar to other runtimes
    let module_bin = wat::parse_str(
        r#"
            (module
              (import "host" "get-values" (func $get-values (result i32) (result i32)))

              (func $add-sub (param $a i32) (param $b i32) (result i32) (result i32)
                  (local $c i32)
                  call $get-values
                  i32.sub

                  local.set $c
                    
                  local.get $a
                  local.get $b
                  i32.add
                  local.get $c)

              (export "add-sub" (func $add-sub)))
            "#,
    )
    .unwrap();

    let func = Func::new(
        &mut store,
        FuncType::new([], [ValueType::I32, ValueType::I32]),
        |_, _, res| {
            res[0] = crate::Value::I32(5);
            res[1] = crate::Value::I32(4);
            Ok(())
        },
    );

    let mut imports = Imports::new();
    imports.define("host", "get-values", crate::Extern::Func(func));

    // Parse the component bytes and load its imports and exports.
    let module = Module::new(engine, Cursor::new(&module_bin)).unwrap();
    let instance = Instance::new(&mut store, &module, &imports).unwrap();

    let exports = instance
        .exports(&store)
        .map(|v| (v.name, v.value))
        .collect::<BTreeMap<_, _>>();

    // Get the function for selecting a list element.
    let Extern::Func(func) = exports.get("add-sub").unwrap() else {
        unreachable!()
    };

    let mut result = [Value::I32(0), Value::I32(0)];
    func.call(&mut store, &[Value::I32(5), Value::I32(7)], &mut result)
        .unwrap();

    assert_eq!(result, [Value::I32(12), Value::I32(1)],);
}
