use wasm_bindgen::JsValue;

use super::StoreInner;

/// Converts a Rust type to and from JavaScript
pub trait JsConvert {
    /// Convert this value to JavaScript
    fn to_js<T>(&self, store: &StoreInner<T>) -> JsValue;
    /// Convert a JavaScript value to this type
    ///
    /// Returns None if the type does not match
    fn from_js<T>(store: &mut StoreInner<T>, value: JsValue) -> Option<Self>
    where
        Self: Sized;
}

macro_rules! conv_number {
    ($ty: ty) => {
        impl JsConvert for $ty {
            fn to_js<T>(&self, _: &StoreInner<T>) -> JsValue {
                JsValue::from(*self)
            }

            fn from_js<T>(store: &mut StoreInner<T>, value: JsValue) -> Option<Self> {
                Some(value.as_f64()? as $ty)
            }
        }
    };
    ($($ty: ty),*) => {
        $(conv_number!{ $ty } )*
    }
}

conv_number!(i32, i64, f32, f64);
