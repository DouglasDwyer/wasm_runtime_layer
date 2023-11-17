use js_sys::Number;
use wasm_bindgen::{convert::js_value_vector_from_abi, JsValue};

use super::StoreInner;

/// Converts a Rust type from JavaScript
pub trait FromJs {
    /// Convert a JavaScript value to this type
    ///
    /// Returns None if the type does not match
    fn from_js<T>(store: &mut StoreInner<T>, value: JsValue) -> Option<Self>
    where
        Self: Sized;
}

/// Converts a Rust type to JavaScript
pub trait ToJs {
    /// The type used to represent the resulting type for JavaScript.
    ///
    /// Sometimes, the exact underlying type is desired in order to be manipulated
    type Repr: Into<JsValue>;

    /// Convert this value to JavaScript
    fn to_js<T>(&self, store: &StoreInner<T>) -> Self::Repr;
}

macro_rules! conv_number {
    ($ty: ty) => {
        impl ToJs for $ty {
            type Repr = Number;

            fn to_js<T>(&self, _: &StoreInner<T>) -> Number {
                From::from(*self)
            }
        }

        impl FromJs for $ty {
            fn from_js<T>(store: &mut StoreInner<T>, value: JsValue) -> Option<Self> {
                Some(value.as_f64()? as $ty)
            }
        }
    };

    ($($ty: ty),*) => {
        $(conv_number!{ $ty } )*
    }
}

conv_number!(i32, f32, f64);

impl ToJs for i64 {
    type Repr = JsValue;

    fn to_js<T>(&self, _: &StoreInner<T>) -> Self::Repr {
        // NOTE: this is the closest representation of a 64 bit integer in javascript without
        // using BigInt.
        //
        // This is a lossy conversion and can only accurately represent 2^53 without significand
        // loss
        JsValue::from(*self as f64)
    }
}

impl FromJs for i64 {
    fn from_js<T>(store: &mut StoreInner<T>, value: JsValue) -> Option<Self>
    where
        Self: Sized,
    {
        value.try_into().ok()
    }
}
