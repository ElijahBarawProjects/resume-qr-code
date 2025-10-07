use std::collections::binary_heap::Iter;

pub trait OneZero {
    fn one() -> Self;
    fn zero() -> Self;
}

impl<T> OneZero for T
where
    T: From<u8>,
{
    fn one() -> Self {
        T::from(1)
    }

    fn zero() -> Self {
        T::from(0)
    }
}