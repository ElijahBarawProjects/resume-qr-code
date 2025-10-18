pub trait OneZero {
    fn one() -> Self;
    fn zero() -> Self;
}

impl<T> OneZero for T
where
    T: TryFrom<u8>,
{
    #[inline]
    fn one() -> Self {
        T::from(1.try_into().ok().unwrap())
    }

    #[inline]
    fn zero() -> Self {
        T::from(0.try_into().ok().unwrap())
    }
}
