#[cfg(test)]
use tested_trait::test_impl;

use num_bigint::BigInt;
use num_iter::range;

use crate::interface::{HasherErr, WindowHasher};

#[derive(Clone)]
pub struct InterfaceHasher {
    base: BigInt,
    modulus: BigInt,
}

impl<'a> InterfaceHasher {
    fn build_hash<TData: Copy + Into<BigInt> + 'a>(&self, data: &'a [TData]) -> BigInt {
        let mut hash_value: BigInt = 0.into();
        for idx in range(0, data.len()) {
            let char = data[idx];
            // add char
            hash_value *= self.base.clone();
            hash_value += char.into();
            // mod
            hash_value %= self.modulus.clone();
            hash_value += self.modulus.clone();
            hash_value %= self.modulus.clone();
        }
        hash_value
    }
}

#[cfg(test)]
use crate::interface::tests::{
    WindowHasherDataSizeTests,
    WindowHasherDataTests,
    WindowHasherWindowSizeTests,
    WindowHasherModulusTests,
    WindowHasherLargeUnsignedTests,
    WindowHasherLargeSignedTests,
    WindowhasherBoundedTDataTests,
    WindowHasherBoundedTHashTests,
    WindowHasherSignedDataTests,
    WindowHasherErrorHandlingTests,
    WindowHasherIteratorTests,
    WindowHasherHashPropertyTests
};

#[cfg_attr(test, test_impl(
    InterfaceHasher: WindowHasher<u8, u8>, 
    InterfaceHasher: WindowHasher<i8, i8>, 
    InterfaceHasher: WindowHasher<u32, u8>,
    InterfaceHasher: WindowHasherDataSizeTests<u32, u32>,
    InterfaceHasher: WindowHasherDataTests<u32, u32>,
    InterfaceHasher: WindowHasherWindowSizeTests<u32, u32>,
    InterfaceHasher: WindowHasherModulusTests<u32, u32>,
    InterfaceHasher: WindowHasherLargeUnsignedTests<u32, u32>,
    InterfaceHasher: WindowHasherLargeSignedTests<i32, i32>,
    InterfaceHasher: WindowhasherBoundedTDataTests<u32, u32>,
    InterfaceHasher: WindowHasherBoundedTHashTests<u32, u32>,
    InterfaceHasher: WindowHasherSignedDataTests<u32, i32>,
    InterfaceHasher: WindowHasherErrorHandlingTests<i32, i8>,
    InterfaceHasher: WindowHasherErrorHandlingTests<i32, u8>,
    InterfaceHasher: WindowHasherIteratorTests<u32, u8>,
    InterfaceHasher: WindowHasherIteratorTests<i32, i8>,
    InterfaceHasher: WindowHasherHashPropertyTests<u32, u8>,
    InterfaceHasher: WindowHasherHashPropertyTests<i32, i16>,
))]
impl<THash, TData> WindowHasher<THash, TData> for InterfaceHasher
where
    TData: Copy + Into<BigInt>,
    THash: TryFrom<BigInt> + Into<BigInt> + Copy,
{
    fn new(base: THash, modulus: THash) -> Result<Self, crate::interface::HasherErr> {
        let base = base.try_into().unwrap();
        let modulus = modulus.try_into().unwrap();

        if modulus == 0.into() {
            return Err(HasherErr::InvalidModulus("Zero is not a valid modulus"));
        }
        if base <= 0.into() {
            return Err(HasherErr::InvalidBase("Non-positive base"));
        }

        Ok(InterfaceHasher { base, modulus })
    }

    fn hash(&self, data: &[TData]) -> THash {
        // we don't expect panics here as the built hash, while a BigInt, will
        // fit in THash since it's smaller in magnitude than modulus
        self.build_hash(data).try_into().ok().unwrap()
    }
}