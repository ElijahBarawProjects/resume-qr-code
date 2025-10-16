use num_traits::{ops::overflowing::OverflowingAdd, CheckedMul};
use std::ops::{Add, BitAnd, Mul, Rem, Shr, Sub};

use crate::{
    interface::{ModWindowHasher, WindowHasher},
    modular::Mod,
    one_zero::OneZero,
};

// Mersenne primes
pub const DEFAULT_MOD_U64: u64 = (1 << 61) - 1;
#[allow(unused)] // TODO REMOVE
pub const DEFAULT_MOD_U32: u32 = (1 << 31) - 1;
// prime & works well for ascii
pub const DEFAULT_BASE: u16 = 257;

#[derive(Clone, Copy)]
pub struct HasherConfig<THash: HashType> {
    base: THash,
    modulus: Mod<THash>,
}

const MOD_INVALID_STR: &str = "Modulus is not valid";
const BASE_INVALID_STR: &str = "Non-positive base";

impl<THash: HashType> HasherConfig<THash> {
    pub fn new(base: THash, modulus: THash) -> Result<Self, &'static str> {
        if base <= THash::zero() {
            return Err(BASE_INVALID_STR);
        }

        let _mod: Mod<THash>;
        match Mod::new(modulus) {
            Some(modulus) => _mod = modulus,
            None => return Err(MOD_INVALID_STR),
        };

        Ok(Self {
            base,
            modulus: _mod,
        })
    }
}

#[derive(Clone, Copy)]
pub struct _RollingHash<'a, TData, THash>
where
    TData: Copy,
    THash: HashType,
{
    data: &'a [TData],
    window_size: usize,
    conf: HasherConfig<THash>,
    curr_start: usize,
    curr_hash: THash,
    highest_power: THash, // (base ^ (window_size - 1)) mod modulus
}

#[derive(Clone, Copy)]
pub enum RollingHash<'a, TData, THash>
where
    TData: Copy,
    THash: HashType,
{
    Empty,
    NonEmpty(_RollingHash<'a, TData, THash>),
}

impl<'a, TData, THash> RollingHash<'a, TData, THash>
where
    TData: Copy,
    THash: HashType + From<TData>,
{
    pub fn new_with_conf(data: &'a [TData], window_size: usize, conf: HasherConfig<THash>) -> Self {
        if window_size == 0 || window_size > data.len() {
            return Self::Empty;
        };

        // non empty

        let modulus = conf.modulus;

        // base^(window_size-1) mod modulus
        let highest_power = modulus.mod_pow(conf.base, (window_size - 1) as u64);

        // Calculate initial hash for first window
        let mut curr_hash = THash::zero();
        for i in 0..window_size {
            curr_hash =
                modulus.mod_add(modulus.mod_mul(curr_hash, conf.base), THash::from(data[i]));
        }

        let state = _RollingHash {
            data,
            window_size,
            conf,
            curr_start: 0,
            curr_hash,
            highest_power,
        };
        Self::NonEmpty(state)
    }

    pub fn new(
        data: &'a [TData],
        window_size: usize,
        base: THash,
        modulus: THash,
    ) -> Result<Self, &'static str> {
        Ok(Self::new_with_conf(
            data,
            window_size,
            HasherConfig::new(base, modulus)?,
        ))
    }
}

impl<'a, TData, THash> Iterator for RollingHash<'a, TData, THash>
where
    TData: Copy,
    THash: From<TData> + HashType,
{
    type Item = (THash, usize);

    fn next(&mut self) -> Option<Self::Item> {
        let state;
        match self {
            RollingHash::Empty => {
                return None;
            }
            RollingHash::NonEmpty(_state) => {
                state = _state;
            }
        };

        let current_end = state.curr_start + state.window_size; // exclusive
        if current_end > state.data.len() {
            return None;
        }

        // we have a result computed
        let result = Some((state.curr_hash, state.curr_start));

        let next_end = current_end + 1;
        if next_end > state.data.len() {
            state.curr_start += 1;
            return result;
        }

        // we have a next iteration, compute it now
        let old_char = state.data[state.curr_start];
        // remove the previous character
        state.curr_hash = state.conf.modulus.mod_sub(
            state.curr_hash,
            state
                .conf
                .modulus
                .mod_mul(state.highest_power, old_char.into()),
        );
        let new_char = state.data[state.curr_start + state.window_size].into();
        // add the new character
        state.curr_hash = state.conf.modulus.mod_add(
            state.conf.modulus.mod_mul(state.curr_hash, state.conf.base),
            new_char,
        );

        state.curr_start += 1;
        result
    }
}

// Numeric traits needed for hashing
pub trait _HashType:
     Copy
    // math ops
    + Add<Output = Self>
    + Mul<Output = Self>
    + Sub<Output = Self>
    + Rem<Output = Self>
    + BitAnd<Output = Self>
    + Shr<Output=Self>
    // equality
    + PartialOrd
    // for modular multiplication
    + CheckedMul
    + OverflowingAdd
{
}

impl<T> _HashType for T where
    T: Copy
        + Add<Output = Self>
        + Mul<Output = Self>
        + Sub<Output = Self>
        + Rem<Output = Self>
        + BitAnd<Output = Self>
        + Shr<Output = Self>
        + PartialOrd
        + CheckedMul
        + OverflowingAdd
{
}

// Type that can be used for rolling hash
pub trait HashType: _HashType + OneZero {}

// Automatically impliment HashType for all types which can be created from u8
// and implement _HashType (including all base integral types)
impl<T> HashType for T where T: TryFrom<u8> + _HashType {}

impl<TData, THash> WindowHasher<THash, TData> for HasherConfig<THash>
where
    TData: Copy,
    THash: From<TData> + HashType,
{
    fn hash<'data>(&self, data: &'data [TData]) -> THash {
        // we've already validated base, mod

        if data.len() == 0 {
            return THash::zero();
        }

        let mut hasher = RollingHash::new_with_conf(data, data.len(), *self);
        let res = hasher.next().unwrap().0;
        assert!(hasher.next() == None);
        res
    }

    fn sliding_hash<'data>(
        &self,
        data: &'data [TData],
        window_size: usize,
    ) -> impl Iterator<Item = (THash, usize)> + 'data
    where
        Self: 'data,
    {
        // we've already validated base, mod
        RollingHash::new_with_conf(data, window_size, *self)
    }
}

#[cfg(test)]
use crate::interface::tests::{
    WindowHasherBoundedTHashTests, WindowHasherDataSizeTests, WindowHasherDataTests,
    WindowHasherErrorHandlingTests, WindowHasherHashPropertyTests, WindowHasherIteratorTests,
    WindowHasherLargeSignedTests, WindowHasherLargeUnsignedTests, WindowHasherModulusTests,
    WindowHasherSignedDataTests, WindowHasherWindowSizeTests, WindowhasherBoundedTDataTests,
};
#[cfg(test)]
use tested_trait::test_impl;
#[cfg_attr(test, test_impl(
    HasherConfig<u8>: ModWindowHasher<u8, u8>,
    HasherConfig<i8>: ModWindowHasher<i8, i8>,
    HasherConfig<u32>: ModWindowHasher<u32, u8>,
    HasherConfig<u32>: WindowHasherDataSizeTests<u32, u32>,
    HasherConfig<u32>: WindowHasherDataTests<u32, u32>,
    HasherConfig<i32>: WindowHasherDataTests<i32, u8>,
    HasherConfig<u32>: WindowHasherWindowSizeTests<u32, u32>,
    HasherConfig<u32>: WindowHasherModulusTests<u32, u32>,
    HasherConfig<u32>: WindowHasherLargeUnsignedTests<u32, u32>,
    HasherConfig<i32>: WindowHasherLargeSignedTests<i32, i32>,
    HasherConfig<u32>: WindowhasherBoundedTDataTests<u32, u32>,
    HasherConfig<u32>: WindowHasherBoundedTHashTests<u32, u32>,
    HasherConfig<i32>: WindowHasherSignedDataTests<i32, i32>,
    HasherConfig<i32>: WindowHasherSignedDataTests<i32, i8>,
    HasherConfig<i32>: WindowHasherErrorHandlingTests<i32, i8>,
    HasherConfig<u32>: WindowHasherIteratorTests<u32, u8>,
    HasherConfig<i32>: WindowHasherIteratorTests<i32, i8>,
    HasherConfig<u32>: WindowHasherHashPropertyTests<u32, u8>,
    HasherConfig<i32>: WindowHasherHashPropertyTests<i32, i16>,
))]
impl<TData, THash> ModWindowHasher<THash, TData> for HasherConfig<THash>
where
    TData: Copy,
    THash: From<TData> + HashType,
{
    fn new(base: THash, modulus: THash) -> Result<Self, crate::interface::HasherErr> {
        match Self::new(base, modulus) {
            Ok(res) => Ok(res),
            Err(err) => match err {
                MOD_INVALID_STR => {
                    Err(crate::interface::HasherErr::InvalidModulus(MOD_INVALID_STR))
                }
                BASE_INVALID_STR => Err(crate::interface::HasherErr::InvalidBase(BASE_INVALID_STR)),
                _ => unreachable!(),
            },
        }
    }
}
