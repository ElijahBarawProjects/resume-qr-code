use num_traits::{ops::overflowing::OverflowingAdd, CheckedMul};
use std::ops::{Add, BitAnd, Mul, Rem, Shr, Sub};

use crate::{modular::Mod, one_zero::OneZero};

// Mersenne primes
pub const DEFAULT_MOD_U64: u64 = 1 << 61 - 1;
pub const DEFAULT_MOD_U32: u32 = 1 << 31 - 1;
// prime & works well for ascii
pub const DEFAULT_BASE: u16 = 257;

pub struct _RollingHash<'a, TData, THash>
where
    TData: Copy,
    THash: HashType,
{
    data: &'a [TData],
    window_size: usize,
    base: THash,
    modulus: Mod<THash>,
    curr_start: usize,
    curr_hash: THash,
    highest_power: THash, // (base ^ (window_size - 1)) mod modulus
}

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
    pub fn new(
        data: &'a [TData],
        window_size: usize,
        base: THash,
        modulus: THash,
    ) -> Result<Self, &'static str> {
        let _mod: Mod<THash>;
        match Mod::new(modulus) {
            Some(modulus) => _mod = modulus,
            None => return Err("Modulus is not valid"),
        }

        if window_size == 0 || window_size > data.len() {
            return Ok(Self::Empty);
        }

        let modulus = _mod;

        // base^(window_size-1) mod modulus
        let highest_power = modulus.mod_pow(base, (window_size - 1) as u64);

        // Calculate initial hash for first window
        let mut curr_hash = THash::zero();
        for i in 0..window_size {
            curr_hash = modulus.mod_add(modulus.mod_mul(curr_hash, base), THash::from(data[i]));
        }

        let state = _RollingHash {
            data,
            window_size,
            base,
            modulus,
            curr_start: 0,
            curr_hash,
            highest_power,
        };
        Ok(Self::NonEmpty(state))
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
        state.curr_hash = state.modulus.mod_sub(
            state.curr_hash,
            state.modulus.mod_mul(state.highest_power, old_char.into()),
        );
        let new_char = state.data[state.curr_start + state.window_size].into();
        // add the new character
        state.curr_hash = state
            .modulus
            .mod_add(state.modulus.mod_mul(state.curr_hash, state.base), new_char);

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
impl<T> HashType for T where T: From<u8> + _HashType {}

#[cfg(test)]
mod tests {
    // check for overlapping
    // check against reference rolling hash
}
