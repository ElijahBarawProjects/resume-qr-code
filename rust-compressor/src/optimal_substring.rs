#![allow(dead_code)] // TODO: fill in

// def max_score(
//     s: str, score_fn: Callable, allow_overlaps: bool = True, max_len: int = -1
// ) -> str:
//     """
//     Given a string `s` and a scoring function which takes in a substring and the
//     number of times it occurs, return a substring which achieves the maximum
//     score among all substrings.

//     Args:
//         s (str): Input string
//         score_fn (Callable): Function with signature (substring: str, count: int) -> int,
//             where count is the number of times the substring appears in `s`.
//         allow_overlaps (bool, optional): Whether to allow overlaps when counting substring.
//             Defaults to True.
//         max_len (int, optional): Maximum length of substrings to consider. Defaults to -1, which indicates 'no maximum length'.
//     """
//     print("¯\_(ツ)_/¯")
//     raise NotImplementedError

use std::{
    collections::{hash_map::Entry, HashMap},
    fmt::Debug,
    hash::Hash,
    ops::{Add, Mul, Sub},
};

use num_bigint::BigInt;

use crate::{
    non_rolling_hash::NonRollingHash,
    rolling_hash::{HashType, RollingHash},
};

pub trait _TCount: Copy + PartialOrd + Add<Output = Self> + Debug {}
impl<T> _TCount for T where T: Copy + PartialOrd + Add<Output = T> + Debug {}
pub trait TCount: _TCount {
    fn one() -> Self;
    fn zero() -> Self;
}

impl<T> TCount for T
where
    T: From<u8> + _TCount,
{
    fn one() -> Self {
        T::from(1)
    }
    fn zero() -> Self {
        T::from(0)
    }
}

pub trait Hasher<'a, TData, THash>: Iterator<Item = (THash, usize)> {
    fn new(
        s: &'a [TData],
        window_len: usize,
        base: THash,
        modulus: THash,
    ) -> Result<Self, &'static str>
    where
        Self: Sized;
}

pub trait HasherNew<'a, TData, THash>: Sized {
    fn new(
        s: &'a [TData],
        window_len: usize,
        base: THash,
        modulus: THash,
    ) -> Result<Self, &'static str>;
}

impl<'a, T, TData, THash> Hasher<'a, TData, THash> for T
where
    T: Iterator<Item = (THash, usize)> + HasherNew<'a, TData, THash>,
{
    fn new(
        s: &'a [TData],
        window_len: usize,
        base: THash,
        modulus: THash,
    ) -> Result<Self, &'static str>
    where
        Self: Sized,
    {
        T::new(s, window_len, base, modulus)
    }
}

impl<'a, TData, THash> HasherNew<'a, TData, THash> for RollingHash<'a, TData, THash>
where
    TData: Copy,
    THash: HashType + From<TData>,
{
    fn new(
        s: &'a [TData],
        window_len: usize,
        base: THash,
        modulus: THash,
    ) -> Result<Self, &'static str> {
        RollingHash::new(s, window_len, base, modulus)
    }
}

impl<'a, TData, THash> HasherNew<'a, TData, THash> for NonRollingHash<'a, TData, THash>
where
    TData: Copy + Into<BigInt>,
    THash: HashType + From<TData> + Into<BigInt> + TryFrom<BigInt>,
{
    fn new(
        s: &'a [TData],
        window_len: usize,
        base: THash,
        modulus: THash,
    ) -> Result<Self, &'static str> {
        Ok(NonRollingHash::new(s, window_len, base, modulus))
    }
}

fn substr_count_vec<THash: Eq + Hash, TSize: Copy, TData: Copy + PartialOrd + Eq>(
    len: usize,
    data: &[TData],
    hash_to_start: &HashMap<THash, usize>,
    hash_to_count: &HashMap<THash, TSize>,
) -> Vec<(Vec<TData>, TSize)> {
    let mut res = Vec::new();
    for (hash, start) in hash_to_start.iter() {
        let end = start + len;
        let substr = data[*start..end].to_vec();
        let count = hash_to_count.get(hash).unwrap();

        res.push((substr, *count))
    }
    res
}

pub fn most_common_of_len<
    'a,
    TData: Copy /*TODO: REMOVE */ + Debug + PartialEq + PartialOrd + Eq,
    TSize: TCount,
    TScore,
    THash: HashType + From<TData> + Hash + Eq + Debug,
    RollHasher: Hasher<'a, TData, THash>,
>(
    s: &'a [TData],
    validator_fn: &dyn Fn(&[TData], usize, usize) -> bool, // whether string is valid
    allow_overlap: bool,
    len: usize,
    base: THash,
    modulus: THash,
    check: bool,
) -> Option<(usize, TSize)> {
    if !allow_overlap {
        unimplemented!("Non-overlapping substring is not supported");
    }
    let mut max: Option<(THash, TSize)> = None;

    // maps hash -> start (length is fixed)
    let mut hash_to_start: HashMap<THash, usize> = HashMap::new();

    // maps hash -> count
    let mut hash_to_count: HashMap<THash, TSize> = HashMap::new();

    for (hash, start) in RollHasher::new(s, len, base, modulus).unwrap() {
        // map hash -> start, hash -> count

        let count: TSize;
        let count_entry = hash_to_count.entry(hash);
        match count_entry {
            Entry::Occupied(mut occupied_count) => {
                count = *occupied_count.get_mut() + TSize::one();
                *occupied_count.get_mut() = count;
                assert!(hash_to_start.contains_key(&hash));
                if check {
                    let prev_start = *hash_to_start.get(&hash).unwrap();
                    assert_eq!(s[start..start + len], s[prev_start..prev_start + len]);
                }
            }
            Entry::Vacant(vacant_count) => {
                count = TSize::one();
                vacant_count.insert(count);
                assert!(hash_to_start.insert(hash, start).is_none())
            }
        }

        if !validator_fn(s, start, len) {
            continue;
        }
        match max {
            Some((_hash, c)) => {
                if c < count {
                    max = Some((hash, count))
                }
            }
            None => max = Some((hash, count)),
        }
    }

    println!(
        "{:?}",
        substr_count_vec(len, s, &hash_to_start, &hash_to_count)
    );

    // return the most common valid substring
    match max {
        Some((hash, count)) => {
            let start = *hash_to_start
                .get(&hash)
                .expect("Substring must be encountered");
            Some((start, count))
        }
        None => None,
    }
}

fn max_score<
    'a,
    TData: Copy /* delete */ + Debug + PartialEq + PartialOrd + Eq,
    TSize: TCount,
    TScore: PartialOrd + Copy,
    THash: HashType + From<TData> + Hash + Eq + Debug,
>(
    s: &'a [TData],
    score_fn: Box<dyn Fn(&[TData], usize, TSize) -> TScore>, // slice, length, count => score
    validator_fn: Box<dyn Fn(&[TData], usize, usize) -> bool>, // slice, start, len
    allow_overlap: bool,
    min_len: usize,
    max_len: usize,
    base: THash,
    modulus: THash,
    check: bool,
) -> Option<(TScore, &'a [TData], TSize)> {
    let mut max: Option<(TScore, &[TData], TSize)> = None;
    for len in min_len..=max_len {
        let res = most_common_of_len::<_, TSize, TScore, _, RollingHash<'a, TData, THash>>(
            s,
            &validator_fn,
            allow_overlap,
            len,
            base,
            modulus,
            check,
        );

        match (res, max) {
            (None, _) => continue,
            (Some((start, count)), _m) => {
                let slice = s.get(start..start + len).expect("");
                let new_score = score_fn(slice, len, count);
                match max {
                    None => max = Some((new_score, slice, count)),
                    Some((old_score, old_slice, old_count)) => {
                        max = if new_score > old_score {
                            Some((new_score, slice, count))
                        } else {
                            Some((old_score, old_slice, old_count))
                        }
                    }
                }
            }
        }
    }

    max
}

#[derive(Debug)]
pub struct BestSubstringRes<TSize, TScore> {
    score: TScore,
    count: TSize,
    substring: String,
}

pub fn best_substring<
    'a,
    THash: HashType + From<char> + PartialOrd + Hash + Eq + Debug,
    TSize: TCount + Mul<Output = TSize> + Sub<Output = TSize> + From<usize>,
>(
    s: &'a str,
    delimiter: char,
    prev_was_disallowed: bool,
    key: &'a str,
    allow_overlap: bool,
    min_len: usize,
    max_len: usize,
    base: THash,
    modulus: THash,
    check: bool,
) -> Option<BestSubstringRes<TSize, TSize>> {
    let validator: Box<dyn Fn(&[char], usize, usize) -> bool>;
    if !s.contains(delimiter) {
        validator = Box::new(|_substring, _start, _length| true);
    } else {
        validator = Box::new(move |substring, start, length| {
            !substring
                .get(start..start + length)
                .expect("")
                .contains(&delimiter)
        })
    }

    let get_savings = {
        let key_len = key.len(); // Copy the length instead of borrowing key
        Box::new(move |_full_str: &[char], len: usize, count: TSize| {
            if len < key_len {
                return TSize::zero();
            }

            let mut word_saving: TSize = count * (len - key_len).into();
            let worddict_cost: TSize = (len + 1).into();
            if worddict_cost > word_saving {
                return TSize::zero();
            }

            word_saving = word_saving - worddict_cost;
            if word_saving == TSize::zero() {
                return TSize::zero();
            }

            if prev_was_disallowed {
                word_saving = word_saving - TSize::one();
            }
            word_saving
        })
    };

    let chars: Vec<char> = s.chars().collect();

    let best = max_score(
        &chars,
        get_savings,
        validator,
        allow_overlap,
        min_len,
        max_len,
        base,
        modulus,
        check,
    );
    match best {
        None => None,
        Some((score, substr, count)) => Some(BestSubstringRes {
            score,
            count,
            substring: substr.iter().collect(),
        }),
    }
}

#[cfg(test)]
mod tests {
    // check against known
    // check against brute force reference
}
