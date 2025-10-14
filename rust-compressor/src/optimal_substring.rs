use std::{
    collections::{hash_map::Entry, HashMap},
    fmt::Debug,
    hash::Hash,
    ops::{Add, Mul, Sub},
    thread,
};

use crate::{
    interface::ModWindowHasher,
    rolling_hash::{HashType, WrappedRollingHash},
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

/// Get a vector of (substring, count) for each substring in `data` of length `len`
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

/// Get the most common substring of length `len`
///
/// Assumptions:
/// - hashmap is O(1)
/// - THash ops are O(1)
///
/// Work:
/// - number of substrings = |data| - len + 1
/// - building initial hash: O(len)
/// - work per substring, each of which has length len: O(1)
/// - => O(|data| - len + 1) + O(len) = O(|data|)
pub fn most_common_of_len<
    'a,
    TData: Copy /*TODO: REMOVE */ + Debug + PartialEq + PartialOrd + Eq,
    TSize: TCount,
    TScore,
    THash: HashType + From<TData> + Hash + Eq + Debug,
    RollHasher: ModWindowHasher<THash, TData>,
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

    let hasher = RollHasher::new(base, modulus).unwrap();

    for (hash, start) in hasher.sliding_hash_owned(s, len) {
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

/// Given a `score_fn`, return the substring of `s` which has the highest score
/// and has `validator_fn` evaluate to `true`
///
/// Evaluates lens on [min_len, max_len] (*inclusive*)
fn max_score<
    'a,
    TData: Copy /* delete */ + Debug + PartialEq + PartialOrd + Eq,
    TSize: TCount,
    TScore: PartialOrd + Copy,
    THash: HashType + From<TData> + Hash + Eq + Debug,
    TScorer: Fn(&[TData], usize, TSize) -> TScore, // slice, length, count => score
    TValidator: Fn(&[TData], usize, usize) -> bool, // slice, start, len
>(
    s: &'a [TData],
    score_fn: TScorer,
    validator_fn: TValidator,
    allow_overlap: bool,
    min_len: usize,
    max_len: usize,
    base: THash,
    modulus: THash,
    check: bool,
) -> Option<(TScore, &'a [TData], TSize)> {
    // TODO: potential improvement: have delimiter be an Option<...> param in this function, instead of using a validator fn
    // this allows us to maintain a count of the delimiter, knowing in O(1) if the number of occurences is zero or positive
    let mut max: Option<(TScore, &[TData], TSize)> = None;
    for len in min_len..=max_len {
        let res = most_common_of_len::<_, TSize, TScore, _, WrappedRollingHash<THash>>(
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

#[derive(Debug, PartialEq)]
pub struct BestSubstringRes<TSize, TScore> {
    pub score: TScore,
    pub count: TSize,
    pub substring: String,
}

pub fn best_substring<
    'a,
    THash: HashType + From<char> + PartialOrd + Hash + Eq + Debug + std::marker::Sync + std::marker::Send,
    TSize: TCount
        + Mul<Output = TSize>
        + Sub<Output = TSize>
        + From<usize>
        + std::marker::Send
        + std::marker::Send,
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
    n: usize,
) -> Option<BestSubstringRes<TSize, TSize>> {
    assert!(n > 0);
    let validator = {
        let could_not_contain_delim = !s.contains(delimiter);
        Box::new(move |substring: &[char], start, length| {
            could_not_contain_delim
                || !substring
                    .get(start..start + length)
                    .unwrap()
                    .contains(&delimiter)
        })
    };

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

    let best = if n == 1 {
        max_score(
            &chars,
            get_savings,
            validator,
            allow_overlap,
            min_len,
            max_len,
            base,
            modulus,
            check,
        )
    } else {
        max_score_par(
            &chars,
            get_savings,
            validator,
            allow_overlap,
            min_len,
            max_len,
            base,
            modulus,
            check,
            n,
        )
    };
    match best {
        None => None,
        Some((score, substr, count)) => Some(BestSubstringRes {
            score,
            count,
            substring: substr.iter().collect(),
        }),
    }
}

/// Given a `score_fn`, return the substring of `s` which has the highest score
/// and has `validator_fn` evaluate to `true`
///
/// Do so using n OS threads
///
/// If thread creation fails, panic
pub fn max_score_par<
    'a,
    TData: Copy + Debug + PartialEq + PartialOrd + Eq + Sync,
    TSize: TCount + Send,
    TScore: PartialOrd + Copy + Send,
    THash: HashType + From<TData> + Hash + Eq + Debug + Sync + Copy + Send,
    TScorer: Fn(&[TData], usize, TSize) -> TScore + Sync, // slice, length, count => score
    TValidator: Fn(&[TData], usize, usize) -> bool + Sync, // slice, start, len
>(
    s: &'a [TData],
    score_fn: TScorer,
    validator_fn: TValidator,
    allow_overlap: bool,
    min_len: usize,
    max_len: usize,
    base: THash,
    modulus: THash,
    check: bool,
    n: usize,
) -> Option<(TScore, &'a [TData], TSize)> {
    assert!(n > 0);
    let num_rounds = max_len - min_len + 1;
    let all_threads_get = num_rounds / n;
    let remainder = num_rounds % n;
    assert!(remainder < n); // cause int division

    thread::scope(|scope| {
        let mut handles = Vec::with_capacity(n);
        let mut next_min = min_len;

        let score_fn = &score_fn;
        let validator_fn = &validator_fn;

        // run in par, using static work assignment
        for i in 0..n {
            let rounds = all_threads_get + if i < remainder { 1 } else { 0 };
            if rounds == 0 {
                continue;
            }

            let thread_min = next_min;
            let thread_max = thread_min + rounds - 1;
            next_min += rounds;

            let handle = scope.spawn(move || {
                max_score(
                    s,
                    score_fn,
                    validator_fn,
                    allow_overlap,
                    thread_min,
                    thread_max,
                    base,
                    modulus,
                    check,
                )
            });
            handles.push(handle);
        }

        // collect, waiting on all
        let mut best: Option<(TScore, &'a [TData], TSize)> = None;
        for res in handles.into_iter().map(|handle| handle.join().unwrap()) {
            match (res, best) {
                (None, _) => (),
                (Some(_), None) => best = res,
                (Some((score, ..)), Some((best_score, ..))) => {
                    if score > best_score {
                        best = res
                    }
                }
            };
        }
        best
    })
}

#[cfg(test)]
mod tests {
    // TODO: Test
    // check against known
    // check against brute force reference
}
