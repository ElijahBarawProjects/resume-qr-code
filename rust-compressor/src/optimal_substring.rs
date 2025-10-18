use std::{
    collections::{hash_map::Entry, HashMap},
    fmt::Debug,
    hash::{BuildHasher, Hash},
    ops::{Add, Mul, Sub},
    thread,
};

use crate::{interface::WindowHasher, rolling_hash::HashType};

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
    THasher: WindowHasher<THash, TData>,
    S: BuildHasher + Default,
>(
    s: &'a [TData],
    validator_fn: &dyn Fn(&[TData], usize, usize) -> bool, // whether string is valid
    allow_overlap: bool,
    len: usize,
    window_hasher: &THasher,
    check: bool,
) -> Option<(usize, TSize)> {
    if !allow_overlap {
        unimplemented!("Non-overlapping substring is not supported");
    }

    struct Best<THash, TSize> {
        hash: THash,
        count: TSize,
        start: usize,
    }
    let mut max: Option<Best<THash, TSize>> = None;

    struct Cell<TSize> {
        count: TSize,
        start: usize,
    }

    type THashMap<A, B, C> = HashMap<A, B, C>;
    let mut hash_to_start_count = THashMap::<THash, Cell<TSize>, S>::with_hasher(S::default());
    for (hash, start) in window_hasher
        .sliding_hash(s, len)
        .filter(|(_hash, start)| validator_fn(s, *start, len))
    {
        // map hash -> start, hash -> count

        let stored_start;
        let count: TSize;

        let count_entry = hash_to_start_count.entry(hash);
        match count_entry {
            Entry::Occupied(mut occupied_count) => {
                // get
                let stored_count;
                Cell {
                    start: stored_start,
                    count: stored_count,
                } = *occupied_count.get();
                // update count
                count = stored_count + TSize::one();
                (*occupied_count.get_mut()).count = count;
                if check {
                    assert_eq!(s[start..start + len], s[stored_start..stored_start + len]);
                }
            }
            Entry::Vacant(vacant_count) => {
                count = TSize::one();
                stored_start = start;
                vacant_count.insert(Cell {
                    start: stored_start,
                    count,
                });
            }
        }

        match max {
            Some(Best {
                hash: _hash,
                count: prev_max_count,
                start: curr_max_start,
            }) => {
                if (prev_max_count < count)
                    // Design Decision
                    // break ties by preferring the earliest first occurence of the substring
                    || ((prev_max_count == count) && (stored_start < curr_max_start))
                {
                    max = Some(Best {
                        hash,
                        count,
                        start: stored_start,
                    })
                }
            }
            None => {
                max = Some(Best {
                    hash,
                    count,
                    start: stored_start,
                })
            }
        }
    }

    // #[cfg(test)]
    // println!(
    //     "{:?}",
    //     substr_count_vec(len, s, &hash_to_start, &hash_to_count)
    // );

    // return the most common valid substring
    match max {
        Some(Best {
            hash: _,
            count,
            start,
        }) => Some((start, count)),
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
    THasher: WindowHasher<THash, TData>,
    S: BuildHasher + Default,
>(
    s: &'a [TData],
    score_fn: TScorer,
    validator_fn: TValidator,
    allow_overlap: bool,
    min_len: usize,
    max_len: usize,
    hasher: &THasher,
    check: bool,
) -> (Option<(TScore, &'a [TData], TSize)>, Option<usize>) {
    // TODO: potential improvement: have delimiter be an Option<...> param in this function, instead of using a validator fn
    // this allows us to maintain a count of the delimiter, knowing in O(1) if the number of occurences is zero or positive
    let mut max: Option<(TScore, &[TData], TSize)> = None;
    let mut longest_with_count_gt_1: Option<usize> = None;

    for len in min_len..=max_len {
        let res = most_common_of_len::<TData, TSize, TScore, THash, THasher, S>(
            s,
            &validator_fn,
            allow_overlap,
            len,
            &hasher,
            check,
        );

        match (res, max) {
            (None, _) => continue,
            (Some((start, count)), _m) => {
                if count > TSize::one() {
                    // we evaluate lengths in order, so future iterations will only overwrite
                    longest_with_count_gt_1 = Some(len)
                }

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

    (max, longest_with_count_gt_1)
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
    THasher: WindowHasher<THash, char> + Sync,
    S: BuildHasher + Default,
>(
    chars: &Vec<char>,
    delimiter: char,
    prev_was_disallowed: bool,
    key: &'a str,
    allow_overlap: bool,
    min_len: usize,
    max_len: usize,
    hasher: &THasher,
    check: bool,
    n: usize,
) -> Option<BestSubstringRes<TSize, TSize>> {
    best_substring_and_longest_gt_one::<_, _, _, S>(
        chars,
        delimiter,
        prev_was_disallowed,
        key,
        allow_overlap,
        min_len,
        max_len,
        hasher,
        check,
        n,
    )
    .0
}

pub fn best_substring_and_longest_gt_one<
    'a,
    THash: HashType + From<char> + PartialOrd + Hash + Eq + Debug + std::marker::Sync + std::marker::Send,
    TSize: TCount
        + Mul<Output = TSize>
        + Sub<Output = TSize>
        + From<usize>
        + std::marker::Send
        + std::marker::Send,
    THasher: WindowHasher<THash, char> + Sync,
    S: BuildHasher + Default,
>(
    chars: &Vec<char>,
    delimiter: char,
    prev_was_disallowed: bool,
    key: &'a str,
    allow_overlap: bool,
    min_len: usize,
    max_len: usize,
    hasher: &THasher,
    check: bool,
    n: usize,
) -> (Option<BestSubstringRes<TSize, TSize>>, Option<usize>) {
    assert!(n > 0);
    let validator = {
        let could_not_contain_delim = !chars.contains(&delimiter);
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

    let (best, longest_gt_1) = if n == 1 {
        max_score::<_, _, _, _, _, _, _, S>(
            chars,
            get_savings,
            validator,
            allow_overlap,
            min_len,
            max_len,
            hasher,
            check,
        )
    } else {
        max_score_par::<_, _, _, _, _, _, _, S>(
            &chars,
            get_savings,
            validator,
            allow_overlap,
            min_len,
            max_len,
            hasher,
            check,
            n,
        )
    };
    let best = match best {
        None => None,
        Some((score, substr, count)) => Some(BestSubstringRes {
            score,
            count,
            substring: substr.iter().collect(),
        }),
    };

    (best, longest_gt_1)
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
    THasher: WindowHasher<THash, TData> + Sync,
    S: BuildHasher + Default,
>(
    s: &'a [TData],
    score_fn: TScorer,
    validator_fn: TValidator,
    allow_overlap: bool,
    min_len: usize,
    max_len: usize,
    hasher: &THasher,
    check: bool,
    n: usize,
) -> (Option<(TScore, &'a [TData], TSize)>, Option<usize>) {
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
                max_score::<TData, TSize, TScore, THash, &TScorer, &TValidator, THasher, S>(
                    s,
                    score_fn,
                    validator_fn,
                    allow_overlap,
                    thread_min,
                    thread_max,
                    hasher,
                    check,
                )
            });
            handles.push(handle);
        }

        // collect, waiting on all
        let mut best: Option<(TScore, &'a [TData], TSize)> = None;
        let mut longest_gt_one: Option<usize> = None;

        for (res, cur_longest_gt_one) in handles.into_iter().map(|handle| handle.join().unwrap()) {
            match (res, best) {
                (None, _) => (),
                (Some(_), None) => best = res,
                (Some((score, ..)), Some((best_score, ..))) => {
                    if score > best_score {
                        best = res
                    }
                }
            };

            if let Some(len) = cur_longest_gt_one {
                longest_gt_one = match longest_gt_one {
                    Some(prev_len) => Some(prev_len.max(len)),
                    None => cur_longest_gt_one,
                }
            }
        }
        (best, longest_gt_one)
    })
}

#[cfg(test)]
mod tests {
    // TODO: Test
    // check against known
    // check against brute force reference
}
