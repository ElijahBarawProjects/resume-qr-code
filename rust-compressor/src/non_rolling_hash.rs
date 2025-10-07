use num_iter::{range, range_inclusive};

use num_bigint::BigInt;
use std::marker::PhantomData;

// reference implementation of rolling hash which doesn't actually 'roll' the hash

pub struct NonRollingHash<'a, TData, THash> {
    starts: num_iter::RangeInclusive<BigInt>,
    window_size: usize,
    base: BigInt,
    modulus: BigInt,
    data: &'a [TData],
    _phantom: PhantomData<THash>,
}

impl<'a, TData, THash> NonRollingHash<'a, TData, THash>
where
    TData: Copy + Into<BigInt>,
    THash: TryFrom<BigInt> + Into<BigInt> + Copy,
{
    pub fn new(data: &'a [TData], window_size: usize, base: THash, modulus: THash) -> Self {
        let max_start: BigInt = BigInt::from(data.len()) - BigInt::from(window_size);
        let starts: num_iter::RangeInclusive<BigInt> = range_inclusive(0.into(), max_start);
        let base: BigInt = base.try_into().unwrap();
        let big_modulus: BigInt = modulus.try_into().unwrap();
        Self {
            starts,
            window_size,
            base,
            modulus: big_modulus,
            data,
            _phantom: PhantomData,
        }
    }

    fn build_hash(&self, start: usize) -> BigInt {
        let mut hash_value: BigInt = 0.into();
        let end = start + self.window_size;
        for idx in range(start, end) {
            let char = self.data[idx];
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

impl<'a, TData, THash> Iterator for NonRollingHash<'a, TData, THash>
where
    TData: Copy + Into<BigInt>,
    THash: TryFrom<BigInt> + Into<BigInt> + Copy + std::fmt::Debug,
{
    type Item = (THash, usize);

    fn next(&mut self) -> Option<Self::Item> {
        match self.starts.next() {
            None => None,
            Some(start) => {
                let start: usize = start.try_into().unwrap();
                let hash = self.build_hash(start);
                Some((THash::try_from(hash).ok().unwrap(), start))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::NonRollingHash;

    #[test]
    fn test_empty_data() {
        let data: Vec<u8> = vec![];
        let hasher = NonRollingHash::new(&data, 1, 257u32, 1000000007u32);
        let results: Vec<_> = hasher.collect();
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_window_size_larger_than_data() {
        let data = vec![1u8, 2, 3];
        let hasher = NonRollingHash::new(&data, 5, 257u32, 1000000007u32);
        let results: Vec<_> = hasher.collect();
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_window_size_equal_to_data() {
        let data = vec![1u8, 2, 3];
        let hasher = NonRollingHash::new(&data, 3, 257u32, 1000000007u32);
        let results: Vec<_> = hasher.collect();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], (66_566, 0)); // start position should be 0
    }

    #[test]
    fn test_single_character_window() {
        let data = vec![65u8, 66, 67]; // "ABC"
        let hasher = NonRollingHash::new(&data, 1, 257u32, 1000000007u32);
        let results: Vec<_> = hasher.collect();

        assert_eq!(results.len(), 3);
        assert_eq!(results[0], (65u32, 0));
        assert_eq!(results[1], (66u32, 1));
        assert_eq!(results[2], (67u32, 2));
    }

    #[test]
    fn test_hash_calculation_manual() {
        let data = vec![1u8, 2];
        let base = 10u32;
        let modulus = 97u32;
        let hasher = NonRollingHash::new(&data, 2, base, modulus);
        let results: Vec<_> = hasher.collect();

        assert_eq!(results.len(), 1);
        // Manual calculation: (1 * 10 + 2) % 97 = 12
        assert_eq!(results[0], (12, 0));
    }

    #[test]
    fn test_hash_calculation_with_modulo() {
        let data = vec![100u8, 100, 100];
        let base = 10u32;
        let modulus = 7u32;
        let hasher = NonRollingHash::new(&data, 2, base, modulus);
        let results: Vec<_> = hasher.collect();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0], (1, 0));
        assert_eq!(results[1], (1, 1));
    }

    #[test]
    fn test_overlapping_windows() {
        let data = vec![1u8, 2, 3, 4];
        let base = 10u32;
        let modulus = 1000000007u32;
        let hasher = NonRollingHash::new(&data, 2, base, modulus);
        let results: Vec<_> = hasher.collect();

        assert_eq!(results.len(), 3);

        // Window 1: [1,2] -> 1*10 + 2 = 12
        assert_eq!(results[0], (12u32, 0));
        // Window 2: [2,3] -> 2*10 + 3 = 23
        assert_eq!(results[1], (23u32, 1));
        // Window 3: [3,4] -> 3*10 + 4 = 34
        assert_eq!(results[2], (34u32, 2));
    }

    #[test]
    fn test_different_data_types() {
        let data = vec![1u16, 2, 3];
        let hasher = NonRollingHash::new(&data, 2, 10u64, 1000000007u64);
        let results: Vec<_> = hasher.collect();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0], (12u64, 0));
        assert_eq!(results[1], (23u64, 1));
    }

    #[test]
    fn test_large_modulus_operations() {
        let data = vec![255u8, 255, 255];
        let base = 256u64;
        let modulus = (1u64 << 32) - 1; // Large modulus
        let hasher = NonRollingHash::new(&data, 3, base, modulus);
        let results: Vec<_> = hasher.collect();

        assert_eq!(results.len(), 1);
        // Should handle large numbers correctly
        let expected = (255u64 * 256 * 256 + 255 * 256 + 255) % modulus;
        assert_eq!(results[0].0, expected);
    }

    #[test]
    fn test_zero_values() {
        let data = vec![0u8, 1, 0];
        let base = 10u32;
        let modulus = 97u32;
        let hasher = NonRollingHash::new(&data, 2, base, modulus);
        let results: Vec<_> = hasher.collect();

        assert_eq!(results.len(), 2);
        // [0,1] -> 0*10 + 1 = 1
        assert_eq!(results[0], (1u32, 0));
        // [1,0] -> 1*10 + 0 = 10
        assert_eq!(results[1], (10u32, 1));
    }

    #[test]
    fn test_hash_uniqueness() {
        let data = vec![1u8, 2, 3, 4, 5];
        let base = 257u32;
        let modulus = 1000000007u32;
        let hasher = NonRollingHash::new(&data, 3, base, modulus);
        let results: Vec<_> = hasher.collect();

        // Check that all hashes are unique for this input
        let mut hashes = std::collections::HashSet::new();
        for (hash, _) in results {
            assert!(hashes.insert(hash), "Hash collision detected!");
        }
    }

    #[test]
    fn test_deterministic_behavior() {
        let data = vec![42u8, 17, 99, 3];
        let base = 31u32;
        let modulus = 1009u32;

        // Run the same hash twice
        let hasher1 = NonRollingHash::new(&data, 2, base, modulus);
        let results1: Vec<_> = hasher1.collect();

        let hasher2 = NonRollingHash::new(&data, 2, base, modulus);
        let results2: Vec<_> = hasher2.collect();

        assert_eq!(results1, results2);
    }

    #[test]
    fn test_edge_case_single_element() {
        let data = vec![42u8];
        let hasher = NonRollingHash::new(&data, 1, 257u32, 1000000007u32);
        let results: Vec<_> = hasher.collect();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0], (42u32, 0));
    }

    #[test]
    fn test_negative() {
        let data = vec![-1, -2, -3];
        let base = 1000000007i32;
        let hasher = NonRollingHash::new(&data, 1, 257i32, base);
        let results: Vec<_> = hasher.collect();
        let expected: Vec<_> = vec![base - 1, base - 2, base - 3]
            .into_iter()
            .enumerate()
            .map(|(start, hash)| (hash, start))
            .collect();
        assert_eq!(results, expected);
    }
    #[test]
    fn test_negative_with_window() {
        let data = vec![-1, -2, -3, -4, -5, -6];
        let base = 1000000007i32;
        let hasher = NonRollingHash::new(&data, 2, 257i32, base);
        let results: Vec<_> = hasher.collect();
        let expected: Vec<_> = vec![
            base - 259,
            base - 517,
            base - 775,
            base - 1_033,
            base - 1_291,
        ]
        .into_iter()
        .enumerate()
        .map(|(start, hash)| (hash, start))
        .collect();
        assert_eq!(results, expected);
    }

    #[test]
    fn test_with_overflow() {
        let data = vec![1u16, 2u16, 3u16];
        let window_size = 3;
        let base = 3u16;
        let modulus = 13u16;
        let result: Vec<_> = NonRollingHash::new(&data, window_size, base, modulus).collect();
        let expected = vec![(5, 0)];
        assert_eq!(result, expected)
    }
}
