use std::marker::PhantomData;

use num_bigint::BigInt;
use num_iter::range;

use crate::interface::WindowHasher;

#[derive(Clone)]
struct InterfaceHasher {
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

impl<THash, TData> WindowHasher<THash, TData> for InterfaceHasher
where
    TData: Copy + Into<BigInt>,
    THash: TryFrom<BigInt> + Into<BigInt> + Copy,
{
    fn new(base: THash, modulus: THash) -> Result<Self, crate::interface::HasherErr> {
        let base = base.try_into().unwrap();
        let modulus = modulus.try_into().unwrap();
        Ok(InterfaceHasher { base, modulus })
    }

    fn hash(&self, data: &[TData]) -> THash {
        // we don't expect panics here as the built hash, while a BigInt, will 
        // fit in THash since it's smaller in magnitude than modulus
        self.build_hash(data).try_into().ok().unwrap()
    }
}


#[cfg(test)]
mod tests {
    use crate::{interface::WindowHasher, interface_non_rolling_hash::InterfaceHasher};


    #[test]
    fn test_empty_data() {
        let data: Vec<u8> = vec![];
        let hasher = InterfaceHasher::single_sliding_hash(257u32, 1000000007u32, &data, 1).unwrap();
        let results: Vec<_> = hasher.collect();
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_window_size_larger_than_data() {
        let data = vec![1u8, 2, 3];
        let hasher = InterfaceHasher::single_sliding_hash(257u32, 1000000007u32, &data, 5).unwrap();
        let results: Vec<_> = hasher.collect();
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_window_size_equal_to_data() {
        let data = vec![1u8, 2, 3];
        let hasher = InterfaceHasher::single_sliding_hash(257u32, 1000000007u32, &data, 3).unwrap();
        let results: Vec<_> = hasher.collect();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], (66_566, 0)); // start position should be 0
    }

    #[test]
    fn test_single_character_window() {
        let data = vec![65u8, 66, 67]; // "ABC"
        let hasher = InterfaceHasher::single_sliding_hash(257u32, 1000000007u32, &data, 1).unwrap();
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
        let hasher = InterfaceHasher::single_sliding_hash(base, modulus, &data, 2).unwrap();
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
        let hasher = InterfaceHasher::single_sliding_hash(base, modulus, &data, 2).unwrap();
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
        let hasher = InterfaceHasher::single_sliding_hash(base, modulus, &data, 2).unwrap();
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
        let hasher = InterfaceHasher::single_sliding_hash(10u64, 1000000007u64, &data, 2).unwrap();
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
        let hasher = InterfaceHasher::single_sliding_hash(base, modulus, &data, 3).unwrap();
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
        let hasher = InterfaceHasher::single_sliding_hash(base, modulus, &data, 2).unwrap();
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
        let hasher = InterfaceHasher::single_sliding_hash(base, modulus, &data, 3).unwrap();
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
        let hasher1 = InterfaceHasher::single_sliding_hash(base, modulus, &data, 2).unwrap();
        let results1: Vec<_> = hasher1.collect();

        let hasher2 = InterfaceHasher::single_sliding_hash(base, modulus, &data, 2).unwrap();
        let results2: Vec<_> = hasher2.collect();

        assert_eq!(results1, results2);
    }

    #[test]
    fn test_edge_case_single_element() {
        let data = vec![42u8];
        let hasher = InterfaceHasher::single_sliding_hash(257u32, 1000000007u32, &data, 1).unwrap();
        let results: Vec<_> = hasher.collect();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0], (42u32, 0));
    }

    #[test]
    fn test_negative() {
        let data = vec![-1, -2, -3];
        let base = 1000000007i32;
        let hasher = InterfaceHasher::single_sliding_hash(257i32, base, &data, 1).unwrap();
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
        let hasher = InterfaceHasher::single_sliding_hash(257i32, base, &data, 2).unwrap();
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
        let result: Vec<_> = InterfaceHasher::single_sliding_hash(base, modulus, &data, window_size).unwrap().collect();
        let expected = vec![(5, 0)];
        assert_eq!(result, expected)
    }
}
