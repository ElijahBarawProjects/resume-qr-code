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
    WindowHasherSignedDataTests
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

#[cfg(test)]
mod tests {
    use crate::{
        interface::{WindowHasher},
        interface_non_rolling_hash::InterfaceHasher,
    };
    use num_bigint::BigInt;
    use std::collections::HashSet;

    // Generic test functions that work with any WindowHasher implementation
    fn test_new_function_generic<THash, TData, H>()
    where
        THash: Copy
            + Clone
            + PartialEq
            + std::fmt::Debug
            + TryFrom<u8>
            + TryFrom<BigInt>
            + Into<BigInt>,
        TData: Copy + Into<BigInt>,
        H: WindowHasher<THash, TData>,
    {
        let base: THash = 10.try_into().ok().unwrap();
        let modulus: THash = 97.try_into().ok().unwrap();
        let hasher = H::new(base, modulus);
        assert!(hasher.is_ok());
    }

    fn test_hash_function_generic<THash, TData, H>(base: THash, modulus: THash, data: &[TData])
    where
        THash: Copy + Clone + PartialEq + std::fmt::Debug + TryFrom<BigInt> + Into<BigInt>,
        TData: Copy + Into<BigInt>,
        H: WindowHasher<THash, TData>,
    {
        let hasher = H::new(base, modulus).unwrap();
        let hash1 = hasher.hash(data);
        let hash2 = hasher.hash(data);
        assert_eq!(hash1, hash2, "Hash function should be deterministic");
    }

    fn test_sliding_hash_consistency<THash, TData, H>(
        base: THash,
        modulus: THash,
        data: &[TData],
        window_size: usize,
    ) where
        THash: Copy + Clone + PartialEq + std::fmt::Debug + TryFrom<BigInt> + Into<BigInt>,
        TData: Copy + Into<BigInt>,
        H: WindowHasher<THash, TData> + 'static,
    {
        let hasher = H::new(base, modulus).unwrap();

        let results1: Vec<_> = hasher
            .clone()
            .sliding_hash_owned(data, window_size)
            .collect();
        let results2: Vec<_> = hasher.slow_sliding_hash_owned(data, window_size).collect();

        assert_eq!(
            results1, results2,
            "sliding_hash_owned and slow_sliding_hash_owned should produce identical results"
        );
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
        let result: Vec<_> =
            InterfaceHasher::single_sliding_hash(base, modulus, &data, window_size)
                .unwrap()
                .collect();
        let expected = vec![(5, 0)];
        assert_eq!(result, expected)
    }

    // COMPREHENSIVE EDGE CASE TESTS

    // Tests for generic functions with different types
    #[test]
    fn test_generic_new_u8() {
        test_new_function_generic::<u8, u8, InterfaceHasher>();
    }

    #[test]
    fn test_generic_new_u16() {
        test_new_function_generic::<u16, u16, InterfaceHasher>();
    }

    #[test]
    fn test_generic_new_u32() {
        test_new_function_generic::<u32, u32, InterfaceHasher>();
    }

    #[test]
    fn test_generic_new_u64() {
        test_new_function_generic::<u64, u64, InterfaceHasher>();
    }

    #[test]
    fn test_generic_new_i8() {
        test_new_function_generic::<i8, i8, InterfaceHasher>();
    }

    #[test]
    fn test_generic_new_i16() {
        test_new_function_generic::<i16, i16, InterfaceHasher>();
    }

    #[test]
    fn test_generic_new_i32() {
        test_new_function_generic::<i32, i32, InterfaceHasher>();
    }

    #[test]
    fn test_generic_new_i64() {
        test_new_function_generic::<i64, i64, InterfaceHasher>();
    }

    // Modulus edge cases
    #[test]
    fn test_small_modulus() {
        let data = vec![1u32, 2, 3, 4, 5];
        let base = 10u32;
        let modulus = 2u32;

        let hasher = <InterfaceHasher as WindowHasher<u32, u32>>::new(base, modulus).unwrap();
        let hash: u32 = hasher.hash(&data);
        assert!(hash < modulus);

        let results: Vec<_> = InterfaceHasher::single_sliding_hash(base, modulus, &data, 2)
            .unwrap()
            .collect();
        for (hash_val, _) in results {
            assert!(hash_val < modulus);
        }
    }

    // Function-specific comprehensive tests
    #[test]
    fn test_new_function_comprehensive() {
        let base = 257u32;
        let modulus = 1000000007u32;

        let hasher1 = <InterfaceHasher as WindowHasher<u32, u32>>::new(base, modulus).unwrap();
        let hasher2 = <InterfaceHasher as WindowHasher<u32, u32>>::new(base, modulus).unwrap();

        let data = vec![1u32, 2, 3];
        let hash1: u32 = hasher1.hash(&data);
        let hash2: u32 = hasher2.hash(&data);
        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_hash_function_comprehensive() {
        let base = 257u32;
        let modulus = 1000000007u32;
        let hasher = <InterfaceHasher as WindowHasher<u32, u32>>::new(base, modulus).unwrap();

        let data1 = vec![1u32, 2, 3];
        let data2 = vec![1u32, 2, 3];
        let data3 = vec![3u32, 2, 1];

        let hash1a: u32 = hasher.hash(&data1);
        let hash2a: u32 = hasher.hash(&data2);
        let hash3a: u32 = hasher.hash(&data3);
        assert_eq!(hash1a, hash2a);
        assert_ne!(hash1a, hash3a);

        let empty_data: Vec<u32> = vec![];
        let empty_hash: u32 = hasher.hash(&empty_data);
        assert_eq!(empty_hash, 0u32);
    }

    // Consistency tests between methods
    #[test]
    fn test_sliding_hash_consistency_u32() {
        let data = vec![1u32, 2, 3, 4, 5];
        test_sliding_hash_consistency::<u32, u32, InterfaceHasher>(257u32, 1000000007u32, &data, 3);
    }

    #[test]
    fn test_sliding_hash_consistency_i32() {
        let data = vec![-1i32, -2, 3, -4, 5];
        test_sliding_hash_consistency::<i32, i32, InterfaceHasher>(257i32, 1000000007i32, &data, 2);
    }

    #[test]
    fn test_sliding_hash_consistency_u64() {
        let data = vec![1u64, 2, 3, 4];
        test_sliding_hash_consistency::<u64, u64, InterfaceHasher>(257u64, 1000000007u64, &data, 2);
    }

    // Edge case combinations
    #[test]
    fn test_zero_data_large_window() {
        let data = vec![0u32; 5];
        let base = 257u32;
        let modulus = 1000000007u32;

        let results: Vec<_> = InterfaceHasher::single_sliding_hash(base, modulus, &data, 10)
            .unwrap()
            .collect();
        assert_eq!(results.len(), 0);
    }

    // Generic hash function tests
    #[test]
    fn test_generic_hash_u8() {
        let data = vec![1u8, 2, 3];
        test_hash_function_generic::<u8, u8, InterfaceHasher>(10u8, 97u8, &data);
    }

    #[test]
    fn test_generic_hash_u16() {
        let data = vec![1u16, 2, 3];
        test_hash_function_generic::<u16, u16, InterfaceHasher>(257u16, 65521u16, &data);
    }

    #[test]
    fn test_generic_hash_i32() {
        let data = vec![-1i32, 0, 1];
        test_hash_function_generic::<i32, i32, InterfaceHasher>(257i32, 1000000007i32, &data);
    }

    // Type boundary tests
    #[test]
    fn test_type_boundaries_u8() {
        let data = vec![0u8, 127u8, 255u8];
        let base = 2u16;
        let modulus = 256u16;

        let hasher = <InterfaceHasher as WindowHasher<u16, u8>>::new(base, modulus).unwrap();
        let hash: u16 = hasher.hash(&data);
        assert!(hash < modulus);
    }

    #[test]
    fn test_type_boundaries_i8() {
        let data = vec![i8::MIN, 0i8, i8::MAX];
        let base = 2i16;
        let modulus = 1000i16;

        let hasher = <InterfaceHasher as WindowHasher<i16, i8>>::new(base, modulus).unwrap();
        let hash: i16 = hasher.hash(&data);
        assert!(hash >= 0);
        assert!(hash < modulus);
    }

    #[test]
    fn test_deterministic_across_runs() {
        let data = vec![42u32, 17, 99, 3, 8];
        let base = 31u32;
        let modulus = 1009u32;

        let mut all_results = Vec::new();
        for _ in 0..5 {
            let results: Vec<_> = InterfaceHasher::single_sliding_hash(base, modulus, &data, 3)
                .unwrap()
                .collect();
            all_results.push(results);
        }

        for i in 1..all_results.len() {
            assert_eq!(all_results[0], all_results[i]);
        }
    }

    #[test]
    fn test_hash_distribution() {
        let data: Vec<u8> = (0..100).map(|i| i as u8).collect();
        let base = 257u32;
        let modulus = 1009u32;

        let results: Vec<_> = InterfaceHasher::single_sliding_hash(base, modulus, &data, 5)
            .unwrap()
            .collect();
        let hash_set: HashSet<u32> = results.iter().map(|(hash, _)| *hash).collect();

        // Should have good distribution - most hashes should be unique
        assert!(hash_set.len() as f64 / results.len() as f64 > 0.8);
    }
}