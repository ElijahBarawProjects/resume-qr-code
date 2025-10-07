// #[allow(dead_code)]
#[derive(Debug)]
pub enum HasherErr {
    InvalidBase(&'static str),
    InvalidModulus(&'static str),
}

pub trait WindowHasher<THash, TData>: Sized + Clone {
    /**
     * Need to Implement
     */

    fn new(base: THash, modulus: THash) -> Result<Self, HasherErr>;

    fn hash<'data>(&self, data: &'data [TData]) -> THash;

    /**
     * Override
     */

    fn sliding_hash_owned<'data>(
        self,
        data: &'data [TData],
        window_size: usize,
    ) -> impl Iterator<Item = (THash, usize)> + 'data
    where
        Self: 'data,
    {
        self.slow_sliding_hash_owned(data, window_size)
    }

    fn sliding_hash<'data>(
        &self,
        data: &'data [TData],
        window_size: usize,
    ) -> impl Iterator<Item = (THash, usize)> + 'data
    where
        Self: 'data,
    {
        let clone = self.clone();
        clone.sliding_hash_owned(data, window_size)
    }

    /**
     * Do not Implement
     */

    /// Default slow sliding implementation that recomputes hashes for each window
    /// Shouldn't be overridden as it can serve as a reference
    fn slow_sliding_hash_owned<'data>(
        self,
        data: &'data [TData],
        window_size: usize,
    ) -> impl Iterator<Item = (THash, usize)> + 'data
    where
        Self: 'data,
    {
        let num_iters = (data.len() + 1).saturating_sub(window_size);
        (0..num_iters).map(move |start| {
            let window = &data[start..start + window_size];
            (self.hash(window), start)
        })
    }

    fn slow_sliding_hash<'data>(
        &self,
        data: &'data [TData],
        window_size: usize,
    ) -> impl Iterator<Item = (THash, usize)> + 'data
    where
        Self: 'data,
    {
        let clone = self.clone();
        clone.slow_sliding_hash_owned(data, window_size)
    }

    fn single_sliding_hash<'data>(
        base: THash,
        modulus: THash,
        data: &'data [TData],
        window_size: usize,
    ) -> Result<impl Iterator<Item = (THash, usize)> + 'data, HasherErr>
    where
        Self: 'data,
    {
        Self::new(base, modulus).and_then(|hasher| Ok(hasher.sliding_hash_owned(data, window_size)))
    }
}

#[cfg(test)]
pub mod tests {

    use super::WindowHasher;
    use num_bigint::BigInt;
    use paste::paste;

    // Macro to generate test functions for different type combinations
    macro_rules! generate_test_new_function {
        ($th:ty, $td:ty) => {
            paste::paste! {
               fn [<test_generic_new_td_ $td _th_ $th>]()
               where
                     Self: WindowHasher<$th, $td>,
                 {
                     <Self as WindowHasherTests<$th, $td>>::test_new_function_generic();
                }
            }
        };
    }
    /**
     * Unsigned Tests
     */

    /**

    ### Type Constraints

    THash: HashType + Debug,
    TData: TryFrom<u8>,

    ### Data Length Variations

    - Empty data (length 0)
    - Single element data (length 1)
    - Small data (length 2-10)
    - Large data (length > 10,000)
    - Very large data (length > 100,000)

    */
    #[tested_trait]
    pub trait WindowHasherDataSizeTests<THash, TData>
    where
        Self: WindowHasher<THash, TData>,
    {
        #[test]
        fn _test_data_size_empty()
        where
            THash: HashType,
        {
            let data: Vec<_> = vec![];
            let hasher = Self::single_sliding_hash(THash::one(), THash::one(), &data, 1).unwrap();
            let result: Vec<_> = hasher.collect();
            assert_eq!(result.len(), 0);
        }

        #[test]
        fn _test_data_size_one()
        where
            THash: HashType + Debug,
            TData: TryFrom<u8>,
        {
            let data = vec![expect_from!(42u8)];
            let hasher = Self::single_sliding_hash(
                unchecked_make_number(257u32),
                unchecked_make_number(1000000007u32),
                &data,
                1,
            )
            .unwrap();
            let results: Vec<_> = hasher.collect();

            assert_eq!(results.len(), 1);
            assert_eq!(
                results[0],
                (expect_from!(unchecked_make_number::<THash, _>(42u32)), 0)
            );
        }

        #[test]
        fn _test_data_size_empty_comprehensive()
        where
            THash: HashType + Debug,
            TData: TryFrom<u8>,
        {
            let data: Vec<TData> = vec![];
            let base = unchecked_make_number(257u32);
            let modulus = unchecked_make_number(1000000007u32);

            let hasher = <Self as WindowHasher<THash, TData>>::new(base, modulus).unwrap();
            let hash = hasher.hash(&data);
            assert_eq!(hash, unchecked_make_number(0));

            let window_size = 1;
            let results: Vec<_> = Self::single_sliding_hash(base, modulus, &data, window_size)
                .unwrap()
                .collect();
            assert_eq!(results.len(), 0);

            let window_size = 5;
            let results: Vec<_> = Self::single_sliding_hash(base, modulus, &data, window_size)
                .unwrap()
                .collect();
            assert_eq!(results.len(), 0);
        }

        fn _test_data_size_of(size: usize)
        where
            TData: TryFrom<u8>,
            THash: HashType + Debug,
        {
            let data: Vec<TData> = (0..size).map(|i| expect_from!((i % 256) as u8)).collect();
            let base = unchecked_make_number(257u32);
            let modulus = unchecked_make_number(1000000007u32);

            let hasher = Self::new(base, modulus).unwrap();
            let hash1: THash = hasher.hash(&data);
            let hash2: THash = hasher.hash(&data);
            assert_eq!(hash1, hash2);

            let window_size = 100;
            let results: Vec<_> = Self::single_sliding_hash(base, modulus, &data, window_size)
                .unwrap()
                .collect();
            assert_eq!(results.len(), size - 100 + 1);
            for (hash, start) in results {
                assert_eq!(hash, hasher.hash(&data[start..start + window_size]))
            }
        }

        #[test]
        fn _test_data_size_large()
        where
            TData: TryFrom<u8>,
            THash: HashType + Debug,
        {
            Self::_test_data_size_of(10_000);
        }

        #[test]
        fn _test_data_size_very_large()
        where
            TData: TryFrom<u8>,
            THash: HashType + Debug,
        {
            Self::_test_data_size_of(100_000);
        }
    }

    /*

    ### Type Constraints

    THash: HashType + Debug,
    TData: TryFrom<u8>,

    ### Data Content Variations

    - hashes smaller than modulus
    - data with zeros
    - specific known hashes

    */
    #[tested_trait]
    pub trait WindowHasherDataTests<THash, TData>
    where
        Self: WindowHasher<THash, TData>,
    {
        #[test]
        fn _test_data_with_zeros()
        where
            THash: HashType + Debug,
            TData: TryFrom<u8>,
        {
            let data = vec![expect_from!(0), expect_from!(1), expect_from!(0)];
            let base = unchecked_make_number(10u32);
            let modulus = unchecked_make_number(97u32);
            let hasher = Self::single_sliding_hash(base, modulus, &data, 2).unwrap();
            let results: Vec<_> = hasher.collect();

            assert_eq!(results.len(), 2);
            // [0,1] -> 0*10 + 1 = 1
            assert_eq!(results[0], (unchecked_make_number(1u32), 0));
            // [1,0] -> 1*10 + 0 = 10
            assert_eq!(results[1], (unchecked_make_number(10u32), 1));
        }

        #[test]
        fn _test_data_hash_determinism()
        where
            THash: HashType + Debug,
            TData: TryFrom<u8>,
        {
            let data: Vec<_> = vec![42u8, 17, 99, 3, 8]
                .into_iter()
                .map(|x| expect_from!(x))
                .collect();
            let base = unchecked_make_number(31u32);
            let modulus = unchecked_make_number(1009u32);

            let mut all_results = Vec::new();
            let num_runs = 5;
            for _ in 0..num_runs {
                let results: Vec<_> = Self::single_sliding_hash(base, modulus, &data, 3)
                    .unwrap()
                    .collect();
                all_results.push(results);
            }

            for i in 1..all_results.len() {
                assert_eq!(all_results[0], all_results[i]);
            }
        }
    }

    /**

    ### Type Constraints

    THash: HashType + Debug,
    TData: TryFrom<u8>,

    ### Window Size Variations

    - Zero window size
    - Window size of 1
    - Window size of 2
    - Small window sizes (3-9)
    - Large window sizes (500+)
    - Window size equal to data length
    - Window size larger than data length

    */
    #[tested_trait]
    pub trait WindowHasherWindowSizeTests<THash, TData>
    where
        Self: WindowHasher<THash, TData>,
    {
        #[test]
        fn _test_window_size_equal_to_data()
        where
            THash: HashType + Debug,
            TData: TryFrom<u8>,
        {
            let data: Vec<TData> = vec![1u8, 2u8, 3u8]
                .iter()
                .map(|&x| expect_from!(x))
                .collect();
            let hasher = Self::single_sliding_hash(
                unchecked_make_number(257),
                unchecked_make_number(1000000007u32),
                &data,
                3,
            )
            .unwrap();
            let results: Vec<_> = hasher.collect();
            assert_eq!(results.len(), 1);
            assert_eq!(results[0], (unchecked_make_number(66_566), 0)); // start position should be 0
        }

        #[test]
        fn _test_window_size_larger_than_data()
        where
            THash: HashType + Debug,
            TData: TryFrom<u8>,
        {
            let data = vec![expect_from!(1u8), expect_from!(2u8), expect_from!(3u8)];
            let hasher = Self::single_sliding_hash(
                unchecked_make_number(257u32),
                unchecked_make_number(1000000007u32),
                &data,
                5,
            )
            .unwrap();
            let results: Vec<_> = hasher.collect();
            assert_eq!(results.len(), 0);
        }

        #[test]
        fn _test_window_size_zero()
        where
            THash: HashType + Debug,
            TData: TryFrom<u8>,
        {
            let start = 1;
            let end = 200;
            let data: Vec<_> = (start..=end).into_iter().map(|x| expect_from!(x)).collect();

            let base = unchecked_make_number(257u32);
            let modulus = unchecked_make_number(1000000007u32);

            let window_size = 0;
            let results: Vec<_> = Self::single_sliding_hash(base, modulus, &data, window_size)
                .unwrap()
                .collect();
            assert_eq!(results.len(), data.len() + 1);

            for (hash_val, _) in results {
                assert_eq!(hash_val, THash::zero());
            }
        }

        #[test]
        fn _test_window_size_one()
        where
            THash: HashType + Debug,
            TData: TryFrom<u8>,
        {
            let data: Vec<_> = vec![65u8, 66u8, 67u8]
                .iter()
                .map(|&x| expect_from!(x))
                .collect(); // "ABC"
            let hasher = Self::single_sliding_hash(
                unchecked_make_number(257u32),
                // as written, this test will only work for i32 or larger
                unchecked_make_number(1000000007u32),
                &data,
                1,
            )
            .unwrap();
            let results: Vec<_> = hasher.collect();

            assert_eq!(results.len(), 3);
            assert_eq!(results[0], (unchecked_make_number(65u32), 0));
            assert_eq!(results[1], (unchecked_make_number(66u32), 1));
            assert_eq!(results[2], (unchecked_make_number(67u32), 2));
        }

        #[test]
        fn _test_window_size_two()
        where
            THash: HashType + Debug,
            TData: TryFrom<u8>,
        {
            let data: Vec<_> = vec![5u8, 10, 15, 20]
                .into_iter()
                .map(|x| expect_from!(x))
                .collect();
            let base = unchecked_make_number(100u32);
            let modulus = unchecked_make_number(1000000007u32);
            let window_size = 2;

            let results: Vec<_> = Self::single_sliding_hash(base, modulus, &data, window_size)
                .unwrap()
                .collect();
            assert_eq!(results.len(), 3);

            let expected: Vec<_> = vec![510u32, 1015u32, 1520u32]
                .into_iter()
                .map(|x| unchecked_make_number(x))
                .collect();

            for (i, expected_hash) in expected.into_iter().enumerate() {
                assert_eq!(results[i], (expected_hash, i));
            }
        }

        #[test]
        fn _test_window_size_overlapping()
        where
            THash: HashType + Debug,
            TData: TryFrom<u8>,
        {
            let data: Vec<_> = vec![1u8, 2, 3, 4]
                .iter()
                .map(|&x| expect_from!(x))
                .collect();
            let base = unchecked_make_number(10u32);
            let modulus = unchecked_make_number(1000000007u32);
            let hasher = Self::single_sliding_hash(base, modulus, &data, 2).unwrap();
            let results: Vec<_> = hasher.collect();

            assert_eq!(results.len(), 3);

            // Window 1: [1,2] -> 1*10 + 2 = 12
            assert_eq!(results[0], (unchecked_make_number(12u32), 0));
            // Window 2: [2,3] -> 2*10 + 3 = 23
            assert_eq!(results[1], (unchecked_make_number(23u32), 1));
            // Window 3: [3,4] -> 3*10 + 4 = 34
            assert_eq!(results[2], (unchecked_make_number(34u32), 2));
        }

        #[test]
        fn _test_window_size_multiple_data_size()
        where
            THash: HashType,
            TData: TryFrom<u8>,
        {
            let data: Vec<_> = vec![0u8; 5].into_iter().map(|x| expect_from!(x)).collect();
            let base = unchecked_make_number(257u32);
            let modulus = unchecked_make_number(1000000007u32);
            let window_size = 10;

            let results: Vec<_> = Self::single_sliding_hash(base, modulus, &data, window_size)
                .unwrap()
                .collect();
            assert_eq!(results.len(), 0);
        }

        #[test]
        fn _test_window_size_large()
        where
            THash: HashType,
            TData: TryFrom<u8>,
        {
            let data: Vec<TData> = (1..=1000)
                .into_iter()
                .map(|x| expect_from!((x % 256) as u8))
                .collect();
            let base = unchecked_make_number(257u32);
            let modulus = unchecked_make_number(1000000007u32);
            let window_size = 500;

            let results: Vec<_> = Self::single_sliding_hash(base, modulus, &data, window_size)
                .unwrap()
                .collect();
            assert_eq!(results.len(), 501);
        }
    }

    /*

    ### Type Constraints

    THash: HashType + Debug,
    TData: TryFrom<u8>,

    ### Window Modulus Variations

    Tests small modulus in {1, 2, 5}

    */
    #[tested_trait]
    pub trait WindowHasherModulusTests<THash, TData>
    where
        Self: WindowHasher<THash, TData>,
    {
        fn _test_modulus_of(original_mod: u8)
        where
            THash: HashType + Debug,
            TData: TryFrom<u8>,
        {
            let modulus = unchecked_make_number(original_mod);
            let start = 1;
            let end = 5;
            let range = start..=end;

            let data: Vec<_> = range.clone().into_iter().map(|x| expect_from!(x)).collect();
            let base = unchecked_make_number(10u32);

            let hasher = Self::new(base, modulus).unwrap();
            let hash = hasher.hash(&data);
            assert!(hash < modulus);
            let window_size = 2;

            // our base is even, `range` increments by one, and window size is 2, so we're looking at (10 * current  + next)
            // 10 * current is always even, so the hash is the modulus of the next value
            let expected: Vec<_> = range
                .clone()
                .map(|i| (i + 1) % original_mod)
                .into_iter()
                .map(|x| unchecked_make_number(x))
                .collect();
            let results: Vec<_> = Self::single_sliding_hash(base, modulus, &data, window_size)
                .unwrap()
                .collect();
            for (i, (hash_val, _)) in results.into_iter().enumerate() {
                assert!(hash_val < modulus);
                assert_eq!(hash_val, expected[i]);
            }
        }

        #[test]
        fn _test_modulus_of_two()
        where
            THash: HashType + Debug,
            TData: TryFrom<u8>,
        {
            <Self as WindowHasherModulusTests<THash, TData>>::_test_modulus_of(2);
        }

        #[test]
        fn _test_modulus_of_one()
        where
            THash: HashType + Debug,
            TData: TryFrom<u8>,
        {
            <Self as WindowHasherModulusTests<THash, TData>>::_test_modulus_of(1);
        }

        #[test]
        fn _test_modulus_of_five()
        where
            THash: HashType + Debug,
            TData: TryFrom<u8>,
        {
            <Self as WindowHasherModulusTests<THash, TData>>::_test_modulus_of(5);
        }
    }


    // THash, TData are only constrained by `where Self: ...`
    pub trait WindowHasherTests<THash, TData>
    where
        Self: WindowHasher<THash, TData>,
    {
        /*
        # Comprehensive Test Suite for interface_non_rolling_hash.rs

        This test suite provides extensive coverage of all edge cases and functionality for the WindowHasher
        implementation of the WindowHasher trait. Tests are designed to be generic where appropriate and cover
        all interface functions with various integral types.

        ## Edge Cases Covered

        ### Data Length Variations

        - Empty data (length 0)
        - Single element data (length 1)
        - Small data (length 2-10)
        - Large data (length > 10,000)
        - Very large data (length > 100,000)

        ### Modulus Values

        - Large negative modulus
        - Small negative modulus (for signed types)
        - Zero-valued modulus (0)
        - Small positive modulus (1, 2, 3)
        - Large modulus values (close to type limits)
        - Prime vs composite modulus values

        ### Data Values

        - Zero values in data
        - Negative values in data (for signed types)
        - Maximum positive values for type
        - Minimum negative values for type (for signed types)
        - Mixed positive/negative/zero combinations

        ### Window Size Variations

        - Zero window size
        - Window size of 1
        - Window size of 2
        - Small window sizes (3-9)
        - Large window sizes (500+)
        - Window size equal to data length
        - Window size larger than data length

        ## Interface Functions Tested

        TODO

        - `new()` - Constructor function with validation
        - `hash()` - Hash computation for data slice with deterministic behavior
        - `sliding_hash_owned()` - Owned sliding window hash iterator
        - `slow_sliding_hash_owned()` - Reference implementation of sliding hash
        - `single_sliding_hash()` - Static convenience method

        ## Generic Type Testing

        Tests are made generic when appropriate and invoked with various signed and unsigned integral types:
        - u8, u16, u32, u64, u128, usize
        - i8, i16, i32, i64, i128, isize

        Tests ensure proper type conversion and boundary handling between TData and THash types.

        ## Test Framework

        - Generic test functions work with any WindowHasher implementation
        - Specific tests invoke generic functions with WindowHasher
        - Consistency verification between sliding_hash_owned and slow_sliding_hash_owned
        - Deterministic behavior validation across multiple runs
        - Hash distribution quality assessment
        */

        // Generic test functions that work with any WindowHasher implementation
        fn test_new_function_generic()
        where
            THash: Copy
                + Clone
                + PartialEq
                + std::fmt::Debug
                + TryFrom<u8>
                + TryFrom<BigInt>
                + Into<BigInt>,
            TData: Copy + Into<BigInt>,
        {
            let base: THash = 10.try_into().ok().unwrap();
            let modulus: THash = 97.try_into().ok().unwrap();
            let hasher = Self::new(base, modulus);
            assert!(hasher.is_ok());
        }

        fn test_hash_function_generic(base: THash, modulus: THash, data: &[TData])
        where
            THash: Copy + Clone + PartialEq + std::fmt::Debug + TryFrom<BigInt> + Into<BigInt>,
            TData: Copy + Into<BigInt>,
        {
            let hasher = Self::new(base, modulus).unwrap();
            let hash1 = hasher.hash(data);
            let hash2 = hasher.hash(data);
            assert_eq!(hash1, hash2, "Hash function should be deterministic");
        }

        fn test_sliding_hash_consistency(
            base: THash,
            modulus: THash,
            data: &[TData],
            window_size: usize,
        ) where
            THash: Copy + Clone + PartialEq + std::fmt::Debug + TryFrom<BigInt> + Into<BigInt>,
            TData: Copy + Into<BigInt>,
        {
            let hasher = Self::new(base, modulus).unwrap();

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

        fn test_empty_data()
        where
            Self: WindowHasher<u32, u8>,
        {
            let data: Vec<u8> = vec![];
            let hasher = Self::single_sliding_hash(257u32, 1000000007u32, &data, 1).unwrap();
            let results: Vec<_> = hasher.collect();
            assert_eq!(results.len(), 0);
        }

        fn test_window_size_larger_than_data()
        where
            Self: WindowHasher<u32, u8>,
        {
            let data = vec![1u8, 2, 3];
            let hasher = Self::single_sliding_hash(257u32, 1000000007u32, &data, 5).unwrap();
            let results: Vec<_> = hasher.collect();
            assert_eq!(results.len(), 0);
        }

        fn test_window_size_equal_to_data()
        where
            Self: WindowHasher<u32, u8>,
        {
            let data = vec![1u8, 2, 3];
            let hasher = Self::single_sliding_hash(257u32, 1000000007u32, &data, 3).unwrap();
            let results: Vec<_> = hasher.collect();
            assert_eq!(results.len(), 1);
            assert_eq!(results[0], (66_566, 0)); // start position should be 0
        }

        fn test_single_character_window()
        where
            Self: WindowHasher<u32, u8>,
        {
            let data = vec![65u8, 66, 67]; // "ABC"
            let hasher = Self::single_sliding_hash(257u32, 1000000007u32, &data, 1).unwrap();
            let results: Vec<_> = hasher.collect();

            assert_eq!(results.len(), 3);
            assert_eq!(results[0], (65u32, 0));
            assert_eq!(results[1], (66u32, 1));
            assert_eq!(results[2], (67u32, 2));
        }

        fn test_hash_calculation_manual()
        where
            Self: WindowHasher<u32, u8>,
        {
            let data = vec![1u8, 2];
            let base = 10u32;
            let modulus = 97u32;
            let hasher = Self::single_sliding_hash(base, modulus, &data, 2).unwrap();
            let results: Vec<_> = hasher.collect();

            assert_eq!(results.len(), 1);
            // Manual calculation: (1 * 10 + 2) % 97 = 12
            assert_eq!(results[0], (12, 0));
        }

        fn test_hash_calculation_with_modulo()
        where
            Self: WindowHasher<u32, u8>,
        {
            let data = vec![100u8, 100, 100];
            let base = 10u32;
            let modulus = 7u32;
            let hasher = Self::single_sliding_hash(base, modulus, &data, 2).unwrap();
            let results: Vec<_> = hasher.collect();

            assert_eq!(results.len(), 2);
            assert_eq!(results[0], (1, 0));
            assert_eq!(results[1], (1, 1));
        }

        fn test_overlapping_windows()
        where
            Self: WindowHasher<u32, u8>,
        {
            let data = vec![1u8, 2, 3, 4];
            let base = 10u32;
            let modulus = 1000000007u32;
            let hasher = Self::single_sliding_hash(base, modulus, &data, 2).unwrap();
            let results: Vec<_> = hasher.collect();

            assert_eq!(results.len(), 3);

            // Window 1: [1,2] -> 1*10 + 2 = 12
            assert_eq!(results[0], (12u32, 0));
            // Window 2: [2,3] -> 2*10 + 3 = 23
            assert_eq!(results[1], (23u32, 1));
            // Window 3: [3,4] -> 3*10 + 4 = 34
            assert_eq!(results[2], (34u32, 2));
        }

        fn test_different_data_types()
        where
            Self: WindowHasher<u64, u16>,
        {
            let data = vec![1u16, 2, 3];
            let hasher = Self::single_sliding_hash(10u64, 1000000007u64, &data, 2).unwrap();
            let results: Vec<_> = hasher.collect();

            assert_eq!(results.len(), 2);
            assert_eq!(results[0], (12u64, 0));
            assert_eq!(results[1], (23u64, 1));
        }

        fn test_large_modulus_operations()
        where
            Self: WindowHasher<u64, u8>,
        {
            let data = vec![255u8, 255, 255];
            let base = 256u64;
            let modulus = (1u64 << 32) - 1; // Large modulus
            let hasher = Self::single_sliding_hash(base, modulus, &data, 3).unwrap();
            let results: Vec<_> = hasher.collect();

            assert_eq!(results.len(), 1);
            // Should handle large numbers correctly
            let expected = (255u64 * 256 * 256 + 255 * 256 + 255) % modulus;
            assert_eq!(results[0].0, expected);
        }

        fn test_zero_values()
        where
            Self: WindowHasher<u32, u8>,
        {
            let data = vec![0u8, 1, 0];
            let base = 10u32;
            let modulus = 97u32;
            let hasher = Self::single_sliding_hash(base, modulus, &data, 2).unwrap();
            let results: Vec<_> = hasher.collect();

            assert_eq!(results.len(), 2);
            // [0,1] -> 0*10 + 1 = 1
            assert_eq!(results[0], (1u32, 0));
            // [1,0] -> 1*10 + 0 = 10
            assert_eq!(results[1], (10u32, 1));
        }

        fn test_hash_uniqueness()
        where
            Self: WindowHasher<u32, u8>,
        {
            let data = vec![1u8, 2, 3, 4, 5];
            let base = 257u32;
            let modulus = 1000000007u32;
            let hasher = Self::single_sliding_hash(base, modulus, &data, 3).unwrap();
            let results: Vec<_> = hasher.collect();

            // Check that all hashes are unique for this input
            let mut hashes = std::collections::HashSet::new();
            for (hash, _) in results {
                assert!(hashes.insert(hash), "Hash collision detected!");
            }
        }

        fn test_deterministic_behavior()
        where
            Self: WindowHasher<u32, u8>,
        {
            let data = vec![42u8, 17, 99, 3];
            let base = 31u32;
            let modulus = 1009u32;

            // Run the same hash twice
            let hasher1 = Self::single_sliding_hash(base, modulus, &data, 2).unwrap();
            let results1: Vec<_> = hasher1.collect();

            let hasher2 = Self::single_sliding_hash(base, modulus, &data, 2).unwrap();
            let results2: Vec<_> = hasher2.collect();

            assert_eq!(results1, results2);
        }

        fn test_edge_case_single_element()
        where
            Self: WindowHasher<u32, u8>,
        {
            let data = vec![42u8];
            let hasher = Self::single_sliding_hash(257u32, 1000000007u32, &data, 1).unwrap();
            let results: Vec<_> = hasher.collect();

            assert_eq!(results.len(), 1);
            assert_eq!(results[0], (42u32, 0));
        }

        fn test_negative()
        where
            Self: WindowHasher<i32, i32>,
        {
            let data = vec![-1, -2, -3];
            let base = 1000000007i32;
            let hasher = Self::single_sliding_hash(257i32, base, &data, 1).unwrap();
            let results: Vec<_> = hasher.collect();
            let expected: Vec<_> = vec![base - 1, base - 2, base - 3]
                .into_iter()
                .enumerate()
                .map(|(start, hash)| (hash, start))
                .collect();
            assert_eq!(results, expected);
        }

        fn test_negative_with_window()
        where
            Self: WindowHasher<i32, i32>,
        {
            let data = vec![-1, -2, -3, -4, -5, -6];
            let base = 1000000007i32;
            let hasher = Self::single_sliding_hash(257i32, base, &data, 2).unwrap();
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

        fn test_with_overflow()
        where
            Self: WindowHasher<u16, u16>,
        {
            let data = vec![1u16, 2u16, 3u16];
            let window_size = 3;
            let base = 3u16;
            let modulus = 13u16;
            let result: Vec<_> = Self::single_sliding_hash(base, modulus, &data, window_size)
                .unwrap()
                .collect();
            let expected = vec![(5, 0)];
            assert_eq!(result, expected)
        }

        // Tests for generic functions with different types

        generate_test_new_function!(u8, u8);
        generate_test_new_function!(u16, u16);
        generate_test_new_function!(u32, u32);
        generate_test_new_function!(u64, u64);
        generate_test_new_function!(i8, i8);
        generate_test_new_function!(i16, i16);
        generate_test_new_function!(i32, i32);
        generate_test_new_function!(i64, i64);

        // Data length edge cases

        fn test_empty_data_comprehensive()
        where
            Self: WindowHasher<u32, u32>,
        {
            let data: Vec<u32> = vec![];
            let base = 257u32;
            let modulus = 1000000007u32;

            let hasher = <Self as WindowHasher<u32, u32>>::new(base, modulus).unwrap();
            let hash: u32 = hasher.hash(&data);
            assert_eq!(hash, 0u32);

            let results: Vec<_> = Self::single_sliding_hash(base, modulus, &data, 1)
                .unwrap()
                .collect();
            assert_eq!(results.len(), 0);

            let results: Vec<_> = Self::single_sliding_hash(base, modulus, &data, 5)
                .unwrap()
                .collect();
            assert_eq!(results.len(), 0);
        }

        fn test_large_data()
        where
            Self: WindowHasher<u32, u8>,
        {
            let data: Vec<u8> = (0..10000).map(|i| (i % 256) as u8).collect();
            let base = 257u32;
            let modulus = 1000000007u32;

            let hasher = <Self as WindowHasher<u32, u8>>::new(base, modulus).unwrap();
            let hash1: u32 = hasher.hash(&data);
            let hash2: u32 = hasher.hash(&data);
            assert_eq!(hash1, hash2);

            let results: Vec<_> = Self::single_sliding_hash(base, modulus, &data, 100)
                .unwrap()
                .collect();
            assert_eq!(results.len(), 10000 - 100 + 1);
        }

        fn test_very_large_data()
        where
            Self: WindowHasher<u32, u8>,
        {
            let data: Vec<u8> = (0..100000).map(|i| (i % 256) as u8).collect();
            let base = 257u32;
            let modulus = 1000000007u32;

            let hasher = <Self as WindowHasher<u32, u8>>::new(base, modulus).unwrap();
            let hash: u32 = hasher.hash(&data[0..1000]);
            assert!(hash < modulus);

            let results: Vec<_> = Self::single_sliding_hash(base, modulus, &data[0..1000], 10)
                .unwrap()
                .collect();
            assert_eq!(results.len(), 1000 - 10 + 1);
        }

        // Modulus edge cases

        fn test_small_modulus()
        where
            Self: WindowHasher<u32, u32>,
        {
            let data = vec![1u32, 2, 3, 4, 5];
            let base = 10u32;
            let modulus = 2u32;

            let hasher = <Self as WindowHasher<u32, u32>>::new(base, modulus).unwrap();
            let hash: u32 = hasher.hash(&data);
            assert!(hash < modulus);

            let results: Vec<_> = Self::single_sliding_hash(base, modulus, &data, 2)
                .unwrap()
                .collect();
            for (hash_val, _) in results {
                assert!(hash_val < modulus);
            }
        }

        fn test_modulus_one()
        where
            Self: WindowHasher<u32, u32>,
        {
            let data = vec![42u32, 17, 99];
            let base = 257u32;
            let modulus = 1u32;

            let hasher = <Self as WindowHasher<u32, u32>>::new(base, modulus).unwrap();
            let hash: u32 = hasher.hash(&data);
            assert_eq!(hash, 0u32);

            let results: Vec<_> = Self::single_sliding_hash(base, modulus, &data, 2)
                .unwrap()
                .collect();
            for (hash_val, _) in results {
                assert_eq!(hash_val, 0u32);
            }
        }

        fn test_large_modulus()
        where
            Self: WindowHasher<u64, u64>,
        {
            let data = vec![1u64, 2, 3];
            let base = 257u64;
            let modulus = u64::MAX - 1;

            let hasher = <Self as WindowHasher<u64, u64>>::new(base, modulus).unwrap();
            let hash: u64 = hasher.hash(&data);
            assert!(hash < modulus);
        }

        // fn test_negative_modulus() {
        //     let data = vec![-1i32, -2, -3];
        //     let base = 257i32;
        //     let modulus = -1000000007i32;

        //     let hasher = <Self as WindowHasher<i32, i32>>::new(base, modulus).unwrap();
        //     let _hash: i32 = hasher.hash(&data);
        //     let results: Vec<_> = Self::single_sliding_hash(base, modulus, &data, 2)
        //         .unwrap()
        //         .collect();
        //     assert_eq!(results.len(), 2);
        // }

        // // Data value edge cases

        // fn test_all_zero_data() {
        //     let data = vec![0u32; 100];
        //     let base = 257u32;
        //     let modulus = 1000000007u32;

        //     let hasher = <Self as WindowHasher<u32, u32>>::new(base, modulus).unwrap();
        //     let hash: u32 = hasher.hash(&data);
        //     assert_eq!(hash, 0u32);

        //     let results: Vec<_> = Self::single_sliding_hash(base, modulus, &data, 10)
        //         .unwrap()
        //         .collect();
        //     for (hash_val, _) in results {
        //         assert_eq!(hash_val, 0u32);
        //     }
        // }

        // fn test_max_value_data_unsigned() {
        //     let data = vec![u16::MAX, u16::MAX, u16::MAX];
        //     let base = 257u32;
        //     let modulus = 1000000007u32;

        //     let hasher = <Self as WindowHasher<u32, u16>>::new(base, modulus).unwrap();
        //     let hash: u32 = hasher.hash(&data);
        //     assert!(hash < modulus);
        // }

        // fn test_min_value_data_signed() {
        //     let data = vec![i16::MIN, i16::MIN, i16::MIN];
        //     let base = 257i32;
        //     let modulus = 1000000007i32;

        //     let hasher = <Self as WindowHasher<i32, i16>>::new(base, modulus).unwrap();
        //     let _hash: i32 = hasher.hash(&data);
        //     let results: Vec<_> = Self::single_sliding_hash(base, modulus, &data, 2)
        //         .unwrap()
        //         .collect();
        //     assert_eq!(results.len(), 2);
        // }

        // fn test_mixed_positive_negative_zero() {
        //     let data = vec![-100i32, 0, 100, -50, 0, 75];
        //     let base = 257i32;
        //     let modulus = 1000000007i32;

        //     let hasher = <Self as WindowHasher<i32, i32>>::new(base, modulus).unwrap();
        //     let hash1: i32 = hasher.hash(&data);
        //     let hash2: i32 = hasher.hash(&data);
        //     assert_eq!(hash1, hash2);

        //     let results: Vec<_> = Self::single_sliding_hash(base, modulus, &data, 3)
        //         .unwrap()
        //         .collect();
        //     assert_eq!(results.len(), 4);
        // }

        // // Window size edge cases

        // fn test_zero_window_size() {
        //     let data = vec![1u32, 2, 3];
        //     let base = 257u32;
        //     let modulus = 1000000007u32;

        //     let results: Vec<_> = Self::single_sliding_hash(base, modulus, &data, 0)
        //         .unwrap()
        //         .collect();
        //     assert_eq!(results.len(), data.len() + 1);

        //     for (hash_val, _) in results {
        //         assert_eq!(hash_val, 0u32);
        //     }
        // }

        // fn test_window_size_two() {
        //     let data = vec![5u32, 10, 15, 20];
        //     let base = 100u32;
        //     let modulus = 1000000007u32;

        //     let results: Vec<_> = Self::single_sliding_hash(base, modulus, &data, 2)
        //         .unwrap()
        //         .collect();
        //     assert_eq!(results.len(), 3);

        //     assert_eq!(results[0], (510u32, 0));
        //     assert_eq!(results[1], (1015u32, 1));
        //     assert_eq!(results[2], (1520u32, 2));
        // }

        // fn test_small_window_sizes() {
        //     let data = vec![1u32, 2, 3, 4, 5, 6, 7, 8, 9];
        //     let base = 10u32;
        //     let modulus = 97u32;

        //     for window_size in 3..=9 {
        //         let results: Vec<_> = Self::single_sliding_hash(base, modulus, &data, window_size)
        //             .unwrap()
        //             .collect();
        //         assert_eq!(results.len(), data.len() - window_size + 1);
        //     }
        // }

        // fn test_large_window_size() {
        //     let data: Vec<u32> = (1..=1000).collect();
        //     let base = 257u32;
        //     let modulus = 1000000007u32;

        //     let results: Vec<_> = Self::single_sliding_hash(base, modulus, &data, 500)
        //         .unwrap()
        //         .collect();
        //     assert_eq!(results.len(), 501);

        //     let hash_set: HashSet<u32> = results.iter().map(|(hash, _)| *hash).collect();
        //     assert!(hash_set.len() > 400);
        // }

        // // Function-specific comprehensive tests

        // fn test_new_function_comprehensive() {
        //     let base = 257u32;
        //     let modulus = 1000000007u32;

        //     let hasher1 = <Self as WindowHasher<u32, u32>>::new(base, modulus).unwrap();
        //     let hasher2 = <Self as WindowHasher<u32, u32>>::new(base, modulus).unwrap();

        //     let data = vec![1u32, 2, 3];
        //     let hash1: u32 = hasher1.hash(&data);
        //     let hash2: u32 = hasher2.hash(&data);
        //     assert_eq!(hash1, hash2);
        // }

        // fn test_hash_function_comprehensive() {
        //     let base = 257u32;
        //     let modulus = 1000000007u32;
        //     let hasher = <Self as WindowHasher<u32, u32>>::new(base, modulus).unwrap();

        //     let data1 = vec![1u32, 2, 3];
        //     let data2 = vec![1u32, 2, 3];
        //     let data3 = vec![3u32, 2, 1];

        //     let hash1a: u32 = hasher.hash(&data1);
        //     let hash2a: u32 = hasher.hash(&data2);
        //     let hash3a: u32 = hasher.hash(&data3);
        //     assert_eq!(hash1a, hash2a);
        //     assert_ne!(hash1a, hash3a);

        //     let empty_data: Vec<u32> = vec![];
        //     let empty_hash: u32 = hasher.hash(&empty_data);
        //     assert_eq!(empty_hash, 0u32);
        // }

        // fn test_sliding_hash_owned_comprehensive() {
        //     let data = vec![1u32, 2, 3, 4, 5];
        //     let base = 10u32;
        //     let modulus = 97u32;
        //     let hasher = <Self as WindowHasher<u32, u32>>::new(base, modulus).unwrap();

        //     let results: Vec<(u32, usize)> = hasher.sliding_hash_owned(&data, 3).collect();
        //     assert_eq!(results.len(), 3);

        //     assert_eq!(results[0], (123 % 97, 0));
        //     assert_eq!(results[1], (234 % 97, 1));
        //     assert_eq!(results[2], (345 % 97, 2));
        // }

        // fn test_slow_sliding_hash_owned_comprehensive() {
        //     let data = vec![7u32, 8, 9, 10];
        //     let base = 10u32;
        //     let modulus = 1000u32;
        //     let hasher = <Self as WindowHasher<u32, u32>>::new(base, modulus).unwrap();

        //     let results: Vec<(u32, usize)> = hasher.slow_sliding_hash_owned(&data, 2).collect();
        //     assert_eq!(results.len(), 3);

        //     assert_eq!(results[0], (78u32, 0));
        //     assert_eq!(results[1], (89u32, 1));
        //     assert_eq!(results[2], (100u32, 2));
        // }

        // fn test_single_sliding_hash_comprehensive() {
        //     let data = vec![11u32, 12, 13];
        //     let base = 10u32;
        //     let modulus = 1000u32;

        //     let results: Vec<_> = Self::single_sliding_hash(base, modulus, &data, 2)
        //         .unwrap()
        //         .collect();
        //     assert_eq!(results.len(), 2);

        //     assert_eq!(results[0], (122u32, 0));
        //     assert_eq!(results[1], (133u32, 1));

        //     let results_empty: Vec<_> = Self::single_sliding_hash(base, modulus, &data, 5)
        //         .unwrap()
        //         .collect();
        //     assert_eq!(results_empty.len(), 0);
        // }

        // // Consistency tests between methods

        // fn test_sliding_hash_consistency_u32() {
        //     let data = vec![1u32, 2, 3, 4, 5];
        //     test_sliding_hash_consistency::<u32, u32, Self>(257u32, 1000000007u32, &data, 3);
        // }

        // fn test_sliding_hash_consistency_i32() {
        //     let data = vec![-1i32, -2, 3, -4, 5];
        //     test_sliding_hash_consistency::<i32, i32, Self>(257i32, 1000000007i32, &data, 2);
        // }

        // fn test_sliding_hash_consistency_u64() {
        //     let data = vec![1u64, 2, 3, 4];
        //     test_sliding_hash_consistency::<u64, u64, Self>(257u64, 1000000007u64, &data, 2);
        // }

        // // Edge case combinations

        // fn test_large_data_small_modulus() {
        //     let data: Vec<u16> = (0..1000).map(|i| i as u16).collect();
        //     let base = 257u32;
        //     let modulus = 3u32;

        //     let results: Vec<_> = Self::single_sliding_hash(base, modulus, &data, 10)
        //         .unwrap()
        //         .collect();
        //     for (hash_val, _) in results {
        //         assert!(hash_val < 3u32);
        //     }
        // }

        // fn test_zero_data_large_window() {
        //     let data = vec![0u32; 5];
        //     let base = 257u32;
        //     let modulus = 1000000007u32;

        //     let results: Vec<_> = Self::single_sliding_hash(base, modulus, &data, 10)
        //         .unwrap()
        //         .collect();
        //     assert_eq!(results.len(), 0);
        // }

        // fn test_negative_values_comprehensive() {
        //     let data = vec![-5i16, -10, -15, 0, 5, 10];
        //     let base = 100i32;
        //     let modulus = 1000000007i32;

        //     let hasher = <Self as WindowHasher<i32, i16>>::new(base, modulus).unwrap();
        //     let _hash: i32 = hasher.hash(&data);

        //     let results: Vec<(i32, usize)> = Self::single_sliding_hash(base, modulus, &data, 3)
        //         .unwrap()
        //         .collect();
        //     assert_eq!(results.len(), 4);

        //     for (hash_val, _) in results {
        //         assert!(hash_val >= 0);
        //         assert!(hash_val < modulus);
        //     }
        // }

        // // Generic hash function tests

        // fn test_generic_hash_u8() {
        //     let data = vec![1u8, 2, 3];
        //     test_hash_function_generic::<u8, u8, Self>(10u8, 97u8, &data);
        // }

        // fn test_generic_hash_u16() {
        //     let data = vec![1u16, 2, 3];
        //     test_hash_function_generic::<u16, u16, Self>(257u16, 65521u16, &data);
        // }

        // fn test_generic_hash_i32() {
        //     let data = vec![-1i32, 0, 1];
        //     test_hash_function_generic::<i32, i32, Self>(257i32, 1000000007i32, &data);
        // }

        // // Type boundary tests

        // fn test_type_boundaries_u8() {
        //     let data = vec![0u8, 127u8, 255u8];
        //     let base = 2u16;
        //     let modulus = 256u16;

        //     let hasher = <Self as WindowHasher<u16, u8>>::new(base, modulus).unwrap();
        //     let hash: u16 = hasher.hash(&data);
        //     assert!(hash < modulus);
        // }

        // fn test_type_boundaries_i8() {
        //     let data = vec![i8::MIN, 0i8, i8::MAX];
        //     let base = 2i16;
        //     let modulus = 1000i16;

        //     let hasher = <Self as WindowHasher<i16, i8>>::new(base, modulus).unwrap();
        //     let hash: i16 = hasher.hash(&data);
        //     assert!(hash >= 0);
        //     assert!(hash < modulus);
        // }

        // fn test_deterministic_across_runs() {
        //     let data = vec![42u32, 17, 99, 3, 8];
        //     let base = 31u32;
        //     let modulus = 1009u32;

        //     let mut all_results = Vec::new();
        //     for _ in 0..5 {
        //         let results: Vec<_> = Self::single_sliding_hash(base, modulus, &data, 3)
        //             .unwrap()
        //             .collect();
        //         all_results.push(results);
        //     }

        //     for i in 1..all_results.len() {
        //         assert_eq!(all_results[0], all_results[i]);
        //     }
        // }

        // fn test_hash_distribution() {
        //     let data: Vec<u8> = (0..100).map(|i| i as u8).collect();
        //     let base = 257u32;
        //     let modulus = 1009u32;

        //     let results: Vec<_> = Self::single_sliding_hash(base, modulus, &data, 5)
        //         .unwrap()
        //         .collect();
        //     let hash_set: HashSet<u32> = results.iter().map(|(hash, _)| *hash).collect();

        //     // Should have good distribution - most hashes should be unique
        //     assert!(hash_set.len() as f64 / results.len() as f64 > 0.8);
        // }
    }

    // Automatically implement WindowHasherTests
    impl<THash, TData, T> WindowHasherTests<THash, TData> for T where T: WindowHasher<THash, TData> {}
}
