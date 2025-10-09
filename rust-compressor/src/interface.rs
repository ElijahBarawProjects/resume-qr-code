#[cfg(test)]
use tested_trait::tested_trait;

#[cfg(test)]
use num_traits::Bounded;

#[cfg(test)]
use std::ops::Div;

#[cfg(test)]
use crate::rolling_hash::HashType;

#[allow(dead_code)]
#[derive(Debug)]
pub enum HasherErr {
    InvalidBase(&'static str),
    InvalidModulus(&'static str),
}

#[allow(dead_code)] // TODO: remove
#[cfg_attr(test, tested_trait)]
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
        let num_iters = if window_size == 0 {
            // this is a design choice, as it's unclear how many zero-length substrings there are
            // considered but not done: |data| + 1 elements equal to THash::zero()
            0
        } else {
            (data.len() + 1).saturating_sub(window_size)
        };

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

    /**
     * Tests
     */

    #[test]
    fn test_type_boundaries()
    where
        TData: Bounded + TryFrom<u8>,
        THash: Div<Output = THash> + Bounded + TryFrom<u8> + PartialOrd + Copy,
    {
        <Self as WindowHasher<THash, TData>>::test_type_boundaries_generic();
    }

    #[test]
    fn test_new_function_generic()
    where
        THash: crate::rolling_hash::HashType,
        TData: Into<num_bigint::BigInt>,
    {
        let base: THash = tests::unchecked_make_number(10);
        let modulus: THash = tests::unchecked_make_number(63);
        let hasher = Self::new(base, modulus);

        assert!(hasher.is_ok());
    }

    /**
     * Test Helpers
     */

    #[cfg(test)]
    fn test_sliding_hash_consistency(
        base: THash,
        modulus: THash,
        data: &[TData],
        window_size: usize,
    ) where
        THash: HashType + std::fmt::Debug,
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

    #[cfg(test)]
    fn test_type_boundaries_generic()
    where
        TData: Bounded + TryFrom<u8>,
        THash: Div<Output = THash> + Bounded + TryFrom<u8> + PartialOrd + Copy,
    {
        let data = vec![
            TData::min_value(),
            TData::try_from(0u8).ok().unwrap(),
            TData::max_value(),
        ];
        let base = THash::try_from(2u8).ok().unwrap(); // multiplying max by two would overflow
        let modulus = THash::max_value() / base;

        let hasher = Self::new(base, modulus).unwrap();
        let hash = hasher.hash(&data);
        assert!(hash >= 0u8.try_into().ok().unwrap());
        assert!(hash < modulus);
    }
}

#[cfg(test)]
pub mod tests {

    use tested_trait::tested_trait;

    use std::fmt::Debug;

    use crate::rolling_hash::HashType;

    use super::WindowHasher;
    use num_bigint::BigInt;
    use num_traits::{Bounded, Signed};

    /// Given a BigInt, attempt to construct the value in THash
    ///
    /// Can overflow or underflow; what happens in that case depends on THash
    pub fn unchecked_make_number<THash: HashType, TNum: Into<BigInt>>(num: TNum) -> THash {
        let num = num.into();
        let zero = THash::zero();
        let one = THash::one();
        let two = one + one;
        let magnitude = num.magnitude();
        let mut res = zero;
        let numbits = magnitude.bits();
        for bit in 0..numbits {
            let bit_val = magnitude.bit(numbits - bit - 1);

            res = res * two; // lshift not required
            if bit_val {
                res = res + one;
            }
        }

        match num.sign() {
            num_bigint::Sign::Minus => zero - res,
            num_bigint::Sign::NoSign | num_bigint::Sign::Plus => res,
        }
    }

    #[test]
    fn test_make_number() {
        assert_eq!(unchecked_make_number::<i32, _>(10), 10);
        assert_eq!(unchecked_make_number::<i32, _>(-10), -10);
        assert_eq!(unchecked_make_number::<i32, _>(0), 0);
        assert_eq!(unchecked_make_number::<i32, _>(1), 1);
    }

    macro_rules! expect_from {
        ($num:expr) => {
            $num.try_into().ok().unwrap()
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

        #[test]
        fn _test_hash_calculation_manual()
        where
            THash: HashType + Debug,
            TData: TryFrom<u8>,
        {
            let data = vec![expect_from!(1u8), expect_from!(2)];
            let base = unchecked_make_number(10u32);
            let modulus = unchecked_make_number(97u32);
            let hasher = Self::single_sliding_hash(base, modulus, &data, 2).unwrap();
            let results: Vec<_> = hasher.collect();

            assert_eq!(results.len(), 1);
            // Manual calculation: (1 * 10 + 2) % 97 = 12
            assert_eq!(results[0], (unchecked_make_number(12u8), 0));
        }

        #[test]
        fn _test_sliding_hash_consistency_unsigned()
        where
            THash: HashType + Debug,
            TData: TryFrom<u8>,
        {
            let data: Vec<u8> = vec![1, 2, 3, 4, 5];
            let data: Vec<_> = data.into_iter().map(|x| expect_from!(x)).collect();
            let modulus = unchecked_make_number(5);
            let base = unchecked_make_number(127);
            let window_size = 3;
            Self::test_sliding_hash_consistency(base, modulus, &data, window_size);
        }

        #[test]
        fn _test_sliding_hash_owned_comprehensive()
        where
            THash: HashType + Debug,
            TData: TryFrom<u8>,
        {
            let data: Vec<_> = vec![1u8, 2, 3, 4, 5]
                .into_iter()
                .map(|x| expect_from!(x))
                .collect();
            let base = unchecked_make_number(10u32);
            let modulus = unchecked_make_number(97u32);
            let hasher = Self::new(base, modulus).unwrap();

            let results: Vec<_> = hasher.sliding_hash_owned(&data, 3).collect();
            assert_eq!(results.len(), 3);

            assert_eq!(results[0], (unchecked_make_number((123 % 97) as u8), 0));
            assert_eq!(results[1], (unchecked_make_number((234 % 97) as u8), 1));
            assert_eq!(results[2], (unchecked_make_number((345 % 97) as u8), 2));
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
        fn _test_small_window_sizes()
        where
            THash: HashType + Debug,
            TData: TryFrom<u8>,
        {
            let data: Vec<_> = vec![1u8, 2, 3, 4, 5, 6, 7, 8, 9]
                .into_iter()
                .map(|x| expect_from!(x))
                .collect();
            let base = unchecked_make_number(10u8);
            let modulus = unchecked_make_number(97u8);

            for window_size in 3..=9 {
                let results: Vec<_> = Self::single_sliding_hash(base, modulus, &data, window_size)
                    .unwrap()
                    .collect();
                assert_eq!(results.len(), data.len() - window_size + 1);
            }
        }

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
            assert_eq!(results.len(), 0);

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

        #[test]
        fn _test_hash_calculation_with_modulo()
        where
            THash: HashType + Debug,
        {
            let data: Vec<_> = vec![100u8, 100, 100]
                .into_iter()
                .map(|x| expect_from!(x))
                .collect();
            let base = unchecked_make_number(10u32);
            let modulus = unchecked_make_number(7u32);
            let hasher = Self::single_sliding_hash(base, modulus, &data, 2).unwrap();
            let results: Vec<_> = hasher.collect();

            assert_eq!(results.len(), 2);
            assert_eq!(results[0], (unchecked_make_number(1), 0));
            assert_eq!(results[1], (unchecked_make_number(1), 1));
        }
    }

    #[tested_trait]
    pub trait WindowhasherBoundedTDataTests<THash: TryFrom<u32>, TData: Bounded>
    where
        Self: WindowHasher<THash, TData>,
    {
        #[test]
        fn _test_min_value_data_signed() {
            let data = vec![TData::min_value(), TData::min_value(), TData::min_value()];

            // hash
            let base = expect_from!(257u32);
            let modulus = expect_from!(1000000007u32);
            let hasher = Self::new(base, modulus).unwrap();
            let _hash = hasher.hash(&data);

            // check
            let base = expect_from!(257u32);
            let modulus = expect_from!(1000000007u32);
            let results: Vec<_> = Self::single_sliding_hash(base, modulus, &data, 2)
                .unwrap()
                .collect();
            assert_eq!(results.len(), 2);
        }
    }

    #[tested_trait]

    pub trait WindowHasherBoundedTHashTests<THash: Bounded, TData>
    where
        Self: WindowHasher<THash, TData>,
    {
        #[test]
        fn _test_large_modulus()
        where
            THash: HashType + TryFrom<u32>,
            TData: TryFrom<u8>,
        {
            let data: Vec<_> = vec![1u8, 2, 3]
                .into_iter()
                .map(|x| expect_from!(x))
                .collect();
            let base = unchecked_make_number(257u16);
            let modulus = THash::max_value() >> THash::one();

            let hasher = Self::new(base, modulus).unwrap();
            let hash = hasher.hash(&data);
            assert!(hash < modulus);
        }

        #[test]
        fn _test_max_value_data_unsigned()
        where
            THash: HashType + TryFrom<u32>,
            TData: TryFrom<u16> + Bounded,
        {
            let data: Vec<_> = vec![TData::max_value(), TData::max_value(), TData::max_value()]
                .into_iter()
                .map(|x| expect_from!(x))
                .collect();
            let base = expect_from!(257u32);
            let modulus = expect_from!(1000000007u32);

            let hasher = Self::new(base, modulus).unwrap();
            let hash = hasher.hash(&data);
            assert!(hash < modulus);
        }
    }

    #[tested_trait]
    pub trait WindowHasherSignedDataTests<THash, TData: Signed>
    where
        Self: WindowHasher<THash, TData>,
    {
        #[test]
        fn _test_mixed_positive_negative_zero()
        where
            THash: HashType + TryFrom<i32> + Debug,
            TData: TryFrom<i8>,
        {
            let data: Vec<_> = vec![-100i8, 0, 100, -50, 0, 75]
                .into_iter()
                .map(|x| expect_from!(x))
                .collect();
            let base = expect_from!(257i32);
            let modulus = expect_from!(1000000007i32);

            let hasher = Self::new(base, modulus).unwrap();
            let hash1 = hasher.hash(&data);
            let hash2 = hasher.hash(&data);
            assert_eq!(hash1, hash2);

            let results: Vec<_> = Self::single_sliding_hash(base, modulus, &data, 3)
                .unwrap()
                .collect();
            assert_eq!(results.len(), 4);
        }

        #[test]
        fn _test_sliding_hash_consistency_signed()
        where
            THash: HashType + Debug,
            TData: TryFrom<i8>,
        {
            let data: Vec<i8> = vec![-1, -2, 3, -4, 5];
            let data: Vec<_> = data.into_iter().map(|x| expect_from!(x)).collect();
            let modulus = unchecked_make_number(5);
            let base = unchecked_make_number(257);
            let window_size = 3;
            Self::test_sliding_hash_consistency(base, modulus, &data, window_size);
        }

        fn test_negative_values_comprehensive(modulus: THash)
        where
            THash: HashType,
            TData: TryFrom<i8>,
        {
            let data: Vec<_> = vec![-5i8, -10, -15, 0, 5, 10]
                .into_iter()
                .map(|x| expect_from!(x))
                .collect();
            let base = unchecked_make_number(100u8);

            let hasher = Self::new(base, modulus).unwrap();
            let _hash = hasher.hash(&data);

            let results: Vec<_> = Self::single_sliding_hash(base, modulus, &data, 3)
                .unwrap()
                .collect();
            assert_eq!(results.len(), 4);

            for (hash_val, _) in results {
                assert!(hash_val >= unchecked_make_number(0));
                assert!(hash_val < modulus);
            }
        }

        #[test]
        fn _test_negative_values_comprehensive()
        where
            THash: HashType,
            TData: TryFrom<i8>,
        {
            Self::test_negative_values_comprehensive(unchecked_make_number(127))
            // TODO: TEST WITH LARGER MODULUS
        }

        #[test]
        fn _test_negative_data()
        where
            THash: HashType + TryFrom<i32> + Debug,
            TData: TryFrom<i8>,
        {
            let data: Vec<_> = vec![-1, -2, -3]
                .into_iter()
                .map(|x| expect_from!(x))
                .collect();
            let base = expect_from!(257i32);
            let modulus_as_i32 = 1000000007i32;
            let modulus = expect_from!(modulus_as_i32);
            let hasher = Self::single_sliding_hash(base, modulus, &data, 1).unwrap();
            let results: Vec<_> = hasher.collect();
            let expected: Vec<_> = vec![modulus_as_i32 - 1, modulus_as_i32 - 2, modulus_as_i32 - 3]
                .into_iter()
                .enumerate()
                .map(|(start, hash)| (expect_from!(hash), start))
                .collect();
            assert_eq!(results, expected);
        }

        #[test]
        fn _test_negative_with_window()
        where
            THash: HashType + TryFrom<i32> + Debug,
            TData: TryFrom<i8>,
        {
            let data: Vec<_> = vec![-1i8, -2, -3, -4, -5, -6]
                .into_iter()
                .map(|x| expect_from!(x))
                .collect();
            let modulus_as_i32 = 1000000007i32;
            let modulus = expect_from!(modulus_as_i32);
            let base = expect_from!(257i32);
            let hasher = Self::single_sliding_hash(base, modulus, &data, 2).unwrap();
            let results: Vec<_> = hasher.collect();
            let expected: Vec<_> = vec![
                modulus_as_i32 - 259,
                modulus_as_i32 - 517,
                modulus_as_i32 - 775,
                modulus_as_i32 - 1_033,
                modulus_as_i32 - 1_291,
            ]
            .into_iter()
            .enumerate()
            .map(|(start, hash)| (expect_from!(hash), start))
            .collect();
            assert_eq!(results, expected);
        }
    }

    /**
     * Some larger tests, specifically THash = u32, TData = u16
     */
    #[tested_trait]
    pub trait WindowHasherLargeUnsignedTests<THash, TData>
    where
        Self: WindowHasher<THash, TData>,
    {
        #[test]
        fn _test_large_data_small_modulus()
        where
            THash: HashType + TryFrom<u32>,
            TData: TryFrom<u16>,
        {
            let data: Vec<_> = (0..1000).map(|i| expect_from!(i as u16)).collect();
            let base = unchecked_make_number(257u32);
            let modulus = unchecked_make_number(3u32);

            let results: Vec<_> = Self::single_sliding_hash(base, modulus, &data, 10)
                .unwrap()
                .collect();
            for (hash_val, _) in results {
                assert!(hash_val < unchecked_make_number(3u32));
            }
        }

        #[test]
        fn _test_single_sliding_hash_comprehensive()
        where
            THash: Debug + HashType + TryFrom<u32>,
            TData: TryFrom<u16>,
        {
            let data: Vec<_> = vec![11u16, 12, 13]
                .into_iter()
                .map(|x| expect_from!(x))
                .collect();
            let base = unchecked_make_number(10u32);
            let modulus = unchecked_make_number(1000u32);

            let results: Vec<_> = Self::single_sliding_hash(base, modulus, &data, 2)
                .unwrap()
                .collect();
            assert_eq!(results.len(), 2);

            assert_eq!(results[0], (unchecked_make_number(122u32), 0));
            assert_eq!(results[1], (unchecked_make_number(133u32), 1));

            let results_empty: Vec<_> = Self::single_sliding_hash(base, modulus, &data, 5)
                .unwrap()
                .collect();
            assert_eq!(results_empty.len(), 0);
        }

        #[test]
        fn _test_large_modulus_small_data()
        where
            THash: Debug + HashType + TryFrom<u32>,
            TData: TryFrom<u16>,
        {
            let data: Vec<_> = vec![1u16, 2, 3]
                .into_iter()
                .map(|x| expect_from!(x))
                .collect();
            let modulus = unchecked_make_number(10);
            let base = unchecked_make_number(1000000007);

            let hasher = Self::single_sliding_hash(modulus, base, &data, 2).unwrap();
            let results: Vec<_> = hasher.collect();

            assert_eq!(results.len(), 2);
            assert_eq!(results[0], (unchecked_make_number(12), 0));
            assert_eq!(results[1], (unchecked_make_number(23), 1));
        }

        #[test]
        fn _test_all_zero_data()
        where
            THash: Debug + HashType + TryFrom<u32>,
            TData: TryFrom<u8>,
        {
            let mut data = vec![];
            for _ in 0..100 {
                data.push(expect_from!(0u8));
            }
            let base = unchecked_make_number(257u32);
            let modulus = unchecked_make_number(1000000007u32);

            let hasher = Self::new(base, modulus).unwrap();
            let hash = hasher.hash(&data);
            assert_eq!(hash, unchecked_make_number(0));

            let results: Vec<_> = Self::single_sliding_hash(base, modulus, &data, 10)
                .unwrap()
                .collect();
            for (hash_val, _) in results {
                assert_eq!(hash_val, unchecked_make_number(0));
            }
        }

        #[test]
        fn _test_large_data_large_base_modulus_one()
        where
            THash: Debug + HashType + TryFrom<u32>,
            TData: TryFrom<u8>,
        {
            let data: Vec<_> = vec![42u8, 17, 99]
                .into_iter()
                .map(|x| expect_from!(x))
                .collect();
            let base = unchecked_make_number(257u32);
            let modulus = unchecked_make_number(1u32);

            let hasher = Self::new(base, modulus).unwrap();
            let hash = hasher.hash(&data);
            assert_eq!(hash, unchecked_make_number(0u32));

            let results: Vec<_> = Self::single_sliding_hash(base, modulus, &data, 2)
                .unwrap()
                .collect();
            for (hash_val, _) in results {
                assert_eq!(hash_val, unchecked_make_number(0u32));
            }
        }

        #[test]
        fn _test_slow_sliding_hash_owned_comprehensive()
        where
            THash: Debug + HashType + TryFrom<u32>,
            TData: TryFrom<u16>,
        {
            let data: Vec<_> = vec![7u16, 8, 9, 10]
                .into_iter()
                .map(|x| expect_from!(x))
                .collect();
            let base = unchecked_make_number(10u32);
            let modulus = unchecked_make_number(1000u32);
            let hasher = Self::new(base, modulus).unwrap();

            let results: Vec<_> = hasher.slow_sliding_hash_owned(&data, 2).collect();
            assert_eq!(results.len(), 3);

            assert_eq!(results[0], (unchecked_make_number(78u32), 0));
            assert_eq!(results[1], (unchecked_make_number(89u32), 1));
            assert_eq!(results[2], (unchecked_make_number(100u32), 2));
        }

        #[test]
        fn _test_large_modulus_operations()
        where
            THash: Debug + HashType + TryFrom<u32> + Bounded,
            TData: TryFrom<u16>,
        {
            let data: Vec<_> = vec![255u16, 255, 255]
                .into_iter()
                .map(|x| expect_from!(x))
                .collect();
            let base = expect_from!(256u32);
            let modulus_as_u32 = u32::MAX >> 1;
            // let modulus = THash::max_value() >> THash::one(); // Large modulus
            let modulus = expect_from!(modulus_as_u32);
            let hasher = Self::single_sliding_hash(base, modulus, &data, 3).unwrap();
            let results: Vec<_> = hasher.collect();

            assert_eq!(results.len(), 1);
            // Should handle large numbers correctly
            let pre_mod = 255u64 * 256 * 256 + 255 * 256 + 255;
            let expected = (pre_mod % modulus_as_u32 as u64) as u32;
            assert_eq!(results[0].0, expect_from!(expected));
        }

        fn _test_new_function_comprehensive()
        where
            THash: Debug + HashType + TryFrom<u32>,
            TData: TryFrom<u16>,
        {
            let base = expect_from!(257u32);
            let modulus = expect_from!(1000000007u32);

            let hasher1 = Self::new(base, modulus).unwrap();
            let hasher2 = Self::new(base, modulus).unwrap();

            let data: Vec<_> = vec![1u16, 2, 3]
                .into_iter()
                .map(|x| expect_from!(x))
                .collect();
            let hash1 = hasher1.hash(&data);
            let hash2 = hasher2.hash(&data);
            assert_eq!(hash1, hash2);
        }
    }

    #[tested_trait]
    pub trait WindowHasherLargeSignedTests<THash, TData>
    where
        Self: WindowHasher<THash, TData>,
    {
        // TODO: this isn't passing
        #[test]
        fn test_negative_modulus()
        where
            THash: HashType + Debug + TryFrom<i32>,
            TData: TryFrom<i16>,
        {
            let data: Vec<_> = vec![-1i16, -2, -3]
                .into_iter()
                .map(|x| expect_from!(x))
                .collect();
            let base = unchecked_make_number(257i32);
            let modulus = unchecked_make_number(1000000007i32);

            let hasher = Self::new(base, modulus).unwrap();
            let _hash = hasher.hash(&data);
            let results: Vec<_> = Self::single_sliding_hash(base, modulus, &data, 2)
                .unwrap()
                .collect();
            assert_eq!(results.len(), 2);
        }
    }

    // Automatically implement WindowHasherTests

    impl<THash, TData, T: WindowHasher<THash, TData>> WindowHasherDataSizeTests<THash, TData> for T {}
    impl<THash, TData, T: WindowHasher<THash, TData>> WindowHasherDataTests<THash, TData> for T {}
    impl<THash, TData, T: WindowHasher<THash, TData>> WindowHasherWindowSizeTests<THash, TData> for T {}
    impl<THash, TData, T: WindowHasher<THash, TData>> WindowHasherModulusTests<THash, TData> for T {}
    impl<TH, TD, T: WindowHasher<TH, TD>> WindowHasherLargeUnsignedTests<TH, TD> for T {}
    impl<THash, TData, T: WindowHasher<THash, TData>> WindowHasherLargeSignedTests<THash, TData> for T {}
    impl<TH: TryFrom<u32>, TD: Bounded, T: WindowHasher<TH, TD>>
        WindowhasherBoundedTDataTests<TH, TD> for T
    {
    }
    impl<TH: Bounded, TD, T: WindowHasher<TH, TD>> WindowHasherBoundedTHashTests<TH, TD> for T {}
    impl<THash, TData: Signed, T: WindowHasher<THash, TData>>
        WindowHasherSignedDataTests<THash, TData> for T
    {
    }
}
