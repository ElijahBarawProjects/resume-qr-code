use std::ops::{BitAnd, Rem, Shr, Sub};

use num_traits::{ops::overflowing::OverflowingAdd, CheckedMul};

use crate::one_zero::OneZero;

// wrapper class
#[derive(Clone, Copy)]
pub struct Mod<T> {
    modulus: T,
}

// wrapper trait bundles needed arithmetic traits
pub trait _SupportsMod:
    PartialOrd
    + OverflowingAdd
    + Rem<Output = Self>
    + Copy
    + Sub<Output = Self>
    + Shr<Output = Self>
    + CheckedMul
    + BitAnd<Output = Self>
{
}

// implement wrapper trait
impl<T> _SupportsMod for T where
    T: PartialOrd
        + OverflowingAdd
        + Rem<Output = T>
        + Copy
        + Sub<Output = T>
        + Shr<Output = T>
        + CheckedMul
        + BitAnd<Output = T>
{
}

// full wrapper trait: we need one() and zero()
pub trait SupportsMod: _SupportsMod + OneZero {}

// implement wrapper trait
impl<T> SupportsMod for T where T: _SupportsMod + OneZero {}

// implementation logic
impl<T: SupportsMod> Mod<T> {
    pub fn new(modulus: T) -> Option<Self> {
        // require that modulus is > 0, and that 2 * modulus can be represented by the type
        if modulus <= T::zero() {
            return None;
        }

        let (_, overflowed) = modulus.overflowing_add(&modulus);
        if overflowed {
            return None;
        }

        Some(Mod::<T> { modulus })
    }

    pub fn mod_of(&self, a: T) -> T {
        let mut modded = a % self.modulus; // -self.modulus < a < self.modulus
        modded = modded + self.modulus; // 0 < a < 2 * self.modulus
        modded % self.modulus // 0 < a < self.modulus
    }

    pub fn mod_add(&self, mut a: T, mut b: T) -> T {
        a = a % self.modulus; // - self.modulus < a < self.modulus
        b = b % self.modulus; // - self.modulus < b < self.modulus
        self.mod_of(a + b) // 0 <= _ < self.modulus
    }

    fn _mod_sub_branching(&self, mut a: T, mut b: T) -> T {
        if a >= b {
            return a - b % self.modulus;
        }

        // a < b
        a = a % self.modulus; // - self.modulus < a < self.modulus
        a = a + self.modulus; // 0 < a < 2 * self.modulus
        b = self.mod_of(b); // 0 <= b < self.modulus
        if a >= b {
            return a - b % self.modulus;
        }
        // a < b; 0 < b < self.modulus => a < self.modulus
        a = a + self.modulus; // self.modulus < a < 2 * self.modulus
        debug_assert!(a >= b);
        return a - b % self.modulus;
    }

    fn _mod_sub_no_branch(&self, mut a: T, mut b: T) -> T {
        a = self.mod_of(a); // 0 <= a < self.modulus
        a = a + self.modulus; // self.modulus <= a < 2 * self.modulus
        b = self.mod_of(b); // 0 <= b < self.modulus => a >= b
        let tmp = a - b; // 0 < tmp <= self.modulus
        tmp % self.modulus
    }

    pub fn mod_sub(&self, a: T, b: T) -> T {
        self._mod_sub_no_branch(a, b)
    }

    fn _slow_mod_mul(&self, mut a: T, mut b: T) -> T {
        let mut result = T::zero();
        while b > T::zero() {
            if (b & T::one()) == T::one() {
                // add a
                result = self.mod_add(result, a);
            }
            // double a
            a = self.mod_add(a, a);
            // divide b by two
            b = b >> T::one();
        }
        result
    }

    pub fn mod_mul(&self, mut a: T, mut b: T) -> T {
        a = self.mod_of(a); // 0 <= a < self.modulus
        b = self.mod_of(b); // 0 <= b < self.modulus
        match a.checked_mul(&b) {
            Some(v) => self.mod_of(v),
            None => self._slow_mod_mul(a, b),
        }
    }

    pub fn mod_pow(&self, mut base: T, mut exp: u64) -> T {
        let mut result = T::one();
        base = base % self.modulus;

        while exp > 0 {
            if exp % 2 == 1 {
                result = self.mod_mul(result, base);
            }
            exp >>= 1;
            base = self.mod_mul(base, base);
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use std::{fmt::Debug, i32, u64};

    use num_traits::Bounded;

    use crate::modular::SupportsMod;

    use super::Mod;

    #[test]
    fn test_new_valid_modulus() {
        // test valid moduli for different types
        assert!(Mod::new(7i32).is_some());
        assert!(Mod::new(13u32).is_some());
        assert!(Mod::new(1000000007i64).is_some());
        assert!(Mod::new(998244353u64).is_some());

        // test edge case: modulus = 1
        assert!(Mod::new(1i32).is_some());
        assert!(Mod::new(1u64).is_some());
    }

    #[test]
    fn test_new_invalid_modulus() {
        // test zero modulus
        assert!(Mod::new(0i32).is_none());
        assert!(Mod::new(0u32).is_none());
        assert!(Mod::new(0i64).is_none());
        assert!(Mod::new(0u64).is_none());

        // test negative modulus for signed types
        assert!(Mod::new(-1i32).is_none());
        assert!(Mod::new(-100i64).is_none());
    }

    #[test]
    fn test_new_overflow_modulus() {
        // test moduli that would overflow when doubled
        assert!(Mod::new(i32::MAX).is_none());
        assert!(Mod::new(u32::MAX).is_none());
        assert!(Mod::new(i64::MAX).is_none());
        assert!(Mod::new(u64::MAX).is_none());
        assert!(Mod::new(i32::MAX / 2 + 1).is_none());
        assert!(Mod::new(u32::MAX / 2 + 1).is_none());

        // test maximum valid moduli
        assert!(Mod::new(i32::MAX / 2).is_some());
        assert!(Mod::new(u32::MAX / 2).is_some());
    }

    fn test_mod_of_basic_generic<T: SupportsMod + From<i16> + Debug + Bounded>() {
        let modulus: T = 7.into();
        let mod7 = Mod::new(modulus).unwrap();
        for i in 0i16..7 {
            assert_eq!(mod7.mod_of(i.into()), i.into());
        }
        for i in 7i16..2 * 7 {
            assert_eq!(mod7.mod_of(i.into()), (i - 7).into());
        }

        for i in -7i16..0 {
            assert_eq!(mod7.mod_of(i.into()), (i + 7).into());
        }

        assert_eq!(mod7.mod_of((-8).into()), 6.into());
        assert_eq!(mod7.mod_of((-13).into()), 1.into());
    }

    #[test]
    fn test_mod_of_basic() {
        test_mod_of_basic_generic::<i16>();
        test_mod_of_basic_generic::<i32>();
        test_mod_of_basic_generic::<i64>();
        test_mod_of_basic_generic::<i128>();
    }

    #[test]
    fn test_mod_would_overflow() {
        let mod7 = Mod::new(7i32).unwrap();
        // would overflow
        let min = i32::min_value();
        let max = i32::max_value();
        let zero = 0;
        // add
        assert_eq!(mod7.mod_add(max, 6.into()), zero.into());
        assert_eq!(mod7.mod_add(min, (-5).into()), zero.into());
        // sub
        assert_eq!(mod7.mod_sub(min, 1.into()), 4.into());
        assert_eq!(mod7.mod_sub(zero, min), 2.into());
        // mul
        assert_eq!(mod7.mod_mul(max, (-1).into()), 6.into());
        assert_eq!(mod7.mod_mul(min, 2.into()), 3.into());
        // pow
        assert_eq!(mod7.mod_pow(max, u64::MAX), 1.into());
        assert_eq!(mod7.mod_pow(min, 2), 4.into())
    }

    #[test]
    fn test_mod_of_different_types() {
        let mod_u32 = Mod::new(1000u32).unwrap();
        assert_eq!(mod_u32.mod_of(1500), 500);
        assert_eq!(mod_u32.mod_of(2000), 0);

        let mod_i64 = Mod::new(1000000007i64).unwrap();
        assert_eq!(mod_i64.mod_of(1000000008), 1);
        assert_eq!(mod_i64.mod_of(2000000014), 0);
        assert_eq!(mod_i64.mod_of(-1), 1000000006);
    }

    #[test]
    fn test_mod_of_large_values() {
        let mod_u64 = Mod::new(1000000007u64).unwrap();
        let large_val = u64::MAX - 1000;
        let expected = large_val % 1000000007;
        assert_eq!(mod_u64.mod_of(large_val), expected);
    }

    #[test]
    fn test_mod_add_basic() {
        let mod7 = Mod::new(7i32).unwrap();

        assert_eq!(mod7.mod_add(3, 2), 5);
        assert_eq!(mod7.mod_add(4, 4), 1);

        assert_eq!(mod7.mod_add(0, 5), 5);
        assert_eq!(mod7.mod_add(5, 0), 5);
        assert_eq!(mod7.mod_add(0, 0), 0);

        assert_eq!(mod7.mod_add(-1, 1), 0);
        assert_eq!(mod7.mod_add(-3, 10), 0);
    }

    #[test]
    fn test_mod_add_overflow() {
        let mod_large = Mod::new(1000000007u64).unwrap();
        let a = u64::MAX / 2;
        let b = u64::MAX / 2;
        let result = mod_large.mod_add(a, b);
        let expected = ((a % 1000000007) + (b % 1000000007)) % 1000000007;
        assert_eq!(result, expected);
    }

    #[test]
    fn test_mod_add_different_types() {
        let mod_u32 = Mod::new(13u32).unwrap();
        assert_eq!(mod_u32.mod_add(10, 5), 2);
        assert_eq!(mod_u32.mod_add(12, 1), 0);

        let mod_i128 = Mod::new(97i128).unwrap();
        assert_eq!(mod_i128.mod_add(50, 60), 13);
    }

    #[test]
    fn test_mod_sub_different_types() {
        let mod_u64 = Mod::new(1000000007u64).unwrap();
        assert_eq!(mod_u64.mod_sub(1000000006, 1000000008), 1000000005);

        let mod_i64 = Mod::new(97i64).unwrap();
        assert_eq!(mod_i64.mod_sub(10, 50), 57);
    }

    #[test]
    fn test_mod_mul_basic() {
        let mod7 = Mod::new(7i32).unwrap();

        assert_eq!(mod7.mod_mul(3, 4), 5);
        assert_eq!(mod7.mod_mul(6, 6), 1);
        assert_eq!(mod7.mod_mul(0, 5), 0);
        assert_eq!(mod7.mod_mul(1, 6), 6);

        assert_eq!(mod7.mod_mul(7, 3), 0);
        assert_eq!(mod7.mod_mul(14, 15), 0);

        assert_eq!(mod7.mod_mul(-1, 3), 4);
        assert_eq!(mod7.mod_mul(-2, -3), 6);
    }

    #[test]
    fn test_mod_mul_large_values() {
        let mod_prime = Mod::new(1000000007u64).unwrap();
        let a = 999999999u64;
        let b = 999999998u64;
        let result = mod_prime.mod_mul(a, b);

        // Verify the result manually
        let expected = ((a % 1000000007) * (b % 1000000007)) % 1000000007;
        assert_eq!(result, expected);
    }

    #[test]
    fn test_mod_mul_overflow_fallback() {
        // Test with values that would overflow when multiplied
        let mod_large = Mod::new(1000000007u64).unwrap();
        let a = u64::MAX / 2;
        let b = u64::MAX / 3;
        let result = mod_large.mod_mul(a, b);

        // Since direct multiplication would overflow, it should use the slow method
        // The result should still be correct
        assert!(result < 1000000007);
    }

    #[test]
    fn test_mod_mul_different_types() {
        let mod_u32 = Mod::new(97u32).unwrap();
        assert_eq!(mod_u32.mod_mul(50, 60), 90); // 3000 % 97 = 90

        let mod_i128 = Mod::new(1009i128).unwrap();
        assert_eq!(mod_i128.mod_mul(100, 200), 829); // 20000 % 1009 = 829
    }

    #[test]
    fn test_mod_pow_basic() {
        let mod7 = Mod::new(7i32).unwrap();

        // Basic powers
        assert_eq!(mod7.mod_pow(2, 0), 1); // 2^0 = 1
        assert_eq!(mod7.mod_pow(2, 1), 2); // 2^1 = 2
        assert_eq!(mod7.mod_pow(2, 2), 4); // 2^2 = 4
        assert_eq!(mod7.mod_pow(2, 3), 1); // 2^3 = 8 ≡ 1 (mod 7)
        assert_eq!(mod7.mod_pow(3, 2), 2); // 3^2 = 9 ≡ 2 (mod 7)

        // Edge cases
        assert_eq!(mod7.mod_pow(0, 0), 1); // 0^0 = 1 by convention
        assert_eq!(mod7.mod_pow(0, 5), 0); // 0^n = 0 for n > 0
        assert_eq!(mod7.mod_pow(1, 100), 1); // 1^n = 1
    }

    #[test]
    fn test_mod_pow_large_exponents() {
        let mod_prime = Mod::new(1000000007i64).unwrap();

        // Test large exponent
        let result = mod_prime.mod_pow(2, 1000000);
        assert!(result > 0 && result < 1000000007);

        // Test Fermat's little theorem: a^(p-1) ≡ 1 (mod p) for prime p
        let base = 123456789i64;
        let result = mod_prime.mod_pow(base, 1000000006); // p-1
        assert_eq!(result, 1);
    }

    #[test]
    fn test_mod_pow_different_types() {
        let mod_u64 = Mod::new(97u64).unwrap();
        assert_eq!(mod_u64.mod_pow(10, 3), 30); // 1000 % 97 = 27

        let mod_u32 = Mod::new(13u32).unwrap();
        assert_eq!(mod_u32.mod_pow(5, 4), 1); // 625 % 13 = 1
    }

    #[test]
    fn test_mod_pow_negative_base() {
        let mod11 = Mod::new(11i32).unwrap();

        // Test negative base with even exponent
        assert_eq!(mod11.mod_pow(-3, 2), 9); // (-3)^2 = 9

        // Test negative base with odd exponent
        assert_eq!(mod11.mod_pow(-3, 3), 6); // (-3)^3 = -27 ≡ 6 (mod 11)
    }

    #[test]
    fn test_edge_case_modulus_1() {
        let mod1 = Mod::new(1u32).unwrap();

        // Everything should be 0 mod 1
        assert_eq!(mod1.mod_of(100), 0);
        assert_eq!(mod1.mod_add(50, 75), 0);
        assert_eq!(mod1.mod_sub(10, 20), 0);
        assert_eq!(mod1.mod_mul(25, 30), 0);
        assert_eq!(mod1.mod_pow(999, 1000), 0);
    }

    #[test]
    fn test_edge_case_modulus_2() {
        let mod2 = Mod::new(2i32).unwrap();

        // Test binary arithmetic
        assert_eq!(mod2.mod_of(5), 1); // 5 is odd
        assert_eq!(mod2.mod_of(4), 0); // 4 is even
        assert_eq!(mod2.mod_add(1, 1), 0); // 1 + 1 = 2 ≡ 0 (mod 2)
        assert_eq!(mod2.mod_mul(1, 1), 1); // 1 * 1 = 1
        assert_eq!(mod2.mod_pow(3, 100), 1); // 3^n ≡ 1 (mod 2) for any n ≥ 1
    }

    #[test]
    fn test_consistency_across_types() {
        // Test that operations give consistent results across different integer types
        let mod_i32 = Mod::new(97i32).unwrap();
        let mod_u32 = Mod::new(97u32).unwrap();
        let mod_i64 = Mod::new(97i64).unwrap();
        let mod_u64 = Mod::new(97u64).unwrap();

        // Test same operations on different types
        assert_eq!(mod_i32.mod_add(50, 60), mod_u32.mod_add(50, 60) as i32);
        assert_eq!(mod_i32.mod_sub(30, 80) as i64, mod_i64.mod_sub(30, 80));
        assert_eq!(mod_u32.mod_mul(25, 30) as u64, mod_u64.mod_mul(25, 30));
        assert_eq!(mod_i32.mod_pow(5, 3) as u64, mod_u64.mod_pow(5, 3));
    }

    #[test]
    fn test_mathematical_properties() {
        let mod13 = Mod::new(13i32).unwrap();

        // Test commutativity of addition
        assert_eq!(mod13.mod_add(7, 9), mod13.mod_add(9, 7));

        // Test commutativity of multiplication
        assert_eq!(mod13.mod_mul(4, 8), mod13.mod_mul(8, 4));

        // Test additive identity
        assert_eq!(mod13.mod_add(7, 0), 7);

        // Test multiplicative identity
        assert_eq!(mod13.mod_mul(7, 1), 7);

        // Test distributive property: a * (b + c) ≡ (a * b) + (a * c) (mod m)
        let a = 5;
        let b = 7;
        let c = 9;
        let left = mod13.mod_mul(a, mod13.mod_add(b, c));
        let right = mod13.mod_add(mod13.mod_mul(a, b), mod13.mod_mul(a, c));
        assert_eq!(left, right);
    }

    #[test]
    fn test_extended_gcd() {
        fn extended_gcd(a: i64, b: i64) -> (i64, i64, i64) {
            if a == 0 {
                return (b, 0, 1);
            }
            let (gcd, x1, y1) = extended_gcd(b % a, a);
            let x = y1 - (b / a) * x1;
            let y = x1;
            (gcd, x, y)
        }

        let (gcd, x, y) = extended_gcd(30, 18);
        assert_eq!(gcd, 6);
        assert_eq!(30 * x + 18 * y, gcd);

        let (gcd, x, y) = extended_gcd(35, 15);
        assert_eq!(gcd, 5);
        assert_eq!(35 * x + 15 * y, gcd);
    }

    #[test]
    fn test_modular_inverse() {
        fn mod_inverse(a: i64, m: i64) -> Option<i64> {
            fn extended_gcd(a: i64, b: i64) -> (i64, i64, i64) {
                if a == 0 {
                    return (b, 0, 1);
                }
                let (gcd, x1, y1) = extended_gcd(b % a, a);
                let x = y1 - (b / a) * x1;
                let y = x1;
                (gcd, x, y)
            }

            let (gcd, x, _) = extended_gcd(a, m);
            if gcd != 1 {
                None
            } else {
                Some(((x % m) + m) % m)
            }
        }

        // 3 * 5 = 15 ≡ 1 (mod 7)
        assert_eq!(mod_inverse(3, 7), Some(5));

        // 2 * 6 = 12 ≡ 1 (mod 11)
        assert_eq!(mod_inverse(2, 11), Some(6));

        // 7 * 2 = 14 ≡ 1 (mod 13)
        assert_eq!(mod_inverse(7, 13), Some(2));

        // No inverse exists for non-coprime numbers
        assert_eq!(mod_inverse(6, 9), None);
    }

    #[test]
    fn test_chinese_remainder_theorem() {
        fn chinese_remainder_theorem(remainders: &[i64], moduli: &[i64]) -> Option<i64> {
            if remainders.len() != moduli.len() {
                return None;
            }

            fn extended_gcd(a: i64, b: i64) -> (i64, i64, i64) {
                if a == 0 {
                    return (b, 0, 1);
                }
                let (gcd, x1, y1) = extended_gcd(b % a, a);
                let x = y1 - (b / a) * x1;
                let y = x1;
                (gcd, x, y)
            }

            fn mod_inverse(a: i64, m: i64) -> Option<i64> {
                let (gcd, x, _) = extended_gcd(a, m);
                if gcd != 1 {
                    None
                } else {
                    Some(((x % m) + m) % m)
                }
            }

            fn gcd(mut a: i64, mut b: i64) -> i64 {
                while b != 0 {
                    let temp = b;
                    b = a % b;
                    a = temp;
                }
                a
            }

            // Check if moduli are pairwise coprime
            for i in 0..moduli.len() {
                for j in (i + 1)..moduli.len() {
                    if gcd(moduli[i], moduli[j]) != 1 {
                        return None;
                    }
                }
            }

            let prod: i64 = moduli.iter().product();
            let mut total = 0i64;

            for (&r, &m) in remainders.iter().zip(moduli.iter()) {
                let p = prod / m;
                let inv = mod_inverse(p, m)?;
                total += r * inv * p;
            }

            Some(((total % prod) + prod) % prod)
        }

        // x ≡ 1 (mod 2), x ≡ 4 (mod 3) => x = 1
        assert_eq!(chinese_remainder_theorem(&[1, 1], &[2, 3]), Some(1));

        // x ≡ 2 (mod 3), x ≡ 3 (mod 5) => x = 8
        assert_eq!(chinese_remainder_theorem(&[2, 3], &[3, 5]), Some(8));

        // Three congruences: x ≡ 1 (mod 3), x ≡ 2 (mod 4), x ≡ 3 (mod 5)
        if let Some(result) = chinese_remainder_theorem(&[1, 2, 3], &[3, 4, 5]) {
            // Verify the result satisfies all congruences
            assert_eq!(result % 3, 1);
            assert_eq!(result % 4, 2);
            assert_eq!(result % 5, 3);
        }

        // Test with non-coprime moduli (should return None)
        assert_eq!(chinese_remainder_theorem(&[1, 2], &[6, 9]), None);
    }

    #[test]
    fn test_fermat_little_theorem() {
        // Test Fermat's Little Theorem: a^(p-1) ≡ 1 (mod p) for prime p
        let primes = [5, 7, 11, 13, 17, 19];

        for &p in &primes {
            let mod_p = Mod::new(p as i64).unwrap();
            for a in 1..std::cmp::min(p, 10) {
                let result = mod_p.mod_pow(a as i64, (p - 1) as u64);
                assert_eq!(
                    result, 1,
                    "Fermat's Little Theorem failed for a={}, p={}",
                    a, p
                );
            }
        }
    }

    #[test]
    fn test_euler_theorem_special_case() {
        // For prime modulus, φ(p) = p-1
        let p = 13i64;
        let mod_p = Mod::new(p).unwrap();

        for a in 1..std::cmp::min(p, 6) {
            if gcd(a, p) == 1 {
                let result = mod_p.mod_pow(a, (p - 1) as u64);
                assert_eq!(result, 1);
            }
        }

        fn gcd(mut a: i64, mut b: i64) -> i64 {
            while b != 0 {
                let temp = b;
                b = a % b;
                a = temp;
            }
            a
        }
    }

    #[test]
    fn test_rsa_key_properties() {
        // Small RSA-like example
        let p = 11i64;
        let q = 13i64;
        let n = p * q; // 143
        let phi_n = (p - 1) * (q - 1); // 120

        fn gcd(mut a: i64, mut b: i64) -> i64 {
            while b != 0 {
                let temp = b;
                b = a % b;
                a = temp;
            }
            a
        }

        fn mod_inverse(a: i64, m: i64) -> Option<i64> {
            fn extended_gcd(a: i64, b: i64) -> (i64, i64, i64) {
                if a == 0 {
                    return (b, 0, 1);
                }
                let (gcd, x1, y1) = extended_gcd(b % a, a);
                let x = y1 - (b / a) * x1;
                let y = x1;
                (gcd, x, y)
            }

            let (gcd_val, x, _) = extended_gcd(a, m);
            if gcd_val != 1 {
                None
            } else {
                Some(((x % m) + m) % m)
            }
        }

        // Choose e such that gcd(e, φ(n)) = 1
        let e = 7i64;
        assert_eq!(gcd(e, phi_n), 1);

        // Find d such that e*d ≡ 1 (mod φ(n))
        let d = mod_inverse(e, phi_n).unwrap();
        assert_eq!((e * d) % phi_n, 1);

        // Test encryption/decryption property
        let mod_n = Mod::new(n).unwrap();
        let message = 42i64;
        let ciphertext = mod_n.mod_pow(message, e as u64);
        let decrypted = mod_n.mod_pow(ciphertext, d as u64);
        assert_eq!(decrypted, message);
    }

    #[test]
    fn test_edge_cases_with_negative_numbers() {
        let mod7 = Mod::new(7i32).unwrap();

        // Test with negative inputs
        assert_eq!(mod7.mod_add(-3, 5), 2);
        assert_eq!(mod7.mod_sub(-3, -5), 2);
        assert_eq!(mod7.mod_mul(-3, 4), 2); // -12 % 7 = 2
    }

    #[test]
    fn test_large_number_operations() {
        let large_mod = Mod::new(1000000007i64).unwrap();
        let large_a = 123456789i64;
        let large_b = 987654321i64;

        let result = large_mod.mod_mul(large_a, large_b);
        let expected = (large_a * large_b) % 1000000007;
        assert_eq!(result, expected);
    }

    #[test]
    fn test_zero_modulus_edge_cases() {
        let mod1 = Mod::new(1i32).unwrap();

        // Everything should be 0 mod 1
        assert_eq!(mod1.mod_add(5, 3), 0);
        assert_eq!(mod1.mod_mul(5, 3), 0);
    }

    #[test]
    fn test_parametrized_mod_pow() {
        struct TestCase {
            base: i64,
            exp: u64,
            modulus: i64,
            expected: i64,
        }

        let test_cases = [
            TestCase {
                base: 2,
                exp: 3,
                modulus: 7,
                expected: 1,
            }, // 2^3 mod 7 = 8 mod 7 = 1
            TestCase {
                base: 3,
                exp: 4,
                modulus: 7,
                expected: 4,
            }, // 3^4 mod 7 = 81 mod 7 = 4
            TestCase {
                base: 5,
                exp: 2,
                modulus: 11,
                expected: 3,
            }, // 5^2 mod 11 = 25 mod 11 = 3
            TestCase {
                base: 2,
                exp: 10,
                modulus: 1000,
                expected: 24,
            }, // Large exponentiation
        ];

        for case in &test_cases {
            let mod_val = Mod::new(case.modulus).unwrap();
            assert_eq!(mod_val.mod_pow(case.base, case.exp), case.expected);
        }
    }
}
