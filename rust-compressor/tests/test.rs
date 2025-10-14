use num_bigint::{BigInt, TryFromBigIntError};
use rolling_hash_rs;

use rolling_hash_rs::interface::WindowHasher;
use rolling_hash_rs::{
    optimal_substring::best_substring,
    rolling_hash::{HashType, DEFAULT_BASE, DEFAULT_MOD_U64},
};

use rolling_hash_rs::interface_non_rolling_hash::InterfaceHasher;
use rolling_hash_rs::rolling_hash::WrappedRollingHash;

fn _test_rolling_hash_str_once<TH, TD>(text: &Vec<TD>, window_size: usize, base: TH, modulus: TH)
where
    TH: std::fmt::Debug + HashType + From<TD> + Into<BigInt> + TryFrom<BigInt>,
    TD: Copy + Into<BigInt>,
{
    // Get rolling hash results

    let rolling_results: Vec<_> =
        <WrappedRollingHash<TH> as rolling_hash_rs::interface::WindowHasher<TH, TD>>::new(
            base, modulus,
        )
        .unwrap()
        .sliding_hash_owned(text, window_size)
        .collect();

    // Calculate expected results manually
    let expected: Vec<_> =
        <InterfaceHasher as rolling_hash_rs::interface::WindowHasher<TH, TD>>::new(base, modulus)
            .unwrap()
            .sliding_hash_owned(text, window_size)
            .collect();

    assert_eq!(rolling_results.len(), expected.len());
    for ((rolling_hash, rolling_start), (non_roll_hash, non_roll_start)) in
        rolling_results.into_iter().zip(expected)
    {
        assert_eq!(rolling_hash, non_roll_hash);
        assert_eq!(rolling_start, non_roll_start);
    }
}

fn _test_rolling_hash_basic_str<TH, TD>(
    text: Vec<TD>,
    min_len: usize,
    max_len: usize,
    base: TH,
    modulus: TH,
) where
    TH: std::fmt::Debug + HashType + Into<BigInt> + TryFrom<BigInt> + From<TD>,
    TD: Copy + Into<BigInt>,
{
    for window_size in min_len..=max_len {
        _test_rolling_hash_str_once(&text, window_size, base, modulus);
    }
}

fn test_rolling_hash_basic<TH, TD>(modulus: TH)
where
    TH: std::fmt::Debug + Into<BigInt> + TryFrom<BigInt> + HashType + From<u16> + From<TD>,
    TD: Copy + Into<BigInt> + TryFrom<char>,
{
    let text = "abcdefg";
    let text = text
        .chars()
        .map(|char| char.try_into().ok().unwrap())
        .collect();
    let min_len = 3;
    let max_len = 50;
    let base = TH::from(DEFAULT_BASE);
    _test_rolling_hash_basic_str::<TH, TD>(text, min_len, max_len, base, modulus);
}

const RESUME: &'static str = "<!doctypehtml><meta charset=\"utf-8\"><center><p><b><a href=\"https://www.elijahbaraw.com/\" style=\"color:#00f\"><u>Elijah Baraw</u></a></b><br><a href=\"mailto:ebaraw@andrew.cmu.edu?subject=RE:%20Your%20Resume\" style=\"color:#00f\"><u>ebaraw@andrew.cmu.edu</u></a> (203) 731-9535 Pittsburgh, PA <a href=\"https://github.com/elijah-bae-raw\" style=\"color:#00f\"><u>github.com/elijah-bae-raw</u></a></center><hr><h1>Education</h1><p><b>Carnegie Mellon University, School of Computer Science</b> <b>Aug 2021 - May 2025</b><br><em>Bachelor of Science in Computer Science. Concentration in Computer Systems</em><br>GPA: 3.97. Relevant courses: Machine Learning, Cloud Computing, Distributed Systems, Data Structures and<br>Algorithms, Functional Programming, Systems, Parallel Algorithms, Linear Algebra, Differential Equations, IDL<h1>Technical Skills</h1><p><b>Languages:</b> C, Python, Go, SQL, Java, HCL<br><b>Technologies:</b> NumPy, PyTorch, Pandas, OpenCV, Linux, Sockets, Git, AWS, GCP, Azure, K8s, Docker<br><b>Topics:</b> Data Structures and Algorithms, Object Oriented Programing, Functional Programming, Systems,<br>Consensus Algorithms, Actor Model, Network Protocols, TCP, Cryptographic Algorithms, Machine Learning<h1>Experience</h1><p><b>Center for Atmospheric Particle Studies</b> <b>Pittsburgh, PA</b><br><b>Research Assistant</b> <b>May 2022 - Aug 2022</b><ul><li>Developed a low cost device for measuring PM2.5 air pollutants collected on a foam tape over several months<li>Implemented an image processing pipeline as a cheaper alternative to traditional particle detection machines<li>Integrated a system of Arduino and Python scripts to control stepper motors based on continuous CV input.</ul><h1>Projects</h1><p><b>Fontify (Solo Python Project)</b> <em>PIL, Image Processing, De-Noising, Computer Vision</em> <b>Jan 2022 - July 2022</b><ul><li>Created image processing software in Python to convert handwritten letters into a personalized bitmap font.<li>Utilized image processing, edge-detection, noise reduction algorithms to detect pencil writing on paper.</ul><p><b>Concurrent Proxy Server (C)</b> <em>Git, HTTP, Sockets</em> <b>July 2023</b><ul><li>Developed a proxy server in C using p_threads and fork to handle requests concurrently, anonymize traffic and cache responses. Utilized Unix sockets and a bounded cache following an LRU eviction policy.</ul><p><b>Distributed Backend (Golang)</b> <em>Replication, Actor Model, Mailbox/Message Passing</em> <b>Nov 2023</b><ul><li>Designed and executed a concurrent server to manage the state for a multiplayer game, accessible via API.<li>Handled client requests about and updates to the game state using RPCs and a message-passing model.<li>Implemented node launching and server groups, ensuring replication and enforcing consistency within groups.</ul><p><b>Poker-Bots Hackathon Dev Team</b> <em>GCP, K8s, GitHub Actions</em> <b>Mar 2024; Mar 2025</b><ul><li>Helped CMU Data Science Club run their first AI Poker-Bot competition, with $6,000 in prizes and 63 teams.<li>Used GitHub Actions to automatically build Docker images of user-submitted Python bots, allowing competitors to use custom dependencies and machine learning libraries of their choice, running containers on GCP.<li>Helped build the second iteration of the competition in 2025 using AWS ECS for bots, Lambda for matches</ul><p><b>x86 IA-32 Kernel from Scratch</b> <em>C, ASM, Simics</em> <b>Aug 2024 - Dec 2024</b><ul><li>Built a complete i386 kernel from scratch, solo, for CMU 15410, implementing preemptive multitasking<li>Engineered hardware interfaces, memory management, and I/O system for concurrent ELF binary execution</ul>";

#[test]
fn test_rolling_hash_integers() {
    #[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
    struct Int {
        val: i32,
    }

    impl Into<i32> for Int {
        fn into(self) -> i32 {
            self.val
        }
    }
    impl From<i32> for Int {
        fn from(val: i32) -> Self {
            Self { val }
        }
    }
    impl From<char> for Int {
        fn from(value: char) -> Self {
            Self { val: value as i32 }
        }
    }
    impl From<u16> for Int {
        fn from(val: u16) -> Self {
            Self { val: val as i32 }
        }
    }

    impl Into<BigInt> for Int {
        fn into(self) -> BigInt {
            self.val.into()
        }
    }

    impl std::ops::Add for Int {
        type Output = Int;

        fn add(self, rhs: Self) -> Self::Output {
            Self {
                val: self.val + rhs.val,
            }
        }
    }

    impl std::ops::Mul for Int {
        type Output = Int;

        fn mul(self, rhs: Self) -> Self::Output {
            Self {
                val: self.val * rhs.val,
            }
        }
    }
    impl TryFrom<BigInt> for Int {
        type Error = TryFromBigIntError<BigInt>;

        fn try_from(value: BigInt) -> Result<Self, Self::Error> {
            match value.try_into() {
                Ok(val) => Ok(Int { val }),
                Err(e) => Err(e),
            }
        }
    }
    impl std::ops::Sub for Int {
        type Output = Int;

        fn sub(self, rhs: Self) -> Self::Output {
            Self {
                val: self.val - rhs.val,
            }
        }
    }
    impl std::ops::Rem for Int {
        type Output = Int;

        fn rem(self, rhs: Self) -> Self::Output {
            Self {
                val: self.val % rhs.val,
            }
        }
    }

    impl std::ops::BitAnd for Int {
        type Output = Int;

        fn bitand(self, rhs: Self) -> Self::Output {
            Self {
                val: self.val & rhs.val,
            }
        }
    }

    impl std::ops::Shr for Int {
        type Output = Int;

        fn shr(self, rhs: Self) -> Self::Output {
            Self {
                val: self.val >> rhs.val,
            }
        }
    }
    impl num_traits::CheckedMul for Int {
        fn checked_mul(&self, v: &Self) -> Option<Self> {
            match self.val.checked_mul(v.val) {
                Some(val) => Some(Self { val }),
                None => None,
            }
        }
    }

    impl num_traits::ops::overflowing::OverflowingAdd for Int {
        fn overflowing_add(&self, v: &Self) -> (Self, bool) {
            let (val, overflowed) = self.val.overflowing_add(v.val);
            (Self { val }, overflowed)
        }
    }

    impl From<u8> for Int {
        fn from(val: u8) -> Self {
            Self { val: val.into() }
        }
    }

    test_rolling_hash_basic::<Int, Int>(1_000_000_007.into());
    test_rolling_hash_basic::<i64, u32>(1_000_000_007);
    test_rolling_hash_basic::<i128, u32>(1_000_000_007);

    test_rolling_hash_basic::<u32, u32>(1_000_000_007);
    test_rolling_hash_basic::<u64, u32>(1_000_000_007);
    test_rolling_hash_basic::<u128, u32>(1_000_000_007);

    test_rolling_hash_basic::<u64, u32>(DEFAULT_MOD_U64);
    test_rolling_hash_basic::<u128, u32>(DEFAULT_MOD_U64.into());

    type TH = u64;
    type TD = u32;
    let simple: Vec<_> = "abc"
        .chars()
        .map(|a| TD::try_from(a).ok().unwrap())
        .collect();
    let resume: Vec<_> = RESUME
        .chars()
        .map(|a| TD::try_from(a).ok().unwrap())
        .collect();
    _test_rolling_hash_basic_str::<TH, TD>(simple, 3, 50, DEFAULT_BASE.into(), 1_000_000_007);
    _test_rolling_hash_basic_str::<TH, TD>(resume, 3, 50, DEFAULT_BASE.into(), DEFAULT_MOD_U64);
}

#[test]
fn test_optimal_substring() {
    type TH = u128;
    let res = best_substring::<TH, usize>(
        RESUME,
        '|',
        false,
        "~0",
        true,
        3,
        50,
        DEFAULT_BASE.into(),
        DEFAULT_MOD_U64.into(),
        true,
        1,
    );

    let expected = dummy_best_substring();

    assert_eq!(expected, res);
}

use optimal_substring::BestSubstringRes;
fn dummy_best_substring() -> Option<BestSubstringRes<usize, usize>> {
    Some(BestSubstringRes {
        score: 47,
        count: 6,
        substring: "</b><ul><li>".to_string(),
    })
}

#[test]
fn test_optimal_substring_par() {
    type TH = u128;
    let par_res = best_substring::<TH, usize>(
        RESUME,
        '|',
        false,
        "~0",
        true,
        3,
        50,
        DEFAULT_BASE.into(),
        DEFAULT_MOD_U64.into(),
        true,
        5,
    )
    .unwrap();
    let seq_res = best_substring::<TH, usize>(
        RESUME,
        '|',
        false,
        "~0",
        true,
        3,
        50,
        DEFAULT_BASE.into(),
        DEFAULT_MOD_U64.into(),
        true,
        5,
    )
    .unwrap();

    assert_eq!(par_res, seq_res);
    assert_eq!(seq_res, dummy_best_substring().unwrap());
}
use rolling_hash_rs::optimal_substring;

#[test]
fn test_most_common_substring() {
    type TH = u64;
    type TD = u32;
    let resume: Vec<_> = RESUME
        .chars()
        .map(|a| TD::try_from(a).ok().unwrap())
        .collect();
    let len: usize = 50;
    let rolling =
        optimal_substring::most_common_of_len::<TD, usize, usize, TH, WrappedRollingHash<TH>>(
            &resume,
            &|_, _, _| true,
            true,
            len,
            DEFAULT_BASE.into(),
            DEFAULT_MOD_U64.into(),
            true,
        );
    let non_rolling = optimal_substring::most_common_of_len::<TD, usize, usize, TH, InterfaceHasher>(
        &resume,
        &|_, _, _| true,
        true,
        len,
        DEFAULT_BASE.into(),
        DEFAULT_MOD_U64.into(),
        true,
    );
    assert_eq!(rolling, non_rolling);

    match rolling {
        None => unreachable!(),
        Some((start, count)) => {
            let substring: String = resume[start..start + len]
                .iter()
                .map(|c| char::from_u32(*c).unwrap())
                .collect();
            println!("Most common substring: \n\tCount: {count}\n\tSubstring: {substring}",)
        }
    }
    println!("{:?}", rolling);
}
