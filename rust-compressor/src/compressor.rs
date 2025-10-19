use std::{
    fmt::Write,
    hash::{BuildHasher, Hasher},
};

use crate::{
    interface::WindowHasher,
    optimal_substring::{best_substring_and_longest_gt_one, BestSubstringRes},
    rolling_hash::{HasherConfig, DEFAULT_BASE, DEFAULT_MOD_U64},
};

// Identity hasher for u64 (no-op hashing since rolling hash values are already well-distributed)
#[derive(Default)]
struct U64IdentityHasher {
    hash: u64,
}

impl Hasher for U64IdentityHasher {
    fn write(&mut self, _bytes: &[u8]) {
        unimplemented!("U64IdentityHasher only supports write_u64")
    }

    fn write_u64(&mut self, i: u64) {
        // self.hash = i;
        self.hash = i ^ (i >> 32) ^ i.wrapping_mul(0x517cc1b727220a95);
    }

    fn finish(&self) -> u64 {
        self.hash
    }
}

#[derive(Default, Clone, Copy)]
struct BuildU64IdentityHasher;

impl BuildHasher for BuildU64IdentityHasher {
    type Hasher = U64IdentityHasher;

    fn build_hasher(&self) -> Self::Hasher {
        U64IdentityHasher::default()
    }
}

pub struct CompressorConfig {
    pub start: char,
    pub end: char,
    pub special: char,
    pub disallowed: Vec<char>,
    pub wordlist_delim: char,
    min_len: usize,
    max_len: usize,
    nproc: usize,
    allow_overlap: bool,
    check: bool,
}

impl Default for CompressorConfig {
    fn default() -> Self {
        const START: char = '"'; // inclusive
        const END: char = SPECIAL; // exclusive
        const SPECIAL: char = '~';
        const DISALLOWED: [char; 2] = ['`', '\\'];
        const WORDLIST_DELIM: char = '|';
        const MIN_LEN: usize = 3;
        const MAX_LEN: usize = 50;
        const NPROC: Option<usize> = None;
        const ALLOW_OVERLAP: bool = true;
        const CHECK: bool = false;

        Self::new(
            START,
            END,
            SPECIAL,
            DISALLOWED.into_iter().collect(),
            WORDLIST_DELIM,
            MIN_LEN,
            MAX_LEN,
            NPROC,
            ALLOW_OVERLAP,
            CHECK,
        )
        .expect("The specified defaults should produce a valid config")
    }
}

pub struct CompressionIntermediate {
    pub num_replacements: usize,
    pub last_symbol: u64,
    pub replacement_list: ReplacementList,
    pub compressed: String,
}

impl CompressionIntermediate {
    pub fn format_to_string(&self) -> String {
        match self {
            CompressionIntermediate {
                num_replacements,
                last_symbol,
                replacement_list,
                compressed,
            } => 
                format!("<meta charset=\"utf-8\"><script>onload=()=>{{let w={replacement_list}.split(\"|\"),h=`{compressed}`;for(i=0;i<{num_replacements};)h=h.replaceAll(\"~\"+String.fromCharCode({last_symbol}-i),w[i++]);document.body.innerHTML=h}};</script>")
        }
    }
}

impl CompressorConfig {
    pub fn new(
        start: char,
        end: char,
        special: char,
        disallowed: Vec<char>,
        wordlist_delim: char,
        min_len: usize,
        max_len: usize,
        nproc: Option<usize>,
        allow_overlap: bool,
        check: bool,
    ) -> Result<Self, &'static str> {
        if start >= end {
            return Err("No chars for substitution");
        }

        Ok(Self {
            start,
            end,
            special,
            disallowed,
            wordlist_delim,
            min_len,
            max_len,
            nproc: nproc.unwrap_or_else(num_cpus::get),
            allow_overlap,
            check,
        })
    }

    pub fn greedy(&self, text: String) -> Result<(Vec<Replacement>, String), &'static str> {
        let mut chosen: Vec<Replacement> = vec![];
        let key_symbols = self.start..self.end;
        let mut prev_was_disallowed = false;

        type TModulus = crate::modular::CustomU64Mod;

        let hasher: HasherConfig<_, TModulus> =
            HasherConfig::new(DEFAULT_BASE.into(), DEFAULT_MOD_U64.into())?;
        let mut chars: Vec<_> = text.chars().collect();

        let mut max_len = self.max_len;

        for symbol in key_symbols {
            let key = String::from_iter([self.special, symbol]);

            if self.disallowed.contains(&symbol) {
                prev_was_disallowed = true;
                chosen.push(Replacement {
                    substring: None, // ie, replacement does nothing
                    key,
                    symbol,
                    count: 0,
                    // design decision: make this one's savings 0 (next will pay for this)
                    savings: 0,
                });
                continue;
            }

            type TH = u64;
            let (best, longest_gt_one) = best_substring_and_longest_gt_one::<
                TH,
                usize,
                HasherConfig<TH, TModulus>,
                BuildU64IdentityHasher,
            >(
                &chars,
                self.wordlist_delim,
                prev_was_disallowed,
                &key,
                self.allow_overlap,
                self.min_len,
                max_len,
                &hasher,
                self.check,
                self.nproc,
            );

            if let Some(upper_len) = longest_gt_one {
                // WTS:
                //      If we use this new upper bound + 2, then we capture the longest
                //      substring after replacement
                // Proof:
                //      We are performing a substitution which will only reduce the length
                //      of any substring which contains replaced substring. So, the only
                //      way which a new, longer repeated substring could be created is
                //      if it is from changes to the edges. This can only be one char
                //      from each edge, as if it included both chars from any edge,
                //      then the length would be sorter as a longer substring was
                //      substituted for a shorter one. So, we have an increase in the
                //      length of the longest repeated substring of at most two.
                max_len = self.max_len.min(upper_len + 2);
            }

            match best {
                Some(BestSubstringRes {
                    score,
                    count,
                    substring,
                }) => {
                    if score > 0 && count > 1 {
                        let pat: Vec<_> = substring.chars().collect();
                        // chars = replace_in_chars_hashing(&chars, &pat, &[SPECIAL, symbol], &hasher);
                        chars = replace_in_chars(&chars, &pat, &[self.special, symbol]);

                        chosen.push(Replacement {
                            substring: Some(substring),
                            key,
                            symbol,
                            savings: score,
                            count: count,
                        });
                    } else {
                        break;
                    }
                }

                None => break,
            };

            prev_was_disallowed = false;
        }

        Ok((chosen, String::from_iter(chars)))
    }

    pub fn prep_compress(&self, text: String) -> Result<CompressionIntermediate, &'static str> {
        let (replacements, compressed) = match self.greedy(text) {
            Ok(x) => x,
            Err(e) => return Err(e),
        };

        let num_replacements = replacements.len();
        let last_symbol = (replacements.last().unwrap().symbol) as u64;

        let replacement_list = ReplacementList {
            replacements,
            delim: '|',
        };

        Ok(CompressionIntermediate {
            num_replacements,
            last_symbol,
            replacement_list,
            compressed,
        })
    }

    pub fn compress(&self, text: String) -> Result<String, &'static str> {
        match self.prep_compress(text) {
            Ok(compr_intermediate) => Ok(compr_intermediate.format_to_string()),
            Err(e) => Err(e),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Replacement {
    substring: Option<String>,
    key: String,
    symbol: char,
    savings: usize,
    count: usize,
}

#[allow(dead_code)]
fn replace_in_chars(chars: &[char], pattern: &[char], replacement: &[char]) -> Vec<char> {
    let mut result = Vec::with_capacity(chars.len());
    let mut i = 0;

    while i < chars.len() {
        // Check if pattern matches at current position
        if i + pattern.len() <= chars.len() && chars[i..i + pattern.len()] == pattern[..] {
            // Match found, add replacement
            result.extend_from_slice(replacement);
            i += pattern.len();
        } else {
            // No match, add current char
            result.push(chars[i]);
            i += 1;
        }
    }

    result
}

/// Replace all occurrences of `pattern` with `replacement` in a Vec<char>
/// Uses rolling hash for O(|chars| + |pattern|) complexity instead of O(|chars| * |pattern|)
#[allow(dead_code)]
fn replace_in_chars_hashing(
    chars: &[char],
    pattern: &[char],
    replacement: &[char],
    hasher: &HasherConfig<u128, crate::modular::Mod<u128>>,
) -> Vec<char> {
    if pattern.is_empty() {
        return chars.to_vec();
    }

    let pattern_len = pattern.len();
    let pattern_hash = hasher.hash(pattern);

    let mut result = Vec::with_capacity(chars.len());
    let mut i = 0; // Current position in input

    for (hash, start) in hasher.sliding_hash(chars, pattern_len) {
        if start < i {
            continue;
        }

        while i < start {
            result.push(chars[i]);
            i += 1;
        }

        if hash == pattern_hash {
            result.extend_from_slice(replacement);
            i = start + pattern_len;
        }
    }

    while i < chars.len() {
        result.push(chars[i]);
        i += 1;
    }

    result
}

#[cfg(test)]
fn decompress(mut text: String, choices: Vec<Replacement>) -> String {
    for Replacement { substring, key, .. } in choices.iter().rev() {
        match substring {
            None => continue,
            Some(substring) => text = text.replace(key, &substring),
        }
    }

    text
}

pub struct ReplacementList {
    replacements: Vec<Replacement>,
    delim: char,
}

impl std::fmt::Display for ReplacementList {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_char('`')?;
        let len = self.replacements.len();
        for (ind, Replacement { substring, .. }) in self.replacements.iter().rev().enumerate() {
            if let Some(substring) = substring {
                f.write_str(&substring)?;
            }
            if ind != (len - 1) {
                f.write_char(self.delim)?;
            }
        }
        f.write_char('`')?;

        Ok(())
    }
}

#[cfg(test)]
mod test {
    use super::CompressorConfig;
    use super::{decompress, Replacement};

    const RESUME_NO_DOCTYPE: &'static str = "<center><p><b><a href=\"https://www.elijahbaraw.com/\" style=\"color:#00f\"><u>Elijah Baraw</u></a></b><br><a href=\"mailto:ebaraw@andrew.cmu.edu?subject=RE:%20Your%20Resume\" style=\"color:#00f\"><u>ebaraw@andrew.cmu.edu</u></a> (203) 731-9535 Pittsburgh, PA <a href=\"https://github.com/elijah-bae-raw\" style=\"color:#00f\"><u>github.com/elijah-bae-raw</u></a></center><hr><h1>Education</h1><p><b>Carnegie Mellon University, School of Computer Science</b> <b>Aug 2021 - May 2025</b><br><em>Bachelor of Science in Computer Science. Concentration in Computer Systems</em><br>GPA: 3.97. Relevant courses: Machine Learning, Cloud Computing, Distributed Systems, Data Structures and<br>Algorithms, Functional Programming, Systems, Parallel Algorithms, Linear Algebra, Differential Equations, IDL<h1>Technical Skills</h1><p><b>Languages:</b> C, Python, Go, SQL, Java, HCL<br><b>Technologies:</b> NumPy, PyTorch, Pandas, OpenCV, Linux, Sockets, Git, AWS, GCP, Azure, K8s, Docker<br><b>Topics:</b> Data Structures and Algorithms, Object Oriented Programing, Functional Programming, Systems,<br>Consensus Algorithms, Actor Model, Network Protocols, TCP, Cryptographic Algorithms, Machine Learning<h1>Experience</h1><p><b>Center for Atmospheric Particle Studies</b> <b>Pittsburgh, PA</b><br><b>Research Assistant</b> <b>May 2022 - Aug 2022</b><ul><li>Developed a low cost device for measuring PM2.5 air pollutants collected on a foam tape over several months<li>Implemented an image processing pipeline as a cheaper alternative to traditional particle detection machines<li>Integrated a system of Arduino and Python scripts to control stepper motors based on continuous CV input.</ul><h1>Projects</h1><p><b>Fontify (Solo Python Project)</b> <em>PIL, Image Processing, De-Noising, Computer Vision</em> <b>Jan 2022 - July 2022</b><ul><li>Created image processing software in Python to convert handwritten letters into a personalized bitmap font.<li>Utilized image processing, edge-detection, noise reduction algorithms to detect pencil writing on paper.</ul><p><b>Concurrent Proxy Server (C)</b> <em>Git, HTTP, Sockets</em> <b>July 2023</b><ul><li>Developed a proxy server in C using p_threads and fork to handle requests concurrently, anonymize traffic and cache responses. Utilized Unix sockets and a bounded cache following an LRU eviction policy.</ul><p><b>Distributed Backend (Golang)</b> <em>Replication, Actor Model, Mailbox/Message Passing</em> <b>Nov 2023</b><ul><li>Designed and executed a concurrent server to manage the state for a multiplayer game, accessible via API.<li>Handled client requests about and updates to the game state using RPCs and a message-passing model.<li>Implemented node launching and server groups, ensuring replication and enforcing consistency within groups.</ul><p><b>Poker-Bots Hackathon Dev Team</b> <em>GCP, K8s, GitHub Actions</em> <b>Mar 2024; Mar 2025</b><ul><li>Helped CMU Data Science Club run their first AI Poker-Bot competition, with $6,000 in prizes and 63 teams.<li>Used GitHub Actions to automatically build Docker images of user-submitted Python bots, allowing competitors to use custom dependencies and machine learning libraries of their choice, running containers on GCP.<li>Helped build the second iteration of the competition in 2025 using AWS ECS for bots, Lambda for matches</ul><p><b>x86 IA-32 Kernel from Scratch</b> <em>C, ASM, Simics</em> <b>Aug 2024 - Dec 2024</b><ul><li>Built a complete i386 kernel from scratch, solo, for CMU 15410, implementing preemptive multitasking<li>Engineered hardware interfaces, memory management, and I/O system for concurrent ELF binary execution</ul>";
    // reference generated by python script
    const REF_COMPRESSED_BODY: &'static str = "<c~Q~b$~8www.elijahbaraw.com/~#Elijah Baraw~@~Fa href=\"mailto:~2?sub~U=RE:%20Your%20Resume~#~2~@ (203) 731-9535 ~A ~8~/~#~/~@</c~Qer><hr>~VEduc~N>Carnegie Mell~^University~cchool of~.~7~(~d~Y1 - May~lFem>B~eelor~M~7~L~.~7. Conc~Qr~NL~.S~Hs</em>~9GPA: 3.97. Relevant courses: ~<, Cloud Comput~T, ~JS~H~E~= ~f~9~%~+ Parallel ~%L~gar Algebra, Diff~bQial Equ~NEIDL~VTechnical Skills~>Langu~_~mC, ~5, Go~cQL, Java, HCL~9~dTechnologie~mNumPy, PyTorch, P~fa~EOpenCV, Linux~c~Ot~EGit, AWS, G~nAzure, K8~ED~Or~9~dTopic~m~=~&~%Ob~U Ori~Q~1Program~T, ~+~9Consensus ~%~BNetwork~Wtocol~ET~nCryptographic ~%~<~VEx~hience~>C~Q~b3Atmospheric P~]Studi~R(~d~A~Fb>Research Assistant~(~dMay~q~Y2~Clow cost device~3measur~-PM2.5 air pollutants collec~u~^a foam tape over several m~ahs~:I~;1an~) pipel~g as a chea~h alternative~4tradi~'al p~]~Z~' m~ein~R:Integra~ua~iH~MArduino~&~5 scripts~4c~arol step~h motors bas~1~^c~ainuous CV~Lput.</ul>~VPro~Us~>F~aify (Solo ~5~W~U)~6PIL, Im~_~Wce~jT, De-Nois~T,~.Vision~*Jan~q~r2~\"Created~) soft~s ~5~4convert h~fwritten letters~Lto a ~hsonaliz~1bitmap f~a~k~ted~), edge-~Z~onoise reduc~' algorithms~4~Zt pencil writ~-~^pa~h.~?C~0~Wxy S~I (C)~6Git, HTTP~c~Ots~*~r3~Cproxy~iI~L C~[p_threads~&fork~4h~fle~Pc~0ly, anonymize traffic~&c~ee responses. ~~uUnix~iOts~&a bound~1c~ee fo~van LRU evic~' policy.~?~JBackend (Golang)~6Replica~o~BMailbox/Me~j_ Pa~jT~*Nov~,3~\"Designed~&~wua c~0~iI~4man~_ ~X~x3a~yplayer game, accessible via API~kH~fl~1cli~Q~Pabout~&updat~R4~Xgame ~x[RPCs~&a me~j_-pa~j-model~kI~;1node launch~T~&s~I~z~Eensur~-replic~N&enforc~-consistency within~zs.~?~Ss Hackath~^Dev Team~6G~nK8~E~D*Mar~,4; Mar~l\"~{CMU Data ~7 Club run ~|first AI ~S ~G~owith $6,000~L priz~R&63 teams~kUs~1~D4automatically ~}D~Or imag~RMuser-submit~u~5 bot~Ea~v~Gtors~4use custom dependenci~R&m~e~g learn~-librari~RM~|choice, runn~-c~aa~grs ~^GCP~k~{~}~Xsecond it~bNM~X~G~'~L~l[AWS ECS~3bot~ELambda~3m~p~R?x86 IA-32 K~KScr~p~6C, ASM~cimics~*~Y4 - Dec~,4~\"Built a complete i386 k~Kscr~p, solo,~3CMU 15410, i~;-preemptive~ytask~T~:Eng~g~b1hard~sterface~Ememory man~_m~Q,~&I/O~iH~3c~0 ELF binary ~w'</ul>";

    #[test]
    fn test_greedy_decompress() {
        let res = CompressorConfig::default()
            .greedy(RESUME_NO_DOCTYPE.to_string())
            .unwrap();
        let (choices, compressed) = res;
        assert_eq!(compressed, REF_COMPRESSED_BODY);

        let _choices = choices.clone();
        assert_eq!(
            RESUME_NO_DOCTYPE.to_string(),
            decompress(compressed, choices)
        );

        let choices = _choices;
        let expected_choices = get_expected();

        let comp_len = choices.len();
        let expected_len = expected_choices.len();

        for (expected_choice, choice) in expected_choices.into_iter().zip(choices) {
            assert_eq!(expected_choice, choice)
        }
        assert_eq!(expected_len, comp_len);
    }

    fn get_expected() -> Vec<Replacement> {
        let expected = vec![
            Replacement {
                substring: Some("</b><ul><li>".to_string()),
                key: "~\"".to_string(),
                symbol: '"',
                savings: 47,
                count: 6,
            },
            Replacement {
                substring: Some("\" style=\"color:#00f\"><u>".to_string()),
                key: "~#".to_string(),
                symbol: '#',
                savings: 41,
                count: 3,
            },
            Replacement {
                substring: Some("><p><b>".to_string()),
                key: "~$".to_string(),
                symbol: '$',
                savings: 37,
                count: 9,
            },
            Replacement {
                substring: Some("Algorithms, ".to_string()),
                key: "~%".to_string(),
                symbol: '%',
                savings: 37,
                count: 5,
            },
            Replacement {
                substring: Some(" and ".to_string()),
                key: "~&".to_string(),
                symbol: '&',
                savings: 33,
                count: 13,
            },
            Replacement {
                substring: Some("tion".to_string()),
                key: "~'".to_string(),
                symbol: '\'',
                savings: 31,
                count: 18,
            },
            Replacement {
                substring: Some("</b> ".to_string()),
                key: "~(".to_string(),
                symbol: '(',
                savings: 27,
                count: 11,
            },
            Replacement {
                substring: Some(" image processing".to_string()),
                key: "~)".to_string(),
                symbol: ')',
                savings: 27,
                count: 3,
            },
            Replacement {
                substring: Some("</em> <b>".to_string()),
                key: "~*".to_string(),
                symbol: '*',
                savings: 25,
                count: 5,
            },
            Replacement {
                substring: Some("Func~'al Programming, Systems,".to_string()),
                key: "~+".to_string(),
                symbol: '+',
                savings: 25,
                count: 2,
            },
            Replacement {
                substring: Some(" 202".to_string()),
                key: "~,".to_string(),
                symbol: ',',
                savings: 21,
                count: 13,
            },
            Replacement {
                substring: Some("ing ".to_string()),
                key: "~-".to_string(),
                symbol: '-',
                savings: 21,
                count: 13,
            },
            Replacement {
                substring: Some(" Computer ".to_string()),
                key: "~.".to_string(),
                symbol: '.',
                savings: 21,
                count: 4,
            },
            Replacement {
                substring: Some("github.com/elijah-bae-raw".to_string()),
                key: "~/".to_string(),
                symbol: '/',
                savings: 20,
                count: 2,
            },
            Replacement {
                substring: Some("oncurrent".to_string()),
                key: "~0".to_string(),
                symbol: '0',
                savings: 18,
                count: 4,
            },
            Replacement {
                substring: Some("ed ".to_string()),
                key: "~1".to_string(),
                symbol: '1',
                savings: 16,
                count: 20,
            },
            Replacement {
                substring: Some("ebaraw@andrew.cmu.edu".to_string()),
                key: "~2".to_string(),
                symbol: '2',
                savings: 16,
                count: 2,
            },
            Replacement {
                substring: Some(" for ".to_string()),
                key: "~3".to_string(),
                symbol: '3',
                savings: 15,
                count: 7,
            },
            Replacement {
                substring: Some(" to ".to_string()),
                key: "~4".to_string(),
                symbol: '4',
                savings: 13,
                count: 9,
            },
            Replacement {
                substring: Some("Python".to_string()),
                key: "~5".to_string(),
                symbol: '5',
                savings: 13,
                count: 5,
            },
            Replacement {
                substring: Some("~(<em>".to_string()),
                key: "~6".to_string(),
                symbol: '6',
                savings: 13,
                count: 5,
            },
            Replacement {
                substring: Some("Science".to_string()),
                key: "~7".to_string(),
                symbol: '7',
                savings: 12,
                count: 4,
            },
            Replacement {
                substring: Some("<a href=\"https://".to_string()),
                key: "~8".to_string(),
                symbol: '8',
                savings: 12,
                count: 2,
            },
            Replacement {
                substring: Some("<br>".to_string()),
                key: "~9".to_string(),
                symbol: '9',
                savings: 11,
                count: 8,
            },
            Replacement {
                substring: Some("<li>".to_string()),
                key: "~:".to_string(),
                symbol: ':',
                savings: 11,
                count: 8,
            },
            Replacement {
                substring: Some("mplement~".to_string()),
                key: "~;".to_string(),
                symbol: ';',
                savings: 11,
                count: 3,
            },
            Replacement {
                substring: Some("Machine Learning".to_string()),
                key: "~<".to_string(),
                symbol: '<',
                savings: 11,
                count: 2,
            },
            Replacement {
                substring: Some("Data Structures".to_string()),
                key: "~=".to_string(),
                symbol: '=',
                savings: 10,
                count: 2,
            },
            Replacement {
                substring: Some("</h1~$".to_string()),
                key: "~>".to_string(),
                symbol: '>',
                savings: 9,
                count: 4,
            },
            Replacement {
                substring: Some("</ul~$".to_string()),
                key: "~?".to_string(),
                symbol: '?',
                savings: 9,
                count: 4,
            },
            Replacement {
                substring: Some("</u></a>".to_string()),
                key: "~@".to_string(),
                symbol: '@',
                savings: 9,
                count: 3,
            },
            Replacement {
                substring: Some("Pittsburgh, PA".to_string()),
                key: "~A".to_string(),
                symbol: 'A',
                savings: 9,
                count: 2,
            },
            Replacement {
                substring: Some("Actor Model, ".to_string()),
                key: "~B".to_string(),
                symbol: 'B',
                savings: 8,
                count: 2,
            },
            Replacement {
                substring: Some("~\"Develop~1a ".to_string()),
                key: "~C".to_string(),
                symbol: 'C',
                savings: 8,
                count: 2,
            },
            Replacement {
                substring: Some("GitHub Ac~'s~".to_string()),
                key: "~D".to_string(),
                symbol: 'D',
                savings: 8,
                count: 2,
            },
            Replacement {
                substring: Some("s, ".to_string()),
                key: "~E".to_string(),
                symbol: 'E',
                savings: 7,
                count: 11,
            },
            Replacement {
                substring: Some("</b>~9<".to_string()),
                key: "~F".to_string(),
                symbol: 'F',
                savings: 7,
                count: 3,
            },
            Replacement {
                substring: Some("competi".to_string()),
                key: "~G".to_string(),
                symbol: 'G',
                savings: 7,
                count: 3,
            },
            Replacement {
                substring: Some("ystem".to_string()),
                key: "~H".to_string(),
                symbol: 'H',
                savings: 6,
                count: 4,
            },
            Replacement {
                substring: Some("erver".to_string()),
                key: "~I".to_string(),
                symbol: 'I',
                savings: 6,
                count: 4,
            },
            Replacement {
                substring: Some("Distribut~1".to_string()),
                key: "~J".to_string(),
                symbol: 'J',
                savings: 6,
                count: 2,
            },
            Replacement {
                substring: Some("ernel from ".to_string()),
                key: "~K".to_string(),
                symbol: 'K',
                savings: 6,
                count: 2,
            },
            Replacement {
                substring: Some(" in".to_string()),
                key: "~L".to_string(),
                symbol: 'L',
                savings: 5,
                count: 9,
            },
            Replacement {
                substring: Some(" of ".to_string()),
                key: "~M".to_string(),
                symbol: 'M',
                savings: 5,
                count: 5,
            },
            Replacement {
                substring: Some("a~'~".to_string()),
                key: "~N".to_string(),
                symbol: 'N',
                savings: 5,
                count: 5,
            },
            Replacement {
                substring: Some("ocke".to_string()),
                key: "~O".to_string(),
                symbol: 'O',
                savings: 5,
                count: 5,
            },
            Replacement {
                substring: Some(" requests ".to_string()),
                key: "~P".to_string(),
                symbol: 'P',
                savings: 5,
                count: 2,
            },
            Replacement {
                substring: Some("ent".to_string()),
                key: "~Q".to_string(),
                symbol: 'Q',
                savings: 4,
                count: 8,
            },
            Replacement {
                substring: Some("es~".to_string()),
                key: "~R".to_string(),
                symbol: 'R',
                savings: 4,
                count: 8,
            },
            Replacement {
                substring: Some("Poker-Bot".to_string()),
                key: "~S".to_string(),
                symbol: 'S',
                savings: 4,
                count: 2,
            },
            Replacement {
                substring: Some("ing".to_string()),
                key: "~T".to_string(),
                symbol: 'T',
                savings: 3,
                count: 7,
            },
            Replacement {
                substring: Some("ject".to_string()),
                key: "~U".to_string(),
                symbol: 'U',
                savings: 3,
                count: 4,
            },
            Replacement {
                substring: Some("<h1>".to_string()),
                key: "~V".to_string(),
                symbol: 'V',
                savings: 3,
                count: 4,
            },
            Replacement {
                substring: Some(" Pro".to_string()),
                key: "~W".to_string(),
                symbol: 'W',
                savings: 3,
                count: 4,
            },
            Replacement {
                substring: Some("the ".to_string()),
                key: "~X".to_string(),
                symbol: 'X',
                savings: 3,
                count: 4,
            },
            Replacement {
                substring: Some("Aug~,".to_string()),
                key: "~Y".to_string(),
                symbol: 'Y',
                savings: 3,
                count: 3,
            },
            Replacement {
                substring: Some("detec".to_string()),
                key: "~Z".to_string(),
                symbol: 'Z',
                savings: 3,
                count: 3,
            },
            Replacement {
                substring: Some(" us~-".to_string()),
                key: "~[".to_string(),
                symbol: '[',
                savings: 3,
                count: 3,
            },
            Replacement {
                substring: None,
                key: "~\\".to_string(),
                symbol: '\\',
                savings: 0,
                count: 0,
            },
            Replacement {
                substring: Some("article ".to_string()),
                key: "~]".to_string(),
                symbol: ']',
                savings: 2,
                count: 2,
            },
            Replacement {
                substring: Some("on ".to_string()),
                key: "~^".to_string(),
                symbol: '^',
                savings: 2,
                count: 6,
            },
            Replacement {
                substring: Some("age".to_string()),
                key: "~_".to_string(),
                symbol: '_',
                savings: 2,
                count: 6,
            },
            Replacement {
                substring: None,
                key: "~`".to_string(),
                symbol: '`',
                savings: 0,
                count: 0,
            },
            Replacement {
                substring: Some("ont".to_string()),
                key: "~a".to_string(),
                symbol: 'a',
                savings: 1,
                count: 6,
            },
            Replacement {
                substring: Some("er~".to_string()),
                key: "~b".to_string(),
                symbol: 'b',
                savings: 1,
                count: 5,
            },
            Replacement {
                substring: Some(", S".to_string()),
                key: "~c".to_string(),
                symbol: 'c',
                savings: 1,
                count: 5,
            },
            Replacement {
                substring: Some("<b>".to_string()),
                key: "~d".to_string(),
                symbol: 'd',
                savings: 1,
                count: 5,
            },
            Replacement {
                substring: Some("ach".to_string()),
                key: "~e".to_string(),
                symbol: 'e',
                savings: 1,
                count: 5,
            },
            Replacement {
                substring: Some("and".to_string()),
                key: "~f".to_string(),
                symbol: 'f',
                savings: 1,
                count: 5,
            },
            Replacement {
                substring: Some("ine".to_string()),
                key: "~g".to_string(),
                symbol: 'g',
                savings: 1,
                count: 5,
            },
            Replacement {
                substring: Some("per".to_string()),
                key: "~h".to_string(),
                symbol: 'h',
                savings: 1,
                count: 5,
            },
            Replacement {
                substring: Some(" s~".to_string()),
                key: "~i".to_string(),
                symbol: 'i',
                savings: 1,
                count: 5,
            },
            Replacement {
                substring: Some("ss~".to_string()),
                key: "~j".to_string(),
                symbol: 'j',
                savings: 1,
                count: 5,
            },
            Replacement {
                substring: Some(".~:".to_string()),
                key: "~k".to_string(),
                symbol: 'k',
                savings: 1,
                count: 5,
            },
            Replacement {
                substring: Some("~,5~".to_string()),
                key: "~l".to_string(),
                symbol: 'l',
                savings: 1,
                count: 3,
            },
            Replacement {
                substring: Some("s:~(".to_string()),
                key: "~m".to_string(),
                symbol: 'm',
                savings: 1,
                count: 3,
            },
            Replacement {
                substring: Some("CP, ".to_string()),
                key: "~n".to_string(),
                symbol: 'n',
                savings: 1,
                count: 3,
            },
            Replacement {
                substring: Some("~', ".to_string()),
                key: "~o".to_string(),
                symbol: 'o',
                savings: 1,
                count: 3,
            },
            Replacement {
                substring: Some("atch".to_string()),
                key: "~p".to_string(),
                symbol: 'p',
                savings: 1,
                count: 3,
            },
            Replacement {
                substring: Some("~,2 - ".to_string()),
                key: "~q".to_string(),
                symbol: 'q',
                savings: 1,
                count: 2,
            },
            Replacement {
                substring: Some("July~,".to_string()),
                key: "~r".to_string(),
                symbol: 'r',
                savings: 1,
                count: 2,
            },
            Replacement {
                substring: Some("ware~L".to_string()),
                key: "~s".to_string(),
                symbol: 's',
                savings: 1,
                count: 2,
            },
            Replacement {
                substring: Some("Utiliz".to_string()),
                key: "~t".to_string(),
                symbol: 't',
                savings: 1,
                count: 2,
            },
            Replacement {
                substring: Some("t~1".to_string()),
                key: "~u".to_string(),
                symbol: 'u',
                savings: 1,
                count: 5,
            },
            Replacement {
                substring: Some("llow~-".to_string()),
                key: "~v".to_string(),
                symbol: 'v',
                savings: 1,
                count: 2,
            },
            Replacement {
                substring: Some("execu~".to_string()),
                key: "~w".to_string(),
                symbol: 'w',
                savings: 1,
                count: 2,
            },
            Replacement {
                substring: Some("state~".to_string()),
                key: "~x".to_string(),
                symbol: 'x',
                savings: 1,
                count: 2,
            },
            Replacement {
                substring: Some(" multi".to_string()),
                key: "~y".to_string(),
                symbol: 'y',
                savings: 1,
                count: 2,
            },
            Replacement {
                substring: Some(" group".to_string()),
                key: "~z".to_string(),
                symbol: 'z',
                savings: 1,
                count: 2,
            },
            Replacement {
                substring: Some("Help~1".to_string()),
                key: "~{".to_string(),
                symbol: '{',
                savings: 1,
                count: 2,
            },
            Replacement {
                substring: Some("their ".to_string()),
                key: "~|".to_string(),
                symbol: '|',
                savings: 1,
                count: 2,
            },
            Replacement {
                substring: Some("build ".to_string()),
                key: "~}".to_string(),
                symbol: '}',
                savings: 1,
                count: 2,
            },
        ];

        expected
    }

    const REF_COMPRESSED_HTML: &str = "<meta charset=\"utf-8\"><script>onload=()=>{let w=`build |their |Help~1| group| multi|state~|execu~|llow~-|t~1|Utiliz|ware~L|July~,|~,2 - |atch|~', |CP, |s:~(|~,5~|.~:|ss~| s~|per|ine|and|ach|<b>|, S|er~|ont||age|on |article || us~-|detec|Aug~,|the | Pro|<h1>|ject|ing|Poker-Bot|es~|ent| requests |ocke|a~'~| of | in|ernel from |Distribut~1|erver|ystem|competi|</b>~9<|s, |GitHub Ac~'s~|~\"Develop~1a |Actor Model, |Pittsburgh, PA|</u></a>|</ul~$|</h1~$|Data Structures|Machine Learning|mplement~|<li>|<br>|<a href=\"https://|Science|~(<em>|Python| to | for |ebaraw@andrew.cmu.edu|ed |oncurrent|github.com/elijah-bae-raw| Computer |ing | 202|Func~'al Programming, Systems,|</em> <b>| image processing|</b> |tion| and |Algorithms, |><p><b>|\" style=\"color:#00f\"><u>|</b><ul><li>`.split(\"|\"),h=`<c~Q~b$~8www.elijahbaraw.com/~#Elijah Baraw~@~Fa href=\"mailto:~2?sub~U=RE:%20Your%20Resume~#~2~@ (203) 731-9535 ~A ~8~/~#~/~@</c~Qer><hr>~VEduc~N>Carnegie Mell~^University~cchool of~.~7~(~d~Y1 - May~lFem>B~eelor~M~7~L~.~7. Conc~Qr~NL~.S~Hs</em>~9GPA: 3.97. Relevant courses: ~<, Cloud Comput~T, ~JS~H~E~= ~f~9~%~+ Parallel ~%L~gar Algebra, Diff~bQial Equ~NEIDL~VTechnical Skills~>Langu~_~mC, ~5, Go~cQL, Java, HCL~9~dTechnologie~mNumPy, PyTorch, P~fa~EOpenCV, Linux~c~Ot~EGit, AWS, G~nAzure, K8~ED~Or~9~dTopic~m~=~&~%Ob~U Ori~Q~1Program~T, ~+~9Consensus ~%~BNetwork~Wtocol~ET~nCryptographic ~%~<~VEx~hience~>C~Q~b3Atmospheric P~]Studi~R(~d~A~Fb>Research Assistant~(~dMay~q~Y2~Clow cost device~3measur~-PM2.5 air pollutants collec~u~^a foam tape over several m~ahs~:I~;1an~) pipel~g as a chea~h alternative~4tradi~'al p~]~Z~' m~ein~R:Integra~ua~iH~MArduino~&~5 scripts~4c~arol step~h motors bas~1~^c~ainuous CV~Lput.</ul>~VPro~Us~>F~aify (Solo ~5~W~U)~6PIL, Im~_~Wce~jT, De-Nois~T,~.Vision~*Jan~q~r2~\"Created~) soft~s ~5~4convert h~fwritten letters~Lto a ~hsonaliz~1bitmap f~a~k~ted~), edge-~Z~onoise reduc~' algorithms~4~Zt pencil writ~-~^pa~h.~?C~0~Wxy S~I (C)~6Git, HTTP~c~Ots~*~r3~Cproxy~iI~L C~[p_threads~&fork~4h~fle~Pc~0ly, anonymize traffic~&c~ee responses. ~~uUnix~iOts~&a bound~1c~ee fo~van LRU evic~' policy.~?~JBackend (Golang)~6Replica~o~BMailbox/Me~j_ Pa~jT~*Nov~,3~\"Designed~&~wua c~0~iI~4man~_ ~X~x3a~yplayer game, accessible via API~kH~fl~1cli~Q~Pabout~&updat~R4~Xgame ~x[RPCs~&a me~j_-pa~j-model~kI~;1node launch~T~&s~I~z~Eensur~-replic~N&enforc~-consistency within~zs.~?~Ss Hackath~^Dev Team~6G~nK8~E~D*Mar~,4; Mar~l\"~{CMU Data ~7 Club run ~|first AI ~S ~G~owith $6,000~L priz~R&63 teams~kUs~1~D4automatically ~}D~Or imag~RMuser-submit~u~5 bot~Ea~v~Gtors~4use custom dependenci~R&m~e~g learn~-librari~RM~|choice, runn~-c~aa~grs ~^GCP~k~{~}~Xsecond it~bNM~X~G~'~L~l[AWS ECS~3bot~ELambda~3m~p~R?x86 IA-32 K~KScr~p~6C, ASM~cimics~*~Y4 - Dec~,4~\"Built a complete i386 k~Kscr~p, solo,~3CMU 15410, i~;-preemptive~ytask~T~:Eng~g~b1hard~sterface~Ememory man~_m~Q,~&I/O~iH~3c~0 ELF binary ~w'</ul>`;for(i=0;i<92;)h=h.replaceAll(\"~\"+String.fromCharCode(125-i),w[i++]);document.body.innerHTML=h};</script>";

    #[test]
    fn test_compress() {
        let computed = CompressorConfig::default()
            .compress(RESUME_NO_DOCTYPE.to_string())
            .unwrap();
        let expected = REF_COMPRESSED_HTML.to_string();
        assert_eq!(computed, expected)
    }
}
