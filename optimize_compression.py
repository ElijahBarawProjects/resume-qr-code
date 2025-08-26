#!/usr/bin/env python3
import argparse
import sys
import re
from collections import defaultdict
from rolling_hash import rolling_hash
from dataclasses import dataclass

# SCRIPT_TEMPLATE = '<script>onload=()=>{{let w=`{word_string}`.split("|"),h={compressed},i=0;for(c of"{letters}")h=h.replaceAll("~"+c,w[i++]);document.body.innerHTML=h}};</script>'
# BASE_SCRIPT_LEN = len(
#     SCRIPT_TEMPLATE.format(word_string="", letters="", compressed="")
# )
SCRIPT_TEMPLATE = '<script>onload=()=>{{let w=`{word_string}`.split("|"),h={compressed};for(i=0;i<{num_iters};)h=h.replaceAll("~"+String.fromCharCode({max_char_code}-i),w[i++]);document.body.innerHTML=h}};</script>'
BASE_SCRIPT_LEN = len(
    SCRIPT_TEMPLATE.format(
        word_string="", compressed="", num_iters="", max_char_code=""
    )
)
ALLOW_OVERLAPS = False
CHECK_COLISSION = False
VERBOSE = True
LOWERCASE = "abcdefghijklmnopqrstuvwxyz"
UPPERCASE = LOWERCASE.upper()
NUMERIC = "0123456789"
ALPHANUMERIC = LOWERCASE + UPPERCASE + NUMERIC
# this range includes '`', which we can never use as that's our string encloser;
# so we will do a redundant substitution
PRINTABLE_UP_TO_BAR = "".join(
    [chr(i) for i in range(ord("0"), ord("}") + 1)]
)  # we don't want '|' to appear inside the words string, but we can use it as our last substitutor
WORDLIST_DELIM = "|"  # used to delimit substrings in the wordlist, which is used to map {KEY_PREFIX+sym => value}
KEY_PREFIX = "~"  # prepended to turn a symbol into a key
QUOTES = "`"
BACKSLASH = "\\"
# can't occur inside either word list or compressed
DISALLOWED = [QUOTES, BACKSLASH]

# class Compressor:
#     def __init__(self, quote_char: str, key_prefix: str, wordlist_delim: str):
#         self.quote_char = quote_char
#         self.key_prefix = key_prefix
#         self.wordlist_delim = wordlist_delim


class IdMaker:
    """
    Iterator which provides keys for substitution
    """

    def __init__(
        self,
        special="~",
        symbols=PRINTABLE_UP_TO_BAR,
    ):
        self.special = special
        self.symbols = symbols
        self.current = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.current == len(self.symbols):
            raise StopIteration

        sym = self.symbols[self.current]
        tmp = self.special + sym

        self.current += 1

        return tmp, sym


def extract_compressible_content(html):
    """Extract all HTML content that can be safely compressed"""
    # remove scripts
    html = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL)

    # compress everything after the 'meta' tag
    meta_match = re.search(r'<meta charset="utf-8">(.*)', html, re.DOTALL)
    return meta_match.group(1) if meta_match else html


def build_all_substring_counts(
    text: str, min_len=3, max_len=-1, allow_overlap=True, check_colissions=False
):
    """Get a dictionary mapping substring to count

    Args:
        text (str): text to build substring count of
        min_len (int, optional): minimum substring length. Defaults to 3.
        allow_overlap (bool, optional):
            Whether to allow overlapping substrings. Defaults to True.
            Much slower if need non-overlapping substrings

    Returns:
        dict[str, int]: mapping of substring to occurrence count
    """
    n = len(text)
    max_len = max_len if max_len > 0 else n

    hash_to_substring: dict[int, str] = {}
    hash_to_count = defaultdict(lambda: 0)

    # count substrings using rolling hash (with overlap and possibly with colissions)
    for length in range(min_len, max_len + 1):
        rh = rolling_hash(text, length, base=257, mod=2**61 - 1)
        for cur_hash, start in rh:
            hash_to_count[cur_hash] += 1
            if cur_hash not in hash_to_substring:
                hash_to_substring[cur_hash] = text[start : start + length]

            if not check_colissions:
                continue

            if hash_to_substring[cur_hash] != text[start : start + length]:
                print(
                    "COLISSION:",
                    text[start : start + length],
                    hash_to_substring[cur_hash],
                    sep="\n",
                )
            else:
                print("NO COLISSION")

    if allow_overlap:
        return {
            hash_to_substring[hash]: count
            for hash, count in hash_to_count.items()
            if count > 1
        }

    return {
        substring: count
        for substring in hash_to_substring.values()
        for count in [count_nonoverlapping_occurrences(text, substring)]
        if count > 1
    }


def count_nonoverlapping_occurrences(text: str, pattern: str):
    """Count non-overlapping occurrences of pattern in text (simulates sed behavior)"""
    count = 0
    start = 0
    while True:
        pos = text.find(pattern, start)
        if pos == -1:
            break
        count += 1
        start = pos + len(pattern)  # Move past this match (non-overlapping)
    return count


def get_savings(word: str, count: int, key: str, sym: str):
    """Computes the savings to be gained by substituting `word` with `key`

    Args:
        word (str): substring being replaced
        count (int): number of times substring occurs
        key (str): key replacing it
        sym (str): symbol added to symbol table

    Returns:
        int: savings in bytes
    """
    word_saving = count * (len(word) - len(key))
    word_saving -= len(word) + 1  # word, plus delimiter, inside of word list
    # word_saving -= len(sym)  # symbol in symbol list
    return word_saving


def generate_compression_script(
    words: list[str], symbols: list[str], compressed="document.body.innerHTML"
):
    """Generate optimized JavaScript compression"""
    word_list = words.copy()
    word_string = "|".join(reversed(word_list))
    letters = "".join(reversed(symbols))

    if VERBOSE:
        print("HERE", word_string, letters, sep="\n\n")
    script = SCRIPT_TEMPLATE.format(
        word_string=word_string,
        compressed=compressed,
        num_iters=len(words),
        max_char_code="0" if len(symbols) == 0 else ord(symbols[-1]),
        # word_string=word_string, letters=letters, compressed=compressed
    )

    return script, word_list


def escape_for_sed(s: str):
    """Escape special characters for sed regex"""
    # Characters that need escaping in sed: . * [ ] ^ $ \ / &
    special_chars = r"\/"
    escaped = s
    for char in special_chars:
        escaped = escaped.replace(char, "\\" + char)
    return escaped


@dataclass
class Replacement:
    substring: str
    key: str
    symbol: str
    count: int
    savings: int


def greedy(txt: str):
    id_maker = IdMaker()
    chosen: list[Replacement] = []
    prev_was_disallowed = False

    def _savings(tup):
        base_savings = tup[2]
        if prev_was_disallowed:
            base_savings -= 1
        return base_savings

    for key, sym in id_maker:
        if sym in DISALLOWED:
            prev_was_disallowed = True

            chosen.append(
                Replacement(
                    substring="",  # ie, replacement does nothing
                    key=key,
                    symbol=sym,
                    count=0,
                    # design decision: make this one's savings 0 (next will pay for this)
                    savings=0,
                )
            )

            continue

        counts = build_all_substring_counts(
            txt,
            min_len=3,
            max_len=50,
            allow_overlap=ALLOW_OVERLAPS,
            check_colissions=CHECK_COLISSION,
        )

        counts_valued = filter(
            lambda x: _savings(x) > 0,
            (
                (count, substring, get_savings(substring, count, key, sym))
                for substring, count in counts.items()
                if count > 1 and len(substring) > 2 and "|" not in substring
            ),
        )

        try:
            best_savings = max(counts_valued, key=_savings)
        except ValueError as e:
            if str(e) != "max() arg is an empty sequence":
                raise e
            break

        prev_was_disallowed = False

        count, chosen_substring, saving = best_savings
        chosen.append(
            Replacement(
                substring=chosen_substring,
                key=key,
                symbol=sym,
                count=count,
                savings=saving,
            )
        )
        txt = txt.replace(chosen_substring, key)
        if VERBOSE:
            print(chosen[-1])

    return chosen, txt


def make_parser():
    parser = argparse.ArgumentParser(
        description="Process input from a file or stdin and output to a file or stdout, "
        "abstracting input/output as file-like objects.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input-file", type=str, help="Path to the input HTML file."
    )
    input_group.add_argument(
        "--stdin", action="store_true", help="Read input from standard input."
    )

    output_group = parser.add_mutually_exclusive_group(required=True)
    output_group.add_argument(
        "--output-file", type=str, help="Path to the output HTML file."
    )
    output_group.add_argument(
        "--stdout", action="store_true", help="Write output to standard output."
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output showing replacement details.",
    )

    parser.add_argument(
        "--check-collisions",
        action="store_true",
        help="Enable hash collision detection and reporting.",
    )

    return parser


def get_io(args: argparse.Namespace):
    if args.stdin:
        input_stream = sys.stdin
    elif args.input_file:
        input_stream = open(args.input_file, "r", encoding="utf-8")
    else:
        raise ValueError("No input source specified")

    if args.stdout:
        output_stream = sys.stdout
    elif args.output_file:
        output_stream = open(args.output_file, "w", encoding="utf-8")
    else:
        raise ValueError("No output destination specified")

    return input_stream, output_stream


def main():
    global CHECK_COLISSION, VERBOSE

    parser = make_parser()
    args = parser.parse_args()

    input_stream, output_stream = get_io(args)
    CHECK_COLISSION = args.check_collisions
    VERBOSE = args.verbose
    try:
        html = input_stream.read()
        text = extract_compressible_content(html)
        if VERBOSE:
            print(
                f"Extracted {len(text)} characters for compression",
                file=sys.stderr,
            )

        optimal_words, compressed = greedy(txt=text)
        if len(optimal_words) > 0 and optimal_words[-1].symbol in DISALLOWED:
            optimal_words.pop()

        script, _word_list = generate_compression_script(
            words=[r.substring for r in optimal_words],
            symbols=[r.symbol for r in optimal_words],
            compressed=f"`{compressed}`",
        )

        total_savings = 0
        for replacement in optimal_words:
            if VERBOSE:
                print(
                    f"{replacement.key}: '{replacement.substring}' ({replacement.count}x) saves {replacement.savings} bytes",
                    file=sys.stderr,
                )
            total_savings += replacement.savings

        if total_savings > 0:
            total_savings += 1  # one fewer delimiter than calculated

        script_overhead = BASE_SCRIPT_LEN
        net_savings = total_savings - script_overhead

        if VERBOSE:
            print(
                f"\nScript overhead: {script_overhead} bytes", file=sys.stderr
            )
            print(f"Total text savings: {total_savings} bytes", file=sys.stderr)
            print(f"Net savings: {net_savings} bytes", file=sys.stderr)

        # Write compressed output
        if net_savings > 0:
            # Write the complete HTML with compression script
            output_stream.write(f'<meta charset="utf-8">{script}')
            if VERBOSE:
                print("\nCompressed HTML written to output", file=sys.stderr)
                print("\nSed commands for manual application:", file=sys.stderr)
                for replacement in optimal_words:
                    escaped_substring = escape_for_sed(replacement.substring)
                    print(
                        f"sed -i '' 's/{escaped_substring}/{replacement.key}/g' input.html",
                        file=sys.stderr,
                    )
        else:
            # No compression benefit, write original
            output_stream.write(html)
            if VERBOSE:
                print(
                    "No compression benefit, writing original HTML",
                    file=sys.stderr,
                )

    finally:
        # Close files if we opened them
        if args.input_file and input_stream != sys.stdin:
            input_stream.close()
        if args.output_file and output_stream != sys.stdout:
            output_stream.close()


if __name__ == "__main__":
    main()
