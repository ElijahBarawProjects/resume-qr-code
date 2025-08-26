def rolling_hash(
    s: str, window_size: int, base: int = 256, mod: int = 10**9 + 7
):
    """
    Calculates the rolling hash values of all substrings of length window_size in string s.
    Uses the polynomial rolling hash algorithm with base and mod as constants.

    :param s: the input string
    :param window_size: the size of the rolling window
    :param base: the base for the polynomial hash function
    :param mod: the modulus for the polynomial hash function
    :return: a list of hash values of all substrings of length window_size in s
    """
    n = len(s)
    power = [1] * (n + 1)

    # Precompute the powers of the base modulo the mod
    for i in range(1, n + 1):
        power[i] = (power[i - 1] * base) % mod

    # Compute the hash value of the first window
    current_hash = 0
    for i in range(window_size):
        current_hash = (current_hash * base + ord(s[i])) % mod

    yield (current_hash, 0)

    # Compute the hash values of the rest of the substrings
    for i in range(1, n - window_size + 1):

        # Remove the contribution of the first character in the window
        current_hash = (
            current_hash - power[window_size - 1] * ord(s[i - 1]) % mod + mod
        ) % mod

        # Shift the window by one character and add the new character to the hash
        current_hash = (current_hash * base + ord(s[i + window_size - 1])) % mod

        yield (current_hash, i)


def test_rolling_hash_basic():
    """Basic test to verify rolling hash correctness"""
    text = "abcdefg"
    window_size = 3
    base, mod = 256, 10**9 + 7
    
    # Get rolling hash results
    rolling_results = list(rolling_hash(text, window_size))
    
    # Calculate expected results manually
    expected = []
    for i in range(len(text) - window_size + 1):
        substring = text[i:i + window_size]
        hash_val = 0
        for char in substring:
            hash_val = (hash_val * base + ord(char)) % mod
        expected.append((hash_val, i))
    
    # Compare
    print("Rolling hash test:")
    all_correct = True
    for (hash_val, start), (exp_hash, exp_start) in zip(rolling_results, expected):
        substring = text[start:start + window_size]
        match = hash_val == exp_hash and start == exp_start
        symbol = "✓" if match else "✗"
        print(f"{symbol} '{substring}' at {start}: {hash_val}")
        if not match:
            all_correct = False
    
    return all_correct


if __name__ == "__main__":
    test_rolling_hash_basic()
