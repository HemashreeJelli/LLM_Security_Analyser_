"""
Luhn & Verhoeff checksum algorithms for credit card and Aadhaar validation.
Pure Python stdlib, zero dependencies.

Design decisions:
  - Luhn (Mod-10) validates credit card numbers (Visa, Mastercard, Amex).
    Without it, any 16-digit number (product SKUs, serial numbers) would
    be a false positive.
  - Verhoeff validates Indian Aadhaar numbers (12 digits). It uses
    Dihedral group D_5 multiplication and permutation tables, catching
    single-digit errors AND adjacent transposition errors that Luhn misses.
  - Both are pure functions — no state, no I/O, instant execution.
"""


# ---------------------------------------------------------------------------
# Luhn Algorithm (Credit Cards — ISO/IEC 7812-1)
# ---------------------------------------------------------------------------

def is_valid_luhn(card_number_str: str) -> bool:
    """
    Validate a credit card number using the Luhn Algorithm (Modulus 10).

    Steps:
      1. Strip non-digit characters.
      2. Reverse the digit sequence.
      3. Double every second digit; if result > 9, subtract 9.
      4. Sum all digits. Valid if total % 10 == 0.

    Args:
        card_number_str: String of digits (spaces/dashes stripped internally).

    Returns:
        True if valid credit card checksum.
    """
    digits = [int(c) for c in card_number_str if c.isdigit()]
    if len(digits) < 13 or len(digits) > 19:
        return False

    checksum = 0
    for i, digit in enumerate(reversed(digits)):
        if i % 2 == 1:
            doubled = digit * 2
            checksum += doubled - 9 if doubled > 9 else doubled
        else:
            checksum += digit

    return checksum % 10 == 0


# ---------------------------------------------------------------------------
# Verhoeff Algorithm (Indian Aadhaar — Dihedral Group D_5)
# ---------------------------------------------------------------------------

# Multiplication table for D_5 group
_D = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
    [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
    [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
    [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
]

# Permutation table
_P = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
    [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
    [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
    [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
]


def is_valid_verhoeff(num_str: str) -> bool:
    """
    Validate a 12-digit Indian Aadhaar number using the Verhoeff checksum.

    The Verhoeff algorithm uses Dihedral group D_5 operations, catching
    ALL single-digit substitution errors and ALL adjacent transposition
    errors (which Luhn cannot).

    UIDAI rules applied:
      - Must be exactly 12 digits.
      - First digit must be 2-9 (never 0 or 1).

    Args:
        num_str: String of 12 digits (spaces stripped internally).

    Returns:
        True if valid Aadhaar Verhoeff checksum.
    """
    digits = [int(c) for c in num_str if c.isdigit()]
    if len(digits) != 12:
        return False
    if digits[0] < 2:
        return False

    c = 0
    for i, item in enumerate(reversed(digits)):
        c = _D[c][_P[i % 8][item]]

    return c == 0
