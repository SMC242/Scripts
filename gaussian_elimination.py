"""
# Objectives
- Learn how to use Gaussian Elimination by implementing it in Python 

# Requirements
- Implement all maths operations myself
- The code should be as close to my thought process as possible
- Support a 3 equation system initially and extend to support n equations
"""

from typing import Tuple, List, NamedTuple, Iterable
from functools import reduce
# Should be in the format: {x coefficient} {y coefficient} {z coefficient} {solution}
EquationCoefficients = Tuple[int, int, int, int]
SystemType = List[EquationCoefficients]


class PrimeFactor(NamedTuple):
    factor: int
    exponent: int


PrimeFactors = List[PrimeFactor]
PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23]

# I will initially assume 3 equations.
# >3 equations will be supported later
SYSTEM: SystemType = [
    (-7, -3, 3, 12),
    (2, 2, 2, 0),
    (-1, -4, 3, -9),
]


def is_int(to_check) -> bool:
    return isinstance(to_check, int)


def is_prime(x: int) -> bool:
    """Check if x is a prime number."""
    # It is not prime if divisible by a prime > 1
    return not any(map(lambda p: is_int(x / p), PRIMES))


def prime_factorise(x: int) -> PrimeFactors:
    """
    Get the numbers and their powers whose product is `x`.
    E.G x = 72 -> 2^2 * 3^2
    NOTE: returns [[1, 1], [x, 1]] if `x` is prime.
    """
    factors: PrimeFactors = [[1, 1]]
    if is_prime(x):
        return factors + [x, 1]

    for prime in PRIMES:
        # Check if cleanly divisible by the prime
        if not is_int(x / prime):
            continue
        # Get the highest exponent of the prime that cleanly divides x
        exponent = 2
        while is_int(x / prime ** exponent):
            exponent += 1
        else:
            factors.append([prime, exponent])
    return factors


def lcm(a: int, b: int) -> int:
    """Get the lowest common multiple."""
    def compare_exponents(a_factors: PrimeFactors, b_factors: PrimeFactors) -> Iterable[PrimeFactor]:
        """Get the highest exponent of each prime."""
        # transform to dict
        factors = {}
        for factor in a_factors + b_factors:
            if factor.factor not in factors or factor.exponent > factors[factor.factor]:
                factors.update({factor.factor: factor.exponent})
        # convert back to PrimeFactor form
        return map(lambda k, v: PrimeFactor(k, v), factors.items())

    v = prime_factorise(a)
    highest_factors = compare_exponents(prime_factorise(b))
    return reduce(lambda acc, factor: acc * factor.factor ** factor.exponent, highest_factors)


print(lcm(25, 6))
