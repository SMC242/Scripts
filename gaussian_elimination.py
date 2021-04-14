"""
# Objectives
- Learn how to use Gaussian Elimination by implementing it in Python 

# Requirements
- Implement all maths operations myself
- The code should be as close to my thought process as possible
- Support a 3 equation system initially and extend to support n equations
"""

from typing import Counter, Tuple, List, NamedTuple, Iterable, Optional
from functools import reduce
# Should be in the format: [{x coefficient}, {y coefficient}, {z coefficient}]
Coefficients = Tuple[int, int, int]
# Should be in the format: [{x coefficient}, {y coefficient}, {z coefficient}, {solution}]
EquationCoefficients = Tuple[int, int, int, int]
SystemType = List[EquationCoefficients]

SYSTEM: SystemType = [
    (-7, -3, 3, 12),
    (2, 2, 2, 0),
    (-1, -4, 3, -9),
]


def is_int(to_check) -> bool:
    return isinstance(to_check, int)


class LCM:
    """Python has no module keyword and I don't want to split files for a little script. I've used a class instead"""
    class PrimeFactor(NamedTuple):
        factor: int
        exponent: int

    PrimeFactors = List[PrimeFactor]
    PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23]

    @staticmethod
    def is_prime(x: int) -> bool:
        """Check if x is a prime number."""
        # It is not prime if divisible by a prime > 1
        return not any(map(lambda p: is_int(x / p), LCM.PRIMES))

    @staticmethod
    def prime_factorise(x: int) -> PrimeFactors:
        """
        Get the numbers and their powers whose product is `x`.
        E.G x = 72 -> 2^2 * 3^2
        NOTE: returns [[1, 1], [x, 1]] if `x` is prime.
        """
        factors: LCM.PrimeFactors = [LCM.PrimeFactor(1, 1)]
        if LCM.is_prime(x):
            return factors + [LCM.PrimeFactor(x, 1)]

        for prime in LCM.PRIMES:
            # Check if cleanly divisible by the prime
            if not is_int(x / prime):
                continue
            # Get the highest exponent of the prime that cleanly divides x
            exponent = 2
            while is_int(x / prime ** exponent):
                exponent += 1
            else:
                factors.append(LCM.PrimeFactor(prime, exponent))
        return factors

    @staticmethod
    def lcm(a: int, b: int) -> int:
        """Get the lowest common multiple."""
        def compare_exponents(a_factors: LCM.PrimeFactors, b_factors: LCM.PrimeFactors) -> Iterable[LCM.PrimeFactor]:
            """Get the highest exponent of each prime."""
            # transform to dict
            factors = {}
            for factor in a_factors + b_factors:
                if factor.factor not in factors or factor.exponent > factors[factor.factor]:
                    factors.update({factor.factor: factor.exponent})
            # convert back to PrimeFactor form
            return map(lambda k, v: LCM.PrimeFactor(k, v), factors.items())

        highest_factors = compare_exponents(
            LCM.prime_factorise(a), LCM.prime_factorise(b))
        return reduce(lambda acc, factor: acc * factor.factor ** factor.exponent, highest_factors, 1)


class GE:
    """
    for system in systems:
        try coefficients / 2 ... 10
        if any divide cleanly:
            for other_system in [s for s in systems if s != system]:
                system - divisor * other_system
                break if any coefficients are now 0         
    """
    @staticmethod
    def remove_solution(equation: EquationCoefficients) -> Coefficients:
        """Remove the solution from an equation."""
        # Unneat solution to satisfy Pylance
        return equation[:3]

    @staticmethod
    def divides_system(eq1: EquationCoefficients, eq2: EquationCoefficients) -> Optional[int]:
        """Check if a coefficient of `eq2` divides into `eq1`."""
        def compare_coefficients(coefficients:  Tuple[int, int]) -> Optional[int]:
            """Get the multiple of c2 to divide c1."""
            # Avoid dividing by zero or dividing 0
            if any(map(lambda e: e == 0, coefficients)):
                return None
            divisor = (coefficients[0] / coefficients[1]) * -1
            divides_evenly = divisor % 1 == 0
            return int(divisor) if divides_evenly else None
        without_solutions = GE.remove_solution(list(zip(eq1, eq2)))
        divisors = map(compare_coefficients, without_solutions)
        # Check if there are any divisors
        no_nones = list(filter(lambda d: d is not None, divisors))
        if not no_nones:
            return None
        return max(no_nones)

    @staticmethod
    def multiply_row(row:  EquationCoefficients, multiplier: int) -> EquationCoefficients:
        """Multiply a row by a number."""
        return [multiplier * e for e in row]

    @staticmethod
    def add_rows(row1: EquationCoefficients, row2: EquationCoefficients) -> EquationCoefficients:
        """Add two rows together."""
        return [e1 + e2 for e1, e2 in zip(row1, row2)]

    @staticmethod
    def eliminate(system: SystemType) -> SystemType:
        """Eliminate at least one coefficient from one of the equations in `system`."""
        for equation in system:
            other_equations = [s for s in system if s != equation]
            for eq in other_equations:
                divisor = GE.divides_system(equation, eq)
                if divisor:
                    scaled_row = GE.multiply_row(eq, divisor)
                    new_row = GE.add_rows(equation, scaled_row)
                    system[0] = new_row
                    break
        return system

    @staticmethod
    def zeroes_in_equation(equation: EquationCoefficients) -> int:
        """Count the number of zeroes in an equation."""
        return reduce(lambda acc, e: acc + 1 if e ==
                      0 else acc, GE.remove_solution(equation), 0)

    @staticmethod
    def can_be_triangular(system: SystemType) -> bool:
        """Check if the system has enough zeroes to be converted to triangular form."""
        has_2_zero_row = False
        rows_with_zero = 0
        for row in system:
            zeroes = GE.zeroes_in_equation(row)
            rows_with_zero += 1 if zeroes else 0  # check for any zeroes in the row
            if zeroes == 3:
                raise ValueError("Row has too many zeroes")
            elif zeroes == 2:  # check if the row could be the bottom row
                has_2_zero_row = True
        return rows_with_zero >= 2 and has_2_zero_row

    @staticmethod
    def to_triangular(system: SystemType) -> SystemType:
        """Converts to triangular form."""
        copy = list(system)
        history: List[SystemType] = [copy]  # record all steps

        # get enough rows with zeroes
        ready_to_convert = False
        while not ready_to_convert:
            history.append(GE.eliminate(history[-1]))
            ready_to_convert = GE.can_be_triangular(history[-1])

        # Find the rows with zeroes
        last_system = history[-1]
        bottom = reduce(lambda acc, r: r if GE.zeroes_in_equation(r)
                        == 2 else acc, last_system, None)
        middle = reduce(lambda acc, r: r if GE.zeroes_in_equation(r)
                        == 1 else acc, last_system, None)
        if not all((bottom, middle)):
            raise RuntimeError("One of the reduces isn't working")
        top = list(filter(lambda r: r != bottom and r != middle),
                   last_system)[0]
        if 0 in top:
            raise RuntimeError("Top row search failed")
        return [top, middle, bottom]


def get_system(variable_rows: List[EquationCoefficients]) -> SystemType:
    return [*variable_rows, *[[1, 1, 1, 1]] * (3 - len(variable_rows))]


data = [[-2, 2, 1, 14], [3, -2, 1, -5], [-1, 1, -2, -8]]
print(GE.to_triangular(data))
