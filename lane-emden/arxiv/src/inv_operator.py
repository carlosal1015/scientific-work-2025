#!/usr/bin/env python
"""
This script calculates the inverse operator L^(-1) for the Lane-Emden equation.
"""

from functools import wraps
from sympy.abc import s, t
from sympy.core import Expr
from sympy.integrals import integrate
from sympy.simplify.simplify import simplify


def print_operator(func):
    """
    Decorator to print the result of the inverse operator.
    """

    @wraps(func)
    def wrapper(expr, variable):
        result = func(expr, variable)
        print(f"--- Inverse Operator for f({variable}) = {expr} ---")
        print(f"L^(-1)(f) = {result}\n")
        return result

    return wrapper


@print_operator
def inverse_operator(expr: Expr, variable: Expr) -> Expr:
    """
    Calculates the inverse operator L^(-1).

    L^(-1)(f) = Integral_0^x ( s^(-2) * Integral_0^s ( t^2 * f(t) ) dt ) ds
    """
    inner_integral = integrate(t**2 * expr.subs(variable, t), (t, 0, s))
    result = integrate(s ** (-2) * inner_integral, (s, 0, variable))
    return simplify(result)


if __name__ == "__main__":
    from sympy.abc import xi

    # Example usage
    inverse_operator(xi**2, xi)
