#!/usr/bin/env python
"""
This script calculates symbolic Adomian Polynomials for a given non-linear function.
It reproduces the examples found in example.tex.
"""

from functools import wraps
from sympy import diff as D
from sympy.core import Symbol, Expr
from sympy.functions.combinatorial.factorials import factorial
from sympy.simplify.simplify import simplify


def print_adomian(func):
    """
    Decorator to print Adomian Polynomials after calculation.
    """

    @wraps(func)
    def wrapper(f_expr, variables, num_terms):
        polynomials = func(f_expr, variables, num_terms)
        print(f"--- Adomian Polynomials for N = {f_expr} ---\n")
        for k, a_k in enumerate(polynomials):
            print(f"A_{k} = {a_k}")
        print("\n")
        return polynomials

    return wrapper


@print_adomian
def adomian(f_expr: Expr, variables, num_terms: int):
    """
    Calculates the first 'num_terms' Adomian Polynomials for the expression f_expr.
    'variables' can be a single Symbol or a list of Symbols.

    A_k = (1/k!) * d^k/d(lambda)^k [ f(sum_{i=0}^k u_i * lambda^i, ...) ] | lambda=0
    """
    if isinstance(variables, Symbol):
        variables = [variables]

    lamda = Symbol("lambda")
    subs_map = {}

    for var in variables:
        name = var.name
        # Split base name from prime suffix (e.g., "u''" -> "u", "''")
        base = name.rstrip("'")
        suffix = name[len(base) :]

        # Create symbolic components: base_i + suffix
        components = [Symbol(f"{base}_{i}{suffix}") for i in range(num_terms)]
        v_lamda = sum(components[i] * lamda**i for i in range(num_terms))
        subs_map[var] = v_lamda

    # Substitute into the non-linear expression
    f_lamda = f_expr.subs(subs_map)

    polynomials = []
    for k in range(num_terms):
        # Calculate the k-th derivative with respect to lambda
        derivative = D(f_lamda, lamda, k)
        # Evaluate at lambda = 0 and divide by k!
        a_k = simplify(derivative.subs(lamda, 0) / factorial(k))
        polynomials.append(a_k)

    return polynomials


if __name__ == "__main__":
    u = Symbol("u")
    u_prime = Symbol("u'")
    # Reproduce the example N(u) = u^2
    adomian(u**2, u, 6)
    # Reproduce the example N(u) = u * u'
    adomian(u * u_prime, [u, u_prime], 5)
