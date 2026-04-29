#!/usr/bin/env python
"""
This script solves the Lane-Emden equation using the functions from
adomian.py and operator.py.
"""

from adomian import adomian
from inv_operator import inverse_operator
from sympy import Symbol, simplify, pprint
from sympy.abc import xi, u


def solve_lane_emden(non_linear_func, initial_condition, num_terms):
    """
    Solves the Lane-Emden equation: L(u) + N(u) = 0
    u_0 = initial_condition
    u_{k+1} = -L^(-1)(A_k)
    """
    print(f"--- Solving Lane-Emden for N(u) = {non_linear_func} ---")

    # Get symbolic Adomian polynomials A_0, A_1, ..., A_{num_terms-2}
    # We need num_terms - 1 polynomials to find num_terms total components (u_0 to u_{n-1})
    symbolic_A = adomian(non_linear_func, u, num_terms - 1)

    us = [initial_condition]

    for k in range(num_terms - 1):
        # Get the k-th Adomian polynomial
        A_k_sym = symbolic_A[k]

        # Substitute the values of u_0, u_1, ..., u_k calculated so far
        subs_map = {Symbol(f"u_{i}"): us[i] for i in range(k + 1)}
        A_k_val = A_k_sym.subs(subs_map)

        # Calculate the next component: u_{k+1} = -L^(-1)(A_k)
        # Note: inverse_operator already has a print decorator
        next_u = -inverse_operator(A_k_val, xi)
        us.append(simplify(next_u))

    approx_solution = sum(us)
    print("--- Final Approximate Solution ---")
    pprint(approx_solution)
    return approx_solution


if __name__ == "__main__":
    # Example: N(u) = u^5 (Lane-Emden equation for n=5)
    # L(u) + u^5 = 0, u(0) = 1
    solve_lane_emden(u**5, 1, 4)
