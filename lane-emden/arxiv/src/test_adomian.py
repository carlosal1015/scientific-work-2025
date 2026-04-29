import unittest
from sympy import Symbol
from adomian import adomian


class TestAdomian(unittest.TestCase):
    def test_u_squared(self):
        u = Symbol("u")
        num_terms = 4
        # We use __wrapped__ to call the original function without the decorator if we wanted to avoid print
        # but adomian returns the list anyway.
        polynomials = adomian.__wrapped__(u**2, u, num_terms)

        u_0 = Symbol("u_0")
        u_1 = Symbol("u_1")
        u_2 = Symbol("u_2")
        u_3 = Symbol("u_3")

        self.assertEqual(polynomials[0], u_0**2)
        self.assertEqual(polynomials[1], 2 * u_0 * u_1)
        self.assertEqual(polynomials[2], 2 * u_0 * u_2 + u_1**2)
        self.assertEqual(polynomials[3], 2 * u_0 * u_3 + 2 * u_1 * u_2)

    def test_u_u_prime(self):
        u = Symbol("u")
        u_prime = Symbol("u'")
        num_terms = 3
        polynomials = adomian.__wrapped__(u * u_prime, [u, u_prime], num_terms)

        u_0 = Symbol("u_0")
        u_1 = Symbol("u_1")
        u_2 = Symbol("u_2")
        up_0 = Symbol("u_0'")
        up_1 = Symbol("u_1'")
        up_2 = Symbol("u_2'")

        # A_0 = u_0 * u_0'
        self.assertEqual(polynomials[0], u_0 * up_0)
        # A_1 = u_0 * u_1' + u_1 * u_0'
        self.assertEqual(polynomials[1], u_0 * up_1 + u_1 * up_0)
        # A_2 = u_0 * u_2' + u_1 * u_1' + u_2 * u_0'
        self.assertEqual(polynomials[2], u_0 * up_2 + u_1 * up_1 + u_2 * up_0)


if __name__ == "__main__":
    unittest.main()
