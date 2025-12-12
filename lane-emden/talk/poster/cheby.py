# mpmath.chebyfit(ctx, f, interval, N, error=False)

from mpmath import mp, chebyfit, polyval
from math import sin, pi

mp.pretty = True  # this will make "mpf('42.0')" display as "42.0"
poly, err = chebyfit(sin, [0, pi / 4], 10, error=True)

print(polyval(poly, 0.7))  #  c_n x^n + \ldots + c_2 x^2 + c_1 x + c_0
# 0.644217687237708
print(err)
# 1.95187454392545e-14
