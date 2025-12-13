import matplotlib.pyplot as plt
import numpy as np

a, b = (0, 1)
x = np.linspace(start=a, stop=b, num=300)
# y0 = 2 / (a + b) * np.ones_like(x)
# y1 = 8 * (a + b - x) / (a**2 + 6 * a * b + b**2)
# y2 = (6 * (3 * a**2 + 10 * a * b + 3 * b**2) - 48 * (a + b) * x + 32 * x**2) / (
#     (a + b) * (a**2 + 14 * a * b + b**2)
# )
# y3 = (
#     32 * (a**3 + 7 * a**2 * b + 7 * a * b**2 + b**3)
#     - (5 * a**2 + 14 * a * b + 5 * b**2) * x
#     + 8 * ((a + b) * x**2 - 4 * x**3)
# ) / (a**4 + 28 * a**3 * b + 70 * a**2 * b**2 + 28 * a * b**3 + b**4)
y = np.atan(np.pi * x / 2)
plt.plot(x, 1 / x, label="1/x")
# plt.plot(x, y0, label="y0")
# plt.plot(x, y1, label="y1")
# plt.plot(x, y2, label="y2")
# plt.plot(x, y3, label="y3")
plt.plot(x, y, label="y")
plt.legend()
plt.show()
