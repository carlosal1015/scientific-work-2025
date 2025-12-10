import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import numpy as np
from numpy.typing import ArrayLike


def lane_emden(t: float, u: ArrayLike, n: float) -> ArrayLike:
    y, z = u
    return [z, -2 * z / t - y**n]


t_0: float = 1e-10
t_final: float = 4
u0: list[float] = [1, 0]
t = np.linspace(t_0, t_final)

for n in range(0, 5):
    sol: ArrayLike = solve_ivp(
        fun=lane_emden, t_span=(t_0, t_final), y0=u0, args=(n,), dense_output=True
    )
    u = sol.sol(t)
    plt.plot(t, u[0, :], label=f"{n}")
    plt.legend(shadow=True)

plt.savefig("lane_emden.pdf")
