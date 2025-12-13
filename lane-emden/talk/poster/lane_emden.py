#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike
from numpy.polynomial import Polynomial
from scipy.integrate import solve_ivp
from scipy.special import factorial

plt.style.use("seaborn-v0_8-white")


def lane_emden(t: float, u: ArrayLike, n: float) -> ArrayLike:
    y, z = u
    return [z, -2 * z / t - y**n]


t_0: float = np.finfo(float).eps  # xe-16
t_final: float = 2
u0: list[float] = [1, 0]
t = np.linspace(start=t_0, stop=t_final, num=2000)
fig, ax = plt.subplots(layout="constrained")
for n in [0, 1, 2, 3, 4, 5]:
    sol: ArrayLike = solve_ivp(
        fun=lane_emden, t_span=(t_0, t_final), y0=u0, args=(n,), dense_output=True
    )
    ax.plot(
        t,
        sol.sol(t)[0, :],
        label=rf"RK5(4) / ${n}$",
        linestyle="solid",
        linewidth=0.7,
    )
    ax.plot(
        t,
        Polynomial(
            coef=(
                1,
                0,
                -1 / 6,
                0,
                n / 120,
                0,
                n * (8 * n - 5) / (3 * factorial(7)),
                0,
                n * (70 - 183 * n + 122 * n**2) / (9 * factorial(9)),
                0,
                n
                * (3150 - 1080 * n + 12642 * n**2 - 5032 * n**3)
                / (45 * factorial(11)),
                0,
                n
                * (183616 * n**4 - 663166 * n**3 + 915935 * n**2 - 574850 * n + 138600)
                / 840647808000,
                0,
                n
                * (
                    -21625216 * n**5
                    + 103178392 * n**4
                    - 200573786 * n**3
                    + 199037015 * n**2
                    - 101038350 * n
                    + 21021000
                )
                / 1235752277760000,
            )
        )(t),
        label=rf"ADM / ${n}$",
        linestyle="dotted",
        linewidth=0.7,
    )
ax.set_xlim(t_0, t_final)
ax.set_ylim(0, 1)
ax.set_xlabel(xlabel=r"$\xi$", fontsize=15)
ax.set_ylabel(ylabel=r"$\theta\left(\xi\right)$", fontsize=15)
ax.set_xticks(ticks=np.linspace(start=t_0, stop=t_final, num=2))
ax.set_yticks(ticks=np.linspace(start=0, stop=1, num=2))
ax.grid(c="gray", linewidth=0.1, linestyle="dashed")
ax.text(
    0.8,
    0.1,
    r"$\boxed{\frac{1}{\xi^{2}}\diff*{\left(\xi^{2}\diff{\theta}{\xi}\right)}{\xi}+\theta^{n}=0, \theta\left(0\right)=1, \theta^{\prime}\left(0\right)=0}$",
    fontsize=12,
)
ax.legend(
    loc="best",
    shadow=True,
    fontsize=12,
    title=r"Numerical Method / Polytropic index $\left(n\right)$",
)
ax.set_title(
    label="Lane-Emden equation, ADM with 15 terms",
    loc="center",
    wrap=True,
    fontsize=15,
)
ax.spines["bottom"].set_color("none")
ax.spines["top"].set_color("none")
ax.spines["left"].set_color("none")
ax.spines["right"].set_color("none")
fig.savefig("lane_emden.pdf", transparent=True, bbox_inches="tight")
fig.clf()
