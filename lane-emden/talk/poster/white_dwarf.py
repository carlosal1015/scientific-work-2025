import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike
from numpy.polynomial import Polynomial
from scipy.integrate import solve_ivp
from scipy.special import factorial

plt.style.use("seaborn-v0_8-white")


def white_dwarf(t: float, u: ArrayLike, c: float) -> ArrayLike:
    y, z = u
    return [z, -2 * z / t - np.power(np.abs(np.power(y, 2) - c), 3 / 2)]


t_0: float = np.finfo(float).eps  # xe-16
t_final: float = 2
u0: list[float] = [1, np.finfo(float).eps]
t = np.linspace(start=t_0, stop=t_final, num=2000)

fig, ax = plt.subplots(layout="constrained")
for c in [0, 0.2, 0.4, 0.6, 0.8]:
    sol: ArrayLike = solve_ivp(
        fun=white_dwarf,
        t_span=(t_0, t_final),
        y0=u0,
        args=(c,),
        dense_output=True,
    )
    ax.plot(
        t,
        sol.sol(t)[0, :],
        label=rf"RK45 / ${c}$",
        linestyle="solid",  # solid dashed dashdot dotted
        linewidth=0.8,
    )
    q = np.power(np.abs(c - 1), 1 / 2)
    ax.plot(
        t,
        Polynomial(
            coef=(
                1,
                0,
                -1 / 6 * q**3,
                0,
                1 / 40 * q**4,
                0,
                q**5 * (5 * q**2 + 14) / factorial(7),
                0,
                q**6 * (339 * q**2 + 280) / (3 * factorial(9)),
                0,
                q**7 * (1425 * q**4 + 11436 * q**2 + 4256) / (5 * factorial(11)),
                0,
            )
        )(t),
        label=rf"ADM / ${c}$",
        linestyle="dotted",
        linewidth=0.7,
    )
ax.set_xlim(t_0, t_final)
ax.set_ylim(1 / 2, 1)
ax.set_xlabel(xlabel=r"$\xi$", fontsize=12)
ax.set_ylabel(ylabel=r"$\theta\left(\xi\right)$", fontsize=12)
ax.set_xticks(ticks=np.linspace(start=t_0, stop=t_final, num=2))
ax.set_yticks(ticks=np.linspace(start=1 / 2, stop=1, num=2))
ax.grid(c="gray", linewidth=0.1, linestyle="dashed")
ax.text(
    0.64,
    0.535,
    r"$\boxed{\frac{1}{\xi^{2}}\diff*{\left(\xi^{2}\diff{\theta}{\xi}\right)}{\xi}+{\left(\theta^{2}-C\right)}^{\frac{3}{2}}=0, \theta\left(0\right)=1, \theta^{\prime}\left(0\right)=0}$",
    fontsize=12,
)
ax.legend(
    loc="best",
    shadow=True,
    fontsize=12,
    title=r"Numerical Method / Constant $\left(C\right)$",
)
ax.set_title(
    label="Numerical solution of the Chandrasekhar's white dwarf equation, ADM with 11 terms",
    loc="center",
    wrap=True,
    fontsize=15,
)
ax.spines["bottom"].set_color("none")
ax.spines["top"].set_color("none")
ax.spines["left"].set_color("none")
ax.spines["right"].set_color("none")
fig.savefig("white_dwarf.pdf", transparent=True, bbox_inches="tight")
fig.clf()
