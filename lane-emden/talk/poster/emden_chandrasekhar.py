import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike
from numpy.polynomial import Polynomial
from scipy.integrate import solve_ivp

plt.style.use("seaborn-v0_8-white")


def emden_chandrasekhar(t: float, u: ArrayLike) -> ArrayLike:
    y, z = u
    return [z, -2 * z / t + np.exp(-y)]


t_0: float = np.finfo(float).eps  # xe-16
t_final: float = 4
u0: list[float] = [np.finfo(float).eps, np.finfo(float).eps]
t = np.linspace(start=t_0, stop=t_final, num=2000)

fig, ax = plt.subplots(layout="constrained")

sol: ArrayLike = solve_ivp(
    fun=emden_chandrasekhar,
    t_span=(t_0, t_final),
    y0=u0,
    dense_output=True,
)
ax.plot(
    t,
    np.exp(-sol.sol(t)[0, :]),
    label=r"RK45",
    linestyle="solid",  # solid dashed dashdot dotted
    linewidth=0.8,
)
ax.plot(
    t,
    np.exp(
        -Polynomial(
            coef=(
                0,
                0,
                1 / 6,
                0,
                -1 / (120),
                0,
                1 / (1890),
                0,
                -61 / (1632960),
                0,
                629 / 224532000,
            )
        )(t)
    ),
    label="ADM",
    linestyle="dotted",
    linewidth=0.7,
)
ax.set_xlim(t_0, t_final)
ax.set_ylim(0, 1)
ax.set_xlabel(xlabel=r"$\xi$", fontsize=12)
ax.set_ylabel(ylabel=r"$e^{-\theta\left(\xi\right)}$", fontsize=12)
ax.set_xticks(ticks=np.linspace(start=t_0, stop=t_final, num=2))
ax.set_yticks(ticks=np.linspace(start=0, stop=1, num=2))
ax.grid(c="gray", linewidth=0.1, linestyle="dashed")
ax.text(
    0.4,
    0.15,
    r"$\boxed{\frac{1}{\xi^{2}}\diff*{\left(\xi^{2}\diff{\theta}{\xi}\right)}{\xi}-e^{-\theta}=0, \theta\left(0\right)=0, \theta^{\prime}\left(0\right)=0}$",
    fontsize=12,
)
ax.legend(loc="best", shadow=True, fontsize=12, title="Legend")
ax.set_title(
    label="Numerical solution of the Emden-Chandrasekhar equation, ADM with 11 terms",
    loc="center",
    wrap=True,
    fontsize=15,
)
ax.spines["bottom"].set_color("none")
ax.spines["top"].set_color("none")
ax.spines["left"].set_color("none")
ax.spines["right"].set_color("none")
fig.savefig("emden_chandrasekhar.pdf", transparent=True, bbox_inches="tight")
fig.clf()
