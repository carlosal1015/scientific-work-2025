#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np

plt.style.use("seaborn-v0_8-white")

ξ = np.linspace(start=0, stop=4)
θ0 = 1 - np.pow(ξ, 2) / 6
θ1 = np.sin(ξ) / ξ
θ2 = np.pow(1 + 1 / 3 * np.pow(ξ, 2), -1 / 2)
fig, ax = plt.subplots(layout="constrained")
ax.plot(
    ξ,
    θ0,
    label=r"$n=0$",
    linewidth=0.7,
)
ax.plot(
    ξ,
    θ1,
    label=r"$n=1$",
    linewidth=0.7,
)
ax.plot(
    ξ,
    θ2,
    label=r"$n=5$",
    linewidth=0.7,
)
ax.legend(loc="best", shadow=True, fontsize=15)
ax.set_xlim(ξ.min(), ξ.max())
ax.set_ylim(-1, 1)
ax.set_xlabel(xlabel=r"$\theta$", fontsize=15)
ax.set_ylabel(ylabel=r"$\theta\left(\xi\right)$", fontsize=15)
ax.set_xticks(np.linspace(start=ξ[0], stop=ξ[-1], num=5))
ax.set_yticks(np.linspace(start=-1, stop=1, num=5))
ax.grid(c="gray", linewidth=0.1, linestyle="dashed")
ax.spines["bottom"].set_color("none")
ax.spines["top"].set_color("none")
ax.spines["left"].set_color("none")
ax.spines["right"].set_color("none")
plt.savefig("exacts.pdf", transparent=True, bbox_inches="tight")
plt.clf()
