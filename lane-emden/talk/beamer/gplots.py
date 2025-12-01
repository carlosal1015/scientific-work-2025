#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np

plt.style.use("seaborn-v0_8-white")

θ = np.linspace(start=0, stop=1)
m = 2
C = 0
fig, ax = plt.subplots(layout="constrained")
ax.plot(
    θ,
    np.pow(θ, m),
    label=r"$\theta^{m}$",
    linewidth=0.7,
)
ax.plot(
    θ,
    np.exp(θ),
    label=r"$\exp\left(\theta\right)$",
    linewidth=0.7,
)
ax.plot(
    θ,
    np.pow(np.pow(θ, 2) - C, 3 / 2),
    label=r"$\left(\theta^{2}-C\right)^{\frac{3}{2}}$",
    linewidth=0.7,
)
ax.legend(loc="best", shadow=True, fontsize=15)
ax.set_xlim(θ.min(), θ.max())
ax.set_ylim(0, np.exp(θ).max())
ax.set_xlabel(xlabel=r"$\theta$", fontsize=15)
ax.set_ylabel(ylabel=r"$g\left(\theta\right)$", fontsize=15)
ax.set_xticks(np.linspace(start=θ[0], stop=θ[-1], num=3))
ax.set_yticks(np.linspace(start=0, stop=2, num=3))
ax.grid(c="gray", linewidth=0.1, linestyle="dashed")
ax.spines["bottom"].set_color("none")
ax.spines["top"].set_color("none")
ax.spines["left"].set_color("none")
ax.spines["right"].set_color("none")
plt.savefig("gplots.pdf", transparent=True, bbox_inches="tight")
plt.clf()
