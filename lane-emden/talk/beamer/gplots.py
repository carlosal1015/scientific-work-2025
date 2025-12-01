#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np

plt.style.use("seaborn-v0_8-white")

theta = np.linspace(start=0, stop=1)
m = 2
C = 1
fig, ax = plt.subplots(layout="constrained")
# ax.plot(
#     theta,
#     np.pow(theta, m),
#     label=r"$\theta^{m}$",
#     linewidth=0.7,
# )
# ax.plot(
#     theta,
#     np.exp(theta),
#     label=r"$\exp\left(\theta\right)$",
#     linewidth=0.7,
# )
ax.plot(
    theta,
    np.pow(np.pow(theta, 2) - C, 3 / 2),
    label=r"$\left(\theta^{2}-C\right)^{\frac{3}{2}}$",
    linewidth=0.7,
)
ax.legend(loc="best")
plt.savefig("picture0.pdf", transparent=True, bbox_inches="tight")
plt.clf()