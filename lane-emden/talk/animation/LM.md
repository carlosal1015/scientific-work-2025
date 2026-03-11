## Abstract

The Lane-Emden equation is a non-linear differential equation governing the equilibrium of polytropic stationary self-gravitating, spherically symmetric star models;

$$\frac1{\xi^2}\frac d{d\xi}\left(\xi^2\frac{d\theta}{d\xi}\right)+\theta^n=0.$$

In the isothermal cases we have the Chandrasekhar equation
$$\frac1{\xi^2}\frac d{d\xi}\left(\xi^2\frac{d\psi}{d\xi}\right)-e^{-\psi}=0$$


After having derived these models, we will go through all cases for which analytic solutions are achievable.
Moreover, we will discuss the existence and uniqueness of positive solutions under specific boundary conditions by transforming the equations to autonomous ones. The analysis depends upon the value of the polytropic index $n$.
We also compute some solutions numerically.

## Introduction: Lane-Emden Equation

In this dissertation we will consider spherically symmetric mass distribution of stellar structures.
We start from the concept that pressure and gravity are the main forces determining such a structure and they must be balanced in order for the star not to collapse under its own weight.
The equations describing the structure of such models are the hydrostatic balance equation:
$$\frac{dP}{dr}=-\frac{Gm}{r^2}\rho$$
and the mass conservation equation:
$$\frac{dm}{dr}=4\pi\rho r^2$$
with mass $m$, pressure $P$, density $\rho$ and radius $r$.
Here G is the gravitational constant.
We can combine the equations above into a single second order ODE.
Multiplying (1) by $r^2/\rho$ and differentiating gives us
$$\frac d{dr}\left(\frac{r^2}\rho\frac{dP}{dr}\right)=-G\frac{dm}{dr}$$
By substituting eq.(2) in eq.(3) we get the hydrostatic equilibrium equation in the form:
$$\frac1{r^2}\frac d{dr}\left(\frac{r^2}\rho\frac{dP}{dr}\right)=-4\pi\rho G$$