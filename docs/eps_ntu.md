# The Effectiveness-NTU Equations

When modelling heat exchangers in a 0-dimensional manner, without marching through the fluids as heat is transfered, the flow configuration limits how close the cold fluid can get in temperature to the hot fluid. This is modelled by using the $\varepsilon-N_\text{tu}$ equations, which have been derived for a certain set of simplifications.

## Assumptions

Constant $c_p$ and uniform overall heat transfer coefficient.

In general $U$ varies through the heat exchanger and we are interested in $\overline{U} \triangleq \frac{1}{A}\int UdA$.

In practical 0D modelling, $\frac{1}{U} = \frac{1}{h_1} + \frac{1}{h_2}$ is used as a first order approximation.

## Specific case of unmixed cross-flow relationship

In the analysis of heat exchangers, particularly for crossflow heat exchangers with both fluids unmixed, the effectiveness ($\varepsilon$) can be expressed using a series expansion involving a special polynomial $P_n(y)$. This approach is part of the standard effectiveness-NTU (Number of Transfer Units) method.

## Definition

The $P_n(y)$ polynomial is defined as:

\begin{equation}
P_n(y) = \frac{1}{(n+1)!} \sum_{j=1}^n \frac{n+1-j}{j!} y^{n+j}
\end{equation}

where:
- $n$ is a positive integer,
- $y$ is typically the $N_{\text{tu}}$ (Number of Transfer Units) parameter.

## Context

This polynomial appears in the series solution for the effectiveness of a crossflow heat exchanger with both fluids unmixed:

\begin{equation}
\varepsilon = 1 - e^{-N_{\text{tu}}} - e^{-(1+C_{\text{r}})N_{\text{tu}}} \sum_{n=1}^{\infty} C_{\text{r}}^n P_n(N_{\text{tu}})
\end{equation}

where $C_{\text{r}}$ is the heat capacity rate ratio ($C_{\text{min}}/C_{\text{max}}$).

## References
- Kays, W. M., & London, A. L. (1984). *Compact Heat Exchangers* (3rd ed.). McGraw-Hill.
- Shah, R. K., & Sekulic, D. P. (2003). *Fundamentals of Heat Exchanger Design*. Wiley.
