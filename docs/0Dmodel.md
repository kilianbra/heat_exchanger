The most basic model for calculating heat exchanger effectiveness and pressure drop will first be presented.
Then the assumptions that have been taken to be able to simplify down to this level will be detailed.

# Heat Transfer

## General definitions and local versus mean properties

For each section of the heat exchanger the heat transfer can be related to the local bulk static
temperature difference by the following equation (Shah eqn 3.4)

$$
\begin{equation}
dq = U \Delta T dA_q
\end{equation}
$$

Once integrated, the total heat transfer can be obtained

$$
\begin{equation}
q = \int U \Delta T dA_q = U_m A_q\Delta T_m
\end{equation}
$$

Where

$$
\begin{equation}
\frac{1}{\Delta T_m} \triangleq \frac{1}{q}\int U \frac{dq}{\Delta T}
\end{equation}
$$

and

$$
\begin{equation}
U_m \triangleq \frac{1}{A_q}\int U dA_q
\end{equation}
$$

## Overall heat transfer coefficient

Assuming no fouling on both fluid sides ($1$ & $2$), the local overall thermal conductance
$\mathbf{R}_\text{o} = \frac{1}{UA}$ (units K/J) can be shown to be
(Shah eqn 3.20 p.108), with surface efficiencies $\eta_o$

$$
\begin{equation}\label{eq:UA}
\frac{1}{UA} = \frac{1}{(\eta_o h A_q)_1 } + \mathbf{R}_\text{w} + \frac{1}{(\eta_o h A_q)_2 }
\end{equation}
$$

Care must be taken to specify which area is taken when quoting $U$ separately from an area such that
$UA = U_1 A_{q,1} = U_2 A_{q,2}$

## Number of transfer units

The number of transfer units is obtained by non-dimensionalising the mean overall thermal conductance
by the lower heat capacity flux of the two fluids $C_\text{min}=\min(\dot{m} \overline{c_p})$ where
$\overline{c_p} \triangleq \frac{\Delta h}{\Delta T}$ for fluids with non-negligible $c_p$ variation.

$$
\begin{equation}
N_\text{tu} \triangleq \frac{U_m A}{\dot{C}_\text{min}}
\end{equation}
$$

Equation $\eqref{eq:UA}$ can be then rewriten as the local or mean overall heat transfer coefficient
as a function of the local or mean Stanton numbers of each fluid.
Assuming a negligible wall resistance:

$$
\begin{align}
\frac{C_\text{min}}{U A} &=  \left(\frac{\dot{m} \overline{c_p}}{\eta_o h A_q }\right)_1  \cdot \frac{C_\text{min}}{C_1} + \left(\frac{\dot{m} \overline{c_p}}{\eta_o h A_q}\right)_2 \cdot \frac{C_\text{min}}{C_2} \\
&= \left(\frac{G \overline{c_p}}{h}\right)_1 \cdot \frac{A_{o,1}}{\eta_{o,1}A_{q,1}} \cdot \frac{C_\text{min}}{C_1} + \left(\frac{G \overline{c_p}}{h}\right)_2 \cdot \frac{A_{o,1}}{\eta_{o,1}A_{q,1}} \cdot \frac{C_\text{min}}{C_2}
\end{align}
$$

Which, if taken for adequate mean values and for example $C_\text{min}=C_2$ and hence $C^*=C_2/C_1$

$$
\begin{equation}
\frac{1}{N_\text{tu}} = \frac{1}{\text{St}_1} \cdot \frac{A_{o,1}C^*}{\eta_{o,1}A_{q,1}} + \frac{1}{\text{St}_2} \cdot \frac{A_{o,1}}{\eta_{o,1}A_{q,1}}
\end{equation}
$$

In the case where $\text{Pr}=1$ and hence $j=\text{St}$, and where no fins are used ($\eta_o=1$).
Using the heat transfer area hydraulic diameter (see below for Note on $4L/d_h$)

$$
\begin{equation}
N_\text{tu} = \frac{1}{\frac{1}{(j/f) \cdot f \cdot (4L/d_\text{h})} + \frac{1}{(j/f) \cdot f \cdot (4L/d_\text{h})}}
\end{equation}
$$

For a counterflow heat exchanger with $C^*=1$

$$
\begin{equation}
\varepsilon = \frac{N_\text{tu}}{1 + N_\text{tu}}
\end{equation}
$$

Hence the temperature change of fluid $i$ can be determined using

$$
\begin{equation}
\Delta T_i = \pm \frac{q}{\overline{c_{p,i}}} =  \pm \varepsilon \frac{ C_\text{min}}{C_i}(T_\text{hot,in} - T_\text{cold,in})
\end{equation}
$$

# Pressure drop

## Full equation $\Delta p$

The full equation for static pressure drop in a heat exchanger is as follows (modified from Shah and Sekulic 2003 equation 6.28):

$$
\begin{equation}\label{eqn:full_dp}
\Delta p = \frac{G^2}{2} \left[ \frac{1}{\rho_{\text{i}}} \left( 1 - \sigma^2 + K_{\text{contr}} \right) - \frac{1}{\rho_{\text{o}}} \left( 1 - \sigma^2 + K_{\text{exp}} \right) + 2 \left( \frac{1}{\rho_{\text{o}}} - \frac{1}{\rho_{\text{i}}} \right) + f \frac{4L}{d_\text{h}} \left( \frac{1}{\rho} \right)_{\text{m}} \right]
\end{equation}
$$

From the full frontal area allocated to each fluid, heat exchangers tend to first have a contraction of area before the heat exchanger core, and are followed by an expansion.
Furthermore there is both a pressure drop due to wall shear stresses (what about pressure forces on the back of cylinders?) and due to momentum effects.

In this equation $K_{\text{contr}}$ and $K_{\text{exp}}$ are the contraction and expansion coefficients, respectively. $K_{\text{contr}}>0$ is always positive (indicating a pressure drop) but $K_{\text{exp}}$ can be a positive or negative, i.e. a pressure drop or a pressure gain, due to momentum change. The pressure drop comes from the irreversible free expansion (potentially after the contraction).
In the case of a fluid with negligible density change, the $1-\sigma^2$ term is fully recovered at the outlet. The mean specific volume $\left( \frac{1}{\rho} \right)_{\text{m}}$ can be approximated by averaging the inlet and oultet specific volumes (this is exact for ideal gases with equal heat capacities $C_r=1$ in cross/counterflow).
In general, if a fluid flows along a total length $L$ with a density profile $\rho(x)$, the mean specific volume is:

$$
\begin{equation}
\left( \frac{1}{\rho} \right)_{\text{m}} = \frac{1}{L} \int_0^{L} \frac{dx}{\rho(x)}
\end{equation}
$$

This way, if the friction factor $f$ is assumed constant (or an appropriate average is taken), the above equation $\eqref{eqn:full_dp}$ can be obtained from integrating through the heat exchanger

## Simplified $\Delta p$

For now, this model neglects the pressure changes due to inlet contraction and exit expansion, as well as the momentum effects.
To further simplify the equations the density is assumed to not change much hence $\left( \frac{1}{\rho} \right)_m = \frac{1}{\rho_\text{in}}$

$$
\begin{equation}
\Delta p = \frac{1}{2} \cdot \left( \frac{\dot{m}}{A_o} \right)^2  \cdot f \cdot \frac{4L}{d_h} \left( \frac{1}{\rho_\text{in}} \right)
\end{equation}
$$

## Note on $\frac{4L}{d_h}$

By definition of the hydraulic diameter

$$
\begin{equation}
\frac{4L}{d_h} = \frac{A_\text{surf}}{A_\text{freeflow}}
\end{equation}
$$

The issue is that when considering friction surface area and heat transfer area there are occasional differences
(most notably in the annular flow of a concentric double pipe heat exchanger, where the outer wall is insulated).

In this document the heat transfer surface area of fluid $j$ will therefore be denoted $A_{q,j}$ whereas the shear
stress and friction surface area will be denoted $A_{\tau,j}$. The minimum free flow area will always be denoted
$A_o$ where $o$ is the effective throat
