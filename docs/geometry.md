# Geometries

The geometry inputs consist of first a geometry type (tube bank in orthogonal cross flow), as well as some overall dimensions (frontal area of one fluid, length) which are integral to the integration problem. The internal dimensions (hydraulic diameter or tube diameter, as well as tube spacing)

# Orthogonal Tube Bank

A geometry where the tubes bank is always orthogonal to the flow around the tube bank. This can for
example be a straight tube bank in a box, or an axial involute tube bank in an annulus.

The tubes per row and number of rows are integers and hence will be chosen as primary input variables. However when a user inputs a tube bank length (in the direction of the flow around the tubes), and the spacing between tubes, one of these two must be rounded to achieve an integer number of tubes.

## Side note on non-dimensional parameters for pressure drop in tube banks

To ensure clear comparison between the pressure drop correlations as presented in Shah 2003, VDI and Martin 2002, the definitions of three non-dimensionals will be clearly presented here. Correlations for tube bundles can be framed in terms of the real tube outer diameter $d_o$, the equivalent diameter of the bundle or of the constriction. Similarly, the (mass) velocity in the narrowest cross section, in the free cross section or an average can be used.

Gaddis and Gnielinski 1985 specify that equations for an ideal tube bundle are valid if there are $\geq 10$ rows of $\geq 10$ tubes each with a length to diameter ratio $\geq 10$. Deviations from this can be accounted for via correction factors.
They have recast the pressure drop in terms of the velocity in the narrowest cross section and using the tube outer diameter as the length scale. Fluid properties ($\mu, \rho$) are evaluated at the arithmetic mean pressure and temperature (inlet and exit).

As there are mistakes in the transcription of the correlation in Shah the comparison is important. (the author has been informed and online versions of the 2023 edition have been corrected)

For a tube bank with tube outer diameter based Reynolds number $\text{Re} \triangleq \frac{Gd}{\mu}$, the mass velocity $G \triangleq \dot{m}/A_o$ is based on the minimum flow area. For inline tubes this always occurs at the same location but for transverse tubes if the tubes are narrowly enough spaced longitudinally, the highest velocities can occur in a direction different to the bulk flow (Make image).

### Hagen number (Martin 2002 and Shah 2003)

The Hagen number is used by Martin 2002 (eqn A.3) and Shah 2003 (eqn 7.115). It's main advantage is that it does not depend on any velocity and hence there can be no confusion/difficulty to determine experimentally.

$$
\begin{equation}
\text{Hg} \triangleq \frac{\Delta p}{N_r} \cdot \frac{\rho d_\text{o}^2}{\mu^2}
\end{equation}
$$

### Drag coefficient $\xi$ (Gaddis and Gnielinski 1985 and VDI Heat Atlas 2010)

In the original Gaddis and Gnielinski 1985 correlation (eqn 1), or as presented in the VDI Heat Atlas 2010 (L1.4 eqn 1), the pressure drop is presented in the form of the drag coefficient $\xi$. This definition depends on the pressure drop per passage through a row of main resistors (which can be thought of as approximately the number of rows for tube banks with many rows).

$$
\begin{equation}
\xi \triangleq \frac{\Delta p}{n_\text{MR}} \cdot \frac{2\rho}{G^2}
\end{equation}
$$

The number of main resistors $n_\text{MR}$ is equal to the number of rows ($n_\text{MR} = N_r$) for inline banks and staggered banks where the minimum free flow area occurs between to tubes of the same row. In the other case the flow has it's highest velocity not between two tubes of the same row but in the diagonal between a tube of one row and the next ($2X_l^* < \sqrt{2X_t^*+1}$), hence for $N_r$ staggered tube rows there are only $n_\text{MR} = N_r-1$ occurences of this maximum velocity.

For the cases where $n_\text{MR} = N_r$, $\text{Hg} = \xi \frac{\text{Re}^2}{2}$.

These two formula for pressure drop assume negligible (static) pressure change due to density changes along the direction of flow. Heat Exchangers have changes in temperature and hence of density, which is not neglected in the defintion of the Kays and London effective friction factor or effective shear stress.

For tube banks

The pressure drop across a heat exchanger with varying density can be expressed as:

$$
\Delta p = \frac{G^2}{2} \left[ (1 + \sigma^2) \left( \frac{1}{\rho_2} - \frac{1}{\rho_1} \right) + f_o \frac{A_\tau}{A_o} \left( \frac{1}{\rho} \right)_\text{m} \right]
$$

where:

- $G$ is the mass velocity (based on minimum free flow area),
- $\sigma$ is the contraction ratio ($A_\text{frontal}/A_o$)
- $\rho_1$ and $\rho_2$ are the fluid densities at the inlet and outlet, respectively,
- $f_o$ is the equivalent friction factor,
- $A_\tau$ is the wetted surface area (over which friction occurs),
- $A_o$ is the minimum free flow area,
- $\left( \frac{1}{\rho} \right)_\text{m}$ is the mean value of the reciprocal density along the flow path.

This equation accounts for both the static pressure change due to density variation and the frictional pressure drop.

### Equivalent friction factor $f_o$ (Kays and London 1985)

If we assume small pressure change due to density change (flow acceleration and entrance/exit effects) and neglect the friction area on the insulated outer and inner walls compared to the heat transfer area of the tubes, then this friction factor $f_o$ is

$$
\begin{equation}
f_o \triangleq \frac{2\rho\tau_o}{G^2} \simeq \frac{A_o\Delta p}{A_\tau} \cdot \frac{2\rho}{G^2}
\end{equation}
$$

and can related to the drag coefficient $\xi$ by the following equation: $f_o = \frac{X_t^*-1}{\pi} \xi$ (for tube bankes that are rectangular or axial involute).

### Corrected half friction factor (Gunter and Shaw 1944)

The correlation used in a recent MIT article (Gomez-Vega et al 2025 [https://doi.org/10.2514/6.2025-0089](https://doi.org/10.2514/6.2025-0089)), another correlation is used by Gunter and Shaw 1944 [https://doi.org/10.1115/1.4018353](https://doi.org/10.1115/1.4018353) is used which is based on the non dimensional group

$$
\begin{equation}
\frac{\Delta p}{L} \cdot \frac{\rho}{G^2} d_v = \frac{\Delta p}{N_r} \cdot \frac{\rho}{G^2} \frac{d_v}{X_l} = \frac{\Delta p}{N_r} \cdot \frac{\rho}{G^2} \frac{\frac{4X_l^*X_t^*}{\pi}-1}{X_l^*}
\end{equation}
$$

where $d_v$ is the volumetric hydraulic diameter which Kilian calculated to be $d_\text{o} (\frac{4X_l^*X_t^*}{\pi}-1)$ for both inline and staggered configurations.

The above assumes that the fluid flow length $L=N_rX_l^*d_\text{o}$

### Different hydraulic diameters $d_v\neq d_h$

Kays and London, Murray and other authors call hydraulic diameter the quantity

$$
\begin{equation}
d_h \triangleq \frac{4A_o L}{A_\tau}
\end{equation}
$$

Where L is, in the case of tube banks, the equivalent flow length from the "leading edge of the first tube row to the leading edge of a tube row that would follow the last tube row, were another tube row present" (KnL p8).

The general definition of a hydraulic diameter for a non-unfirom and non-circular cross-section is called the volumetric hydraulic diameter by Gunter and Shaw and is based on the wetted volume and is independent of the minimum free-flow cross sectional area $A_o$.

$$
\begin{equation}
d_v \triangleq \frac{4V}{A_\tau} \geq d_h
\end{equation}
$$

The inequality is strict for cases like tube banks where the cross sectional flow area is on average greater than the minimum flow area.

For all straight tube banks, $d_v/d_\text{o} =\frac{4X_l^*X_t^*}{\pi}-1$
And if the dimensionless throat spacing is denoted $X_o^* = \text{min}(X^*_t-1, 2(X_d^*-1))$ where the dimensionless diagonal spacing is $(X_d^*)^2 = (X_t^*/2)^2 + (X_l^*)^2$, then $d_h/d_o = 4X_l^*X_o^*/\pi$

## Property variation of properties in the direction of flow

Kays and London in Section 4 calim that if absolute temperature variation is less than 2:1 from one end to the other, using a mean temperature with respect to flow length is fine. Mainly an issue for pressure drop which relies on density.
