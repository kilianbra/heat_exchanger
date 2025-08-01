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

The Hagen number is used by Martin 2002 (eqn A.3) and Shah 2003 (eqn 7.115)

$$
\begin{equation}
\text{Hg} \triangleq \frac{\Delta p}{N_r} \cdot \frac{\rho d_\text{o}^2}{\mu^2}
\end{equation}
$$

In the original Gaddis and Gnielinski 1985 correlation (eqn 1), or as presented in the VDI Heat Atlas 2010 (L1.4 eqn 1), the pressure drop is presented in the form of the drag coefficient $\xi$

$$
\begin{equation}
\xi \triangleq \frac{\Delta p}{n_\text{MR}} \cdot \frac{2\rho}{G^2}
\end{equation}
$$

Where the number of main resistors $n_\text{MR}$ is equal to the number of rows ($n_\text{MR} = N_r$) for inline banks and staggered banks where the minimum free flow area occurs between to tubes of the same row. In the other case ($2X_l^* < \sqrt{2X_t^*+1}$), $n_\text{MR} = N_r-1$.

This means that for the cases where $n_\text{MR} = N_r$, $\text{Hg} = \xi \frac{\text{Re}^2}{2}$.

These two formula for pressure drop assume negligible (static) pressure change due to density changes along the direction of flow. Heat Exchangers have changes in temperature and hence of density, which is not neglected in the defintion of the Kays and London effective friction factor or effective shear stress.

For tube banks

The pressure drop across a heat exchanger with varying density can be expressed as:

$$
\Delta p = \frac{G^2}{2} \left[ (1 + \sigma^2) \left( \frac{1}{\rho_2} - \frac{1}{\rho_1} \right) + f_o \frac{A_q}{A_o} \left( \frac{1}{\rho} \right)_\text{m} \right]
$$

where:

- $G$ is the mass velocity (based on minimum free flow area),
- $\sigma$ is the contraction ratio (A_frontal/A_o)
- $\rho_1$ and $\rho_2$ are the fluid densities at the inlet and outlet, respectively,
- $f_o$ is the equivalent friction factor,
- $A_q$ is the heat transfer area,
- $A_o$ is the minimum free flow area,
- $\left( \frac{1}{\rho} \right)_\text{m}$ is the mean value of the reciprocal density along the flow path.

This equation accounts for both the static pressure change due to density variation and the frictional pressure drop.

If we assume small pressure change due to density change (flow acceleration and entrance/exit effects), then this friction factor can be related to $\text{Hg} = \xi \text{Re}^2$ by the following equation: $f_o = \frac{X_t^*-1}{\pi} \xi$ for a rectangular or axial involute tube bank.

## Property variation of properties in the direction of flow

Kays and London in Section 4 calim that if absolute temperature variation is less than 2:1 from one end to the other, using a mean temperature with respect to flow length is fine. Mainly an issue for pressure drop which relies on density.
