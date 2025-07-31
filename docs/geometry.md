# Geometries

The geometry inputs consist of first a geometry type (tube bank in orthogonal cross flow), as well as some overall dimensions (frontal area of one fluid, length) which are integral to the integration problem. The internal dimensions (hydraulic diameter or tube diameter, as well as tube spacing)

# Orthogonal Tube Bank

A geometry where the tubes bank is always orthogonal to the flow around the tube bank. This can for
example be a straight tube bank in a box, or an axial involute tube bank in an annulus.

The tubes per row and number of rows are integers and hence will be chosen as primary input variables. However when a user inputs a tube bank length (in the direction of the flow around the tubes), and the spacing between tubes, one of these two must be rounded to achieve an integer number of tubes.

## Side note on non-dimensional parameters for pressure drop in tube banks

To ensure clear comparison between the pressure drop correlations as presented in Shah 2003, VDI and Martin 2002, the definitions of three non-dimensionals will be clearly presented here.

As there are mistakes in the transcription of the correlation in Shah the comparison is important. (the author has been informed and online versions of the 2023 edition have been corrected)

For a tube bank with tube outer diameter based Reynolds number $\text{Re} \triangleq \frac{Gd}{\mu}$, the mass velocity $G \triangleq \dot{m}/A_o$ is based on the minimum flow area. For inline tubes this always occurs at the same location but for transverse tubes if the tubes are narrowly enough spaced longitudinally, the highest velocities can occur in a direction different to the bulk flow (Make image).

The Hagen number is used by Martin 2002 and Shah 2003

$$
\begin{equation}
\text{Hg} \triangleq \frac{\Delta p}{N_r} \cdot \frac{\rho d_\text{o}^2}{\mu^2}
\end{equation}
$$

In the original Gaddis and Gnielinski 1985 correlation, or as presented in the VDI Heat Atlas, the pressure drop is presented in the form of the drag coefficient $\xi$

$$
\begin{equation}
\xi \triangleq \frac{\Delta p}{n_\text{MR}} \cdot \frac{2\rho}{G^2}
\end{equation}
$$

Where the number of main resistors $n_\text{MR}$ is equal to the number of rows ($n_\text{MR} = N_r$) for inline banks and staggered banks where the minimum free flow area occurs between to tubes of the same row. In the other case ($2X_l^* < \sqrt{2X_t^*+1}$), $n_\text{MR} = N_r-1$.

This means that for the cases where $n_\text{MR} = N_r$, $\text{Hg} = \xi \text{Re}^2$.
