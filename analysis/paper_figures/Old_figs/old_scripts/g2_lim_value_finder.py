"""

kpb30 5/12/25

This script is used to find the limiting value of g^2 for the full entropy equation of a heat exchanger.
b is the constant before the pressure drop terms (2 (gamma-1) / gamma)
c is the constant before the NTU terms (if t is T_hi/T_ci then C = (t-1)^2/t)
a is the term who'se limiting value we are looking for and is such that x = NTU
therefore the pressure drop term g^2 f A/Ao = a x = a NTU

If the correlation over a reasonable range of t is approximately linear,
say a_lim = 0.03 * (t-1), then a_lim = g^2_lim * f C_min / U / Ao i.e.
a = g^2 f / St_both_sides




bax^3 + (ca + ba(3 + c))x^2 + (-c(a + 1) + ba(3 + c))x + (c + ba)

3bax^2 + 2(ca + ba(3 + c))x + (-c(a + 1) + ba(3 + c)) = 0


quadratic solution (positive root):

x = ((-2 a b c - 6 a b - 2 a c) +
sqrt((2 a b c + 6 a b + 2 a c)^2 - 12 a b (a b c + 3 a b - a c - c)))/(6 a b)

substitue x into to original cubic

(-2 a b c + sqrt((2 a b c + 6 a b + 2 a c)^2 - 12 a b (a b c + 3 a b - a c - c)) - 6 a b - 2 a c)^3/(216 a^2 b^2) + ((a b (c + 3) + a c) (-2 a b c + sqrt((2 a b c + 6 a b + 2 a c)^2 - 12 a b (a b c + 3 a b - a c - c)) - 6 a b - 2 a c)^2)/(36 a^2 b^2) + ((a b (c + 3) - (a + 1) c) (-2 a b c + sqrt((2 a b c + 6 a b + 2 a c)^2 - 12 a b (a b c + 3 a b - a c - c)) - 6 a b - 2 a c))/(6 a b) + a b + c = 0

a b + c + ((-((1 + a) c) + a b (3 + c)) (-6 a b - 2 a c - 2 a b c + Sqrt[-12 a b (3 a b - c - a c + a b c) + (6 a b + 2 a c + 2 a b c)^2]))/(6 a b) + ((a c + a b (3 + c)) (-6 a b - 2 a c - 2 a b c + Sqrt[-12 a b (3 a b - c - a c + a b c) + (6 a b + 2 a c + 2 a b c)^2])^2)/(36 a^2 b^2) + (-6 a b - 2 a c - 2 a b c + Sqrt[-12 a b (3 a b - c - a c + a b c) + (6 a b + 2 a c + 2 a b c)^2])^3/(216 a^2 b^2)


substitue instead with b = 2* 2/7 to simplify equations


c - c x + (a (4 - 3 (-4 + c) x + (12 + 11 c) x^2 + 4 x^3))/7

(-7 c + 12 a (1 + x)^2 + a c (-3 + 22 x))/7

positive root of quadratic

x = (7 ((-(22 a c)/7 - (24 a)/7) + sqrt(((22 a c)/7 + (24 a)/7)^2 - 48/7 a (-(3 a c)/7 + (12 a)/7 - c))))/(24 a)

subsitutuing back into cubic


(c (-84 Sqrt[a c (84 + 300 a + 121 a c)] + a^2 (3024 + 4950 c + 1331 c^2) - a (-3024 + 300 Sqrt[a c (84 + 300 a + 121 a c)] + 11 c (-126 + 11 Sqrt[a c (84 + 300 a + 121 a c)]))))/(1512 a)


(c (a^2 (1331 c^2 + 4950 c + 3024) - a (11 c (11 sqrt(a c (121 a c + 300 a + 84)) - 126) + 300 sqrt(a c (121 a c + 300 a + 84)) - 3024) - 84 sqrt(a c (121 a c + 300 a + 84))))/(1512 a)


substitute instead for one fluid dominating pressure drop entropy creation b = 2/7

c - c x + (a (2 + (6 - 5 c) x + (6 + 9 c) x^2 + 2 x^3))/7
(-7 c + 6 a (1 + x)^2 + a c (-5 + 18 x))/7

positive root of quadratic
x = (7 ((-(18 a c)/7 - (12 a)/7) + sqrt(((18 a c)/7 + (12 a)/7)^2 - 24/7 a (-(5 a c)/7 + (6 a)/7 - c))))/(12 a)

substituting back into cubic
(c (9 a^2 (27 c^2 + 69 c + 28) - a (27 c (sqrt(3) sqrt(a c (27 a c + 46 a + 14)) - 7) + 46 sqrt(3) sqrt(a c (27 a c + 46 a + 14)) - 252) - 14 sqrt(3) sqrt(a c (27 a c + 46 a + 14))))/(126 a)

(c (-14 Sqrt[3] Sqrt[a c (14 + 46 a + 27 a c)] + 9 a^2 (28 + 69 c + 27 c^2) - a (-252 + 46 Sqrt[3] Sqrt[a c (14 + 46 a + 27 a c)] + 27 c (-7 + Sqrt[3] Sqrt[a c (14 + 46 a + 27 a c)]))))/(126 a)


"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

# Parameter sweep for t and corresponding c values
t = np.linspace(1.1, 4, 10)
c_arr = (t - 1) ** 2 / t


def f_double_fluid(a, c):
    """
    Function f(a,c) as given; returns f for arrays or scalars.
    """
    sqrt_expr = np.sqrt(a * c * (121 * a * c + 300 * a + 84))
    num = c * (
        a**2 * (1331 * c**2 + 4950 * c + 3024)
        - a * (11 * c * (11 * sqrt_expr - 126) + 300 * sqrt_expr - 3024)
        - 84 * sqrt_expr
    )
    den = 1512 * a
    return num / den


def f_single_fluid(a, c):
    """
    Function f_single_fluid(a, c) for the one-fluid limit; returns f for arrays or scalars.
    """
    sqrt_expr = np.sqrt(3 * a * c * (27 * a * c + 46 * a + 14))
    num = c * (
        9 * a**2 * (27 * c**2 + 69 * c + 28) - a * (27 * c * (sqrt_expr - 7) + 46 * sqrt_expr - 252) - 14 * sqrt_expr
    )
    den = 126 * a
    return num / den


f = f_single_fluid

# Choose a range for a: positive between 10^-1 and 10^-3.5 (~0.000316)
a_min = 10**-3.5
a_max = 0.2

roots = []

for cval in c_arr:
    # for each c, find where f(a, c) == 0 in the interval
    # Evaluate on a logarithmic grid for plotting / diagnostics
    a_grid = np.logspace(np.log10(a_min), np.log10(a_max), 200)
    y_grid = f(a_grid, cval)

    # Find sign changes in f(a, cval) for possible roots
    sign_changes = np.where(np.diff(np.sign(y_grid)))[0]

    found_root = None
    for idx in sign_changes:
        a_lo = a_grid[idx]
        a_hi = a_grid[idx + 1]
        try:
            root = brentq(f, a_lo, a_hi, args=(cval,), maxiter=500)
            if root > 0:
                found_root = root
                break  # Take the first positive root found
        except Exception:
            continue

    roots.append(found_root)


plt.scatter(t, roots, label="Roots")


# Remove None values for fitting
t_fit = np.array([ti for ti, ri in zip(t, roots, strict=True) if ri is not None])
roots_fit = np.array([ri for ri in roots if ri is not None])

# Linear fit: roots = m * t + b
coeffs = np.polyfit(t_fit, roots_fit, 1)
m, b = coeffs

# Predicted values and R^2 computation
roots_pred = m * t_fit + b
ss_res = np.sum((roots_fit - roots_pred) ** 2)
ss_tot = np.sum((roots_fit - np.mean(roots_fit)) ** 2)
r_squared = 1 - ss_res / ss_tot

# Plot fit
t_line = np.linspace(min(t), max(t), 100)
roots_line = m * t_line + b
plt.plot(t_line, roots_line, "r--", label=f"Fit: a = {m:.3g} t + {b:.3g}\n$R^2$ = {r_squared:.4f}")

plt.legend()

plt.xlabel("t")
plt.ylabel("a")
plt.title("a vs t")
plt.show()
