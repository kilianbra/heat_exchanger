from __future__ import annotations

import typing
from dataclasses import dataclass
from functools import cached_property

import numpy as np


class FluidState(typing.Protocol):
    """
    Thermodynamic state for a fluid at a specified temperature and pressure.

    Units:
      - T [K], P [Pa]
      - rho [kg/m^3]
      - cp [J/(kg·K)]
      - mu [Pa·s]
      - k  [W/(m·K)]
      - h  [J/kg]  (specific enthalpy)
      - s  [J/(kg·K)] (specific entropy; reference defined by model)
      - a  [m/s]  (speed of sound)
    """

    @property
    def T(self) -> float: ...

    @property
    def P(self) -> float: ...

    @property
    def rho(self) -> float: ...

    @property
    def cp(self) -> float: ...

    @property
    def mu(self) -> float: ...

    @property
    def k(self) -> float: ...

    @property
    def h(self) -> float: ...

    @property
    def s(self) -> float: ...

    @property
    def a(self) -> float: ...


class FluidModel(typing.Protocol):
    """
    Fluid model capable of producing a state at (T, P).

    For now, states are defined strictly by temperature and pressure; additional
    state definitions (e.g., (p, h)) can be added later without changing callers
    that use this interface.
    """

    def state(self, T: float, P: float) -> FluidState: ...


class PerfectGasFluid:
    """
    Perfect gas model with constant specific heat and Sutherland viscosity.

    Parameters
    ----------
    molecular_weight : float
        Molecular weight [kg/kmol] used to compute R_specific.
    gamma : float, default 1.4
        Ratio of specific heats (cp/cv). Model-level parameter.
    Pr : float, default 0.7
        Prandtl number (assumed constant). Model-level parameter.
    mu_ref : float, default 1.8e-5
        Reference dynamic viscosity [Pa·s] at T_ref (Sutherland law).
    T_ref : float, default 300.0
        Reference temperature [K] for Sutherland law.
    S : float, default 110.4
        Sutherland's constant [K].

    Notes
    -----
    - Model parameters (gamma, Pr, mu_ref, T_ref, S, R_specific) are properties of the
        model and are not stored on the state; the state exposes only thermodynamic
        properties at (T, P) and uses cached properties to avoid redundant computation.
    - Entropy reference is s = 0 at T0 = 300 K and P0 = 101325 Pa.
    -Sutherland's law constants for common gases:

      Gas      S (K)     T_ref (K)    mu_ref (Pa.s)
    ---------------------------------------------------------------
      Air      111       273          1.716e-5
      N2       107       273          1.663e-5
      H2        97       273          8.411e-6
      Steam   1064       350          1.12e-5
      Helium    79.4     273          1.9e-5

    Reference: https://doc.comsol.com/6.2/doc/com.comsol.help.cfd/cfd_ug_fluidflow_high_mach.08.41.html
    except Helium from German Wikipedia: https://de.wikipedia.org/wiki/Sutherland-Modell
    """

    R_UNIVERSAL = 8314.462618  # J/(kmol·K)

    def __init__(
        self,
        molecular_weight: float,
        *,
        gamma: float = 1.4,
        Pr: float = 0.7,
        mu_ref: float = 1.8e-5,
        T_ref: float = 300.0,
        S: float = 110.4,
    ) -> None:
        self.R_specific = self.R_UNIVERSAL / molecular_weight  # J/(kg·K)
        self.gamma = gamma
        self.Pr = Pr
        self.mu_ref = mu_ref
        self.T_ref = T_ref
        self.S = S

    def state(self, T: float, P: float) -> FluidState:
        return _PerfectGas(T=T, P=P, _model=self)


@dataclass(frozen=True)
class _PerfectGas:
    """
    Perfect gas thermodynamic state at (T, P).

    Properties are provided as cached properties and depend only on (T, P) and
    the associated model's immutable parameters.
    """

    T: float  # K
    P: float  # Pa
    _model: PerfectGasFluid

    @cached_property
    def rho(self) -> float:
        """Density [kg/m^3] from ideal gas law: rho = P / (R_specific * T)."""
        return self.P / (self._model.R_specific * self.T)

    @cached_property
    def cp(self) -> float:
        """Specific heat at constant pressure cp [J/(kg·K)] (constant)."""
        return self._model.R_specific * self._model.gamma / (self._model.gamma - 1.0)

    @cached_property
    def mu(self) -> float:
        """Dynamic viscosity mu [Pa·s] via Sutherland's law."""
        m = self._model
        return m.mu_ref * ((m.T_ref + m.S) / (self.T + m.S)) * ((self.T / m.T_ref) ** 1.5)

    @cached_property
    def k(self) -> float:
        """Thermal conductivity k [W/(m·K)] via constant Pr: k = mu * cp / Pr."""
        m = self._model
        return self.mu * self.cp / m.Pr

    @cached_property
    def h(self) -> float:
        """Specific enthalpy h [J/kg] assuming h = 0 at T = 0 K: h = cp · T."""
        return self.cp * self.T

    @cached_property
    def s(self) -> float:
        """
        Specific entropy s [J/(kg·K)] with reference s=0 at T0=300 K, P0=101325 Pa:
            s = cp · ln(T/T0) - R_specific · ln(P/P0)
        """
        m = self._model
        T0 = 300.0
        P0 = 101325.0
        return self.cp * np.log(self.T / T0) - m.R_specific * np.log(self.P / P0)

    @cached_property
    def a(self) -> float:
        """Speed of sound a [m/s] for ideal gas: a = sqrt(gamma · R_specific · T)."""
        m = self._model
        return np.sqrt(m.gamma * m.R_specific * self.T)


__all__ = [
    "FluidState",
    "FluidModel",
    "PerfectGasFluid",
]
