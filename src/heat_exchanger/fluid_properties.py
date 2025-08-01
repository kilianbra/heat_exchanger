import sys

from abc import ABC, abstractmethod

import numpy as np


class FluidPropertiesStrategy(ABC):
    """Abstract strategy for fluid thermophysical property calculations."""

    @abstractmethod
    def get_density(self, T: float, P: float) -> float:
        """Return fluid density (kg/m^3) at temperature T (K) and pressure P (Pa)."""
        pass

    @abstractmethod
    def get_cp(self, T: float, P: float) -> float:
        """Return specific heat capacity (J/kg.K) at temperature T and pressure P."""
        pass

    @abstractmethod
    def get_specific_enthalpy(self, T: float, P: float) -> float:
        """Return specific enthalpy (J/kg) at temperature T and pressure P."""
        pass

    @abstractmethod
    def get_viscosity(self, T: float, P: float) -> float:
        """Return dynamic viscosity (Pa.s) at temperature T and pressure P."""
        pass

    @abstractmethod
    def get_thermal_conductivity(self, T: float, P: float) -> float:
        """Return thermal conductivity (W/m.K) at temperature T and pressure P."""
        pass


class PerfectGasProperties(FluidPropertiesStrategy):
    """
    Perfect gas model with constant specific heat.
    Uses Sutherland's formula for viscosity and assumes constant Prandtl number.
    """

    def __init__(
        self,
        molecular_weight: float,
        gamma: float,
        Pr: float = 0.7,
        mu_ref: float = 1.8e-5,
        T_ref: float = 300.0,
        S: float = 110.4,
    ):
        """
        :param molecular_weight: Molecular weight of gas (kg/kmol) to compute specific gas constant.
        :param cp: Specific heat capacity at constant pressure (J/kg.K) (assumed constant).
        :param Pr: Prandtl number (assumed constant).
        :param mu_ref: Reference viscosity at T_ref (Pa.s).
        :param T_ref: Reference temperature for viscosity (K).
        :param S: Sutherland's constant (K) for viscosity calculation.
        """
        self.R_specific = 8314.462618 / molecular_weight  # specific gas constant J/(kg.K)
        self.gamma = gamma
        self.cp = self.R_specific * gamma / (gamma - 1)
        self.Pr = Pr
        # viscosity parameters for Sutherland's law
        self.mu_ref = mu_ref
        self.T_ref = T_ref
        self.S = S

    def get_transport_properties(self, T: float, P: float) -> tuple:
        # Calculate all properties at once
        rho = P / (self.R_specific * T)
        cp = self.cp
        mu = self.mu_ref * ((self.T_ref + self.S) / (T + self.S)) * ((T / self.T_ref) ** 1.5)
        k = mu * cp / self.Pr
        return rho, cp, mu, k

    def get_density(self, T: float, P: float) -> float:
        # Ideal gas law: rho = P / (R_specific * T)
        return P / (self.R_specific * T)

    def get_speed_of_sound(self, T: float, P: float) -> float:
        # assume ideal gas
        return (self.gamma * self.R_specific * T) ** 0.5

    def get_cp(self, T: float, P: float) -> float:
        # Constant cp (J/kg.K)
        return self.cp

    def get_specific_enthalpy(self, T: float, P: float) -> float:
        # assume enthalpy is zero at zero temperature
        return self.cp * T

    def get_viscosity(self, T: float, P: float) -> float:
        # Sutherland's law for viscosity variation with temperature
        # mu = mu_ref * (T_ref + S) / (T + S) * (T / T_ref)^(3/2)
        return self.mu_ref * ((self.T_ref + self.S) / (T + self.S)) * ((T / self.T_ref) ** 1.5)

    def get_thermal_conductivity(self, T: float, P: float) -> float:
        # Assume constant Prandtl -> k = mu * cp / Pr
        mu = self.get_viscosity(T, P)
        return mu * self.cp / self.Pr

    def get_specific_entropy(self, T: float, P: float) -> float:
        # assume entropy is zero at zero T=300K, P=101325Pa
        return self.cp * np.log(T / 300) - self.R_specific * np.log(P / 101325)

    def get_hse_and_Tse_p_T(self, P: float, T: float, pd: float) -> float:
        # get isentropic exit temperature from isentropic expansion
        Tse = T * (pd / P) ** ((self.gamma - 1) / self.gamma)
        return self.get_specific_enthalpy(Tse, pd), Tse

    def get_hse_and_Tse_s(self, s: float, pd: float, Td: float) -> float:
        # get isentropic exit temperature from isobaric entropy increase
        Tse = Td * np.exp((s - self.get_specific_entropy(Td, pd)) / self.cp)
        return self.get_specific_enthalpy(Tse, pd), Tse
