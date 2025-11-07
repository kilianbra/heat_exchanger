import logging
import os
import sys
from abc import ABC, abstractmethod

import CoolProp.CoolProp as CP
import numpy as np

logger = logging.getLogger(__name__)

_REFPROP_CONFIGURED = False


def configure_refprop() -> None:
    """Configure REFPROP path once; safe to call multiple times."""

    global _REFPROP_CONFIGURED
    if _REFPROP_CONFIGURED:
        return

    try:
        refprop_path = os.environ.get("REFPROP_PATH") or os.environ.get("RPPREFIX")
        candidate_paths: list[str] = []

        if not refprop_path:
            if sys.platform.startswith("win"):
                candidate_paths = [
                    r"C:\\Program Files\\REFPROP",
                    r"C:\\Program Files (x86)\\REFPROP",
                ]
            elif sys.platform == "darwin":
                candidate_paths = [
                    "/Applications/REFPROP",
                    "/usr/local/REFPROP",
                ]
            elif sys.platform.startswith("linux"):
                candidate_paths = [
                    "/usr/local/share/REFPROP",
                    "/opt/REFPROP",
                ]
            for candidate in candidate_paths:
                if os.path.isdir(candidate):
                    refprop_path = candidate
                    break

        if refprop_path:
            CP.set_config_string(CP.ALTERNATIVE_REFPROP_PATH, refprop_path)
            logger.info(
                "REFPROP path set to: %s",
                CP.get_config_string(CP.ALTERNATIVE_REFPROP_PATH),
            )
        else:
            logger.debug("REFPROP path not set; using CoolProp defaults.")

    except Exception as exc:  # pragma: no cover - defensive path setup
        logger.warning(
            "Failed to configure REFPROP path: %s. Using CoolProp defaults. Current path: %s",
            exc,
            CP.get_config_string(CP.ALTERNATIVE_REFPROP_PATH),
        )

    _REFPROP_CONFIGURED = True


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

    Sutherland's law constants for common gases:

      Gas      S (K)     T_ref (K)    mu_ref (Pa.s)
    ---------------------------------------------------------------
      Air      111       273          1.716e-5
      N2       107       273          1.663e-5
      H2        97       273          8.411e-6
      Steam   1064       350          1.12e-5
      Helium    79.4     273          1.9e-5

    Reference: https://doc.comsol.com/6.2/doc/com.comsol.help.cfd/cfd_ug_fluidflow_high_mach.08.41.html
    except Helium use https://de.wikipedia.org/wiki/Sutherland-Modell
    """

    def __init__(
        self,
        molecular_weight: float,
        gamma: float = 1.4,
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

    def get_isentropic_exit_h_and_temp_from_p_temp(self, P: float, T: float, pd: float) -> float:
        # get isentropic exit temperature from isentropic expansion
        Tse = T * (pd / P) ** ((self.gamma - 1) / self.gamma)
        return self.get_specific_enthalpy(Tse, pd), Tse

    def get_isentropic_exit_h_and_temp_from_s(self, s: float, pd: float, Td: float) -> float:
        # get isentropic exit temperature from isobaric entropy increase
        Tse = Td * np.exp((s - self.get_specific_entropy(Td, pd)) / self.cp)
        return self.get_specific_enthalpy(Tse, pd), Tse


class CoolPropProperties(FluidPropertiesStrategy):
    """
    Fluid properties from CoolProp library. Requires CoolProp to be installed.
    """

    def __init__(self, fluid_name: str):
        configure_refprop()
        self.fluid = fluid_name
        try:
            self.CP = CP
            self.AS = CP.AbstractState("HEOS", fluid_name)
        except ImportError as err:
            raise ImportError("CoolProp library is required for CoolPropProperties") from err

    def get_transport_properties(self, T: float, P: float) -> tuple:
        self.AS.update(self.CP.PT_INPUTS, P, T)
        rho = self.AS.rhomass()
        cp = self.AS.cpmass()
        mu = self.AS.viscosity()
        k = self.AS.conductivity()
        return rho, cp, mu, k

    def get_density(self, T: float, P: float) -> float:
        return self.CP.PropsSI("D", "T", T, "P", P, self.fluid)

    def get_cp(self, T: float, P: float) -> float:
        return self.CP.PropsSI("Cpmass", "T", T, "P", P, self.fluid)

    def get_specific_enthalpy(self, T: float, P: float) -> float:
        return self.CP.PropsSI("Hmass", "T", T, "P", P, self.fluid)

    def get_viscosity(self, T: float, P: float) -> float:
        return self.CP.PropsSI("V", "T", T, "P", P, self.fluid)

    def get_thermal_conductivity(self, T: float, P: float) -> float:
        return self.CP.PropsSI("L", "T", T, "P", P, self.fluid)

    def get_specific_entropy(self, T: float, P: float) -> float:
        return self.CP.PropsSI("Smass", "T", T, "P", P, self.fluid) - self.CP.PropsSI(
            "Smass", "T", 300, "P", 101325, self.fluid
        )

    def get_speed_of_sound(self, T: float, P: float) -> float:
        return self.CP.PropsSI("A", "T", T, "P", P, self.fluid)

    def get_isentropic_exit_h_and_temp_from_p_temp(self, P: float, T: float, pd: float) -> float:
        # get isentropic exit temperature from isentropic expansion
        self.AS.update(self.CP.PT_INPUTS, P, T)
        s = self.AS.smass()
        self.AS.update(self.CP.PSmass_INPUTS, pd, s)
        return self.AS.hmass(), self.AS.T()

    def get_isentropic_exit_h_and_temp_from_s(self, s: float, pd: float, Td: float) -> float:
        # get isentropic exit temperature from isobaric entropy increase
        self.AS.update(self.CP.PT_INPUTS, pd, Td)
        _sd = self.AS.smass()
        self.AS.update(self.CP.PSmass_INPUTS, pd, s)
        return self.AS.hmass(), self.AS.T()


class RefPropProperties(FluidPropertiesStrategy):
    """
    Fluid properties from REFPROP (via CoolProp REFPROP backend if available).
    """

    def __init__(self, fluid_name: str):
        configure_refprop()
        self.fluid = fluid_name
        try:
            # Try to create REFPROP AbstractState
            self.CP = CP
            try:
                self.AS = CP.AbstractState("REFPROP", fluid_name)
            except ValueError as e:
                # If direct fluid name fails, try with standardized fluid name
                fluid_map = {
                    "H2O": "WATER",
                    "N2": "NITROGEN",
                    "O2": "OXYGEN",
                    "H2": "HYDROGEN",
                    "CO2": "CO2",
                }
                if fluid_name in fluid_map:
                    self.AS = CP.AbstractState("REFPROP", fluid_map[fluid_name])
                else:
                    raise e

        except Exception as e:
            error_msg = [
                f"Error initializing REFPROP for fluid '{fluid_name}':",
                f"Original error: {str(e)}",
                f"Current REFPROP path: {CP.get_config_string(CP.ALTERNATIVE_REFPROP_PATH)}",  # ,
                # f"Available backends: {CP.get_global_param_string('fluids_list')}"
            ]

            raise ImportError(f"REFPROP initialization failed: {'\n'.join(error_msg)}") from e

    def get_transport_properties(self, T: float, P: float) -> tuple:
        self.AS.update(self.CP.PT_INPUTS, P, T)
        rho = self.AS.rhomass()
        cp = self.AS.cpmass()
        mu = self.AS.viscosity()
        k = self.AS.conductivity()
        return rho, cp, mu, k

    def get_density(self, T: float, P: float) -> float:
        return self.CP.PropsSI("D", "T", T, "P", P, self.fluid)

    def get_cp(self, T: float, P: float) -> float:
        return self.CP.PropsSI("Cpmass", "T", T, "P", P, self.fluid)

    def get_specific_enthalpy(self, T: float, P: float) -> float:
        return self.CP.PropsSI("Hmass", "T", T, "P", P, self.fluid)

    def get_viscosity(self, T: float, P: float) -> float:
        return self.CP.PropsSI("V", "T", T, "P", P, self.fluid)

    def get_thermal_conductivity(self, T: float, P: float) -> float:
        return self.CP.PropsSI("L", "T", T, "P", P, self.fluid)

    def get_specific_entropy(self, T: float, P: float) -> float:
        return self.CP.PropsSI("Smass", "T", T, "P", P, self.fluid) - self.CP.PropsSI(
            "Smass", "T", 300, "P", 101325, self.fluid
        )

    def get_speed_of_sound(self, T: float, P: float) -> float:
        return self.CP.PropsSI("A", "T", T, "P", P, self.fluid)

    def get_hse_and_Tse_p_T(self, P: float, T: float, pd: float) -> float:
        # get isentropic exit temperature from isentropic expansion
        self.AS.update(self.CP.PT_INPUTS, P, T)
        s = self.AS.smass()
        self.AS.update(self.CP.PSmass_INPUTS, pd, s)
        return self.AS.hmass(), self.AS.T()

    def get_hse_and_Tse_s(self, s: float, pd: float, Td: float) -> float:
        # get isentropic exit temperature from isobaric entropy increase
        self.AS.update(self.CP.PT_INPUTS, pd, Td)
        # sd = self.AS.smass()
        self.AS.update(self.CP.PSmass_INPUTS, pd, s)
        return self.AS.hmass(), self.AS.T()


class MixtureProperties(FluidPropertiesStrategy):
    """
    Mixture fluid properties with REFPROP as primary backend and CoolProp as fallback.
    Handles high water content mixtures using correlation approach when needed.

    For molar water content <= 5% can use REFPROP directly to get transport properties.
    If molar water content > 5%, REFPROP can return rho and c_p but not viscosity and conductivity.
    Hence use a correlation for polar mixtures to get viscosity and conductivity.
    The correlation is based on a constant where the default value of 3.5 is unsatisfactory
    so hence a better value is derived by checking the transport properties of a scaled mixture
    with 4.5% vol water fraction. So if 9% H2O and 93% N2 by vol, then instead will calculate
    the transport properties of 4.5% H2O and 95.5% N2 and determine the constant from applying
    the correlation at that value. This ensures relative continuity with respect to FAR.
    If Refprop is not available or fails, uses CoolProps HAPropsSI for humid air at T < 350 C.
    At higher temperatures just defaults to returning the properties of air.

    Note that therefore this should mainly be used for air like mixtures
    """

    def __init__(self, components: list, mole_fractions: list, prefer_refprop: bool = True):
        configure_refprop()
        """
        Initialize mixture properties.

        Args:
            components: List of component names (e.g., ['H2O', 'N2', 'O2'])
            mole_fractions: List of mole fractions corresponding to components
            prefer_refprop: If True, try REFPROP first, otherwise use CoolProp
        """
        self.components = components
        self.mole_fractions = np.array(mole_fractions)
        self.prefer_refprop = prefer_refprop
        self.CP = CP

        # Normalize mole fractions
        self.mole_fractions = self.mole_fractions / np.sum(self.mole_fractions)

        # Molar masses for calculations
        self.molar_masses = {
            "H2O": 18.015,
            "N2": 28.013,
            "O2": 31.999,
            "CO2": 44.01,
            "H2": 2.016,
            "Air": 28.97,
        }

        # Calculate water mole fraction for correlation, which deifferentiates polar components.
        self.x_H2O = 0.0
        if "H2O" in components:
            h2o_idx = components.index("H2O")
            self.x_H2O = mole_fractions[h2o_idx]

        # Initialize backend states
        self._init_backends()

    def _init_backends(self):
        """Initialize REFPROP and CoolProp backends"""
        self.refprop_available = False
        self.mixture_state = None
        self.dry_state = None
        self.wet_state = None

        if self.prefer_refprop:
            try:
                # Try REFPROP mixture
                component_string = "&".join(self.components)
                self.mixture_state = self.CP.AbstractState("REFPROP", component_string)
                self.mixture_state.set_mole_fractions(self.mole_fractions)
                self.refprop_available = True

                # For high water content, prepare dry and wet states for correlation
                if self.x_H2O > 0.05:
                    self._init_correlation_states()

            except Exception as e:
                logger.debug(f"REFPROP mixture initialization failed: {e}")
                self.refprop_available = False

        # CoolProp fallback initialization
        if not self.refprop_available:
            self._init_coolprop_fallback()

    def _init_correlation_states(self):
        """Initialize separate dry and wet states for correlation approach"""
        try:
            # Create dry components (everything except water)
            dry_components = [comp for comp in self.components if comp != "H2O"]
            if dry_components:
                dry_component_string = "&".join(dry_components)
                self.dry_state = self.CP.AbstractState("REFPROP", dry_component_string)

                # Calculate dry component fractions (normalized without water)
                dry_fractions = []
                for comp in dry_components:
                    idx = self.components.index(comp)
                    dry_fractions.append(self.mole_fractions[idx])

                if sum(dry_fractions) > 0:
                    dry_fractions = np.array(dry_fractions) / sum(dry_fractions)
                    self.dry_state.set_mole_fractions(dry_fractions)

            # Create wet state (pure water)
            self.wet_state = self.CP.AbstractState("REFPROP", "WATER")

        except Exception as e:
            logger.debug(f"Warning: Could not initialize correlation states: {e}")
            self.dry_state = None
            self.wet_state = None

    def _init_coolprop_fallback(self):
        """Initialize CoolProp fallback approach"""
        # Calculate mass fraction of water for HAPropsSI
        if self.x_H2O > 0:
            total_mass = sum(
                self.mole_fractions[i] * self.molar_masses.get(comp, 28.97)
                for i, comp in enumerate(self.components)
            )
            h2o_mass = self.x_H2O * self.molar_masses["H2O"]
            self.w_H2O = h2o_mass / total_mass
        else:
            self.w_H2O = 0

    def _get_correlation_vectors(self, T: float, P: float):
        """Calculate correlation vectors for high water content mixtures"""
        if not (self.dry_state and self.wet_state) or self.x_H2O <= 0.05:
            return np.array([3.5, 3.5])  # Default values

        try:
            # Update mixture state at reference water content (4.5%)
            x_H2O_ref = 0.045
            ref_fractions = self.mole_fractions.copy()
            # h2o_idx = self.components.index("H2O")

            # Adjust fractions for reference case
            scale_factor = (1 - x_H2O_ref) / (1 - self.x_H2O)
            for i, comp in enumerate(self.components):
                if comp != "H2O":
                    ref_fractions[i] *= scale_factor
                else:
                    ref_fractions[i] = x_H2O_ref

            ref_mixture = self.CP.AbstractState("REFPROP", "&".join(self.components))
            ref_mixture.set_mole_fractions(ref_fractions)
            ref_mixture.update(self.CP.PT_INPUTS, P, T)

            # Update component states
            self.dry_state.update(self.CP.PT_INPUTS, P * (1 - x_H2O_ref), T)
            self.wet_state.update(self.CP.PT_INPUTS, P * x_H2O_ref, T)

            # Get transport properties
            properties = [self.CP.iviscosity, self.CP.iconductivity]
            vect_mix = np.array([ref_mixture.keyed_output(k) for k in properties])
            vect_dry = np.array([self.dry_state.keyed_output(k) for k in properties])
            vect_wet = np.array([self.wet_state.keyed_output(k) for k in properties])

            # Calculate correlation vectors
            denominator = vect_mix / (x_H2O_ref * vect_wet + (1 - x_H2O_ref) * vect_dry) - 1
            correlation_vectors = (x_H2O_ref * (1 - x_H2O_ref)) / denominator

            return correlation_vectors

        except Exception as e:
            logger.debug(f"Warning: Correlation calculation failed: {e}")
            return np.array([3.5, 3.5])  # Default fallback

    def get_transport_properties(self, T: float, P: float) -> tuple:
        """Get all transport properties at once"""
        if self.refprop_available:
            try:
                self.mixture_state.update(self.CP.PT_INPUTS, P, T)
                rho = self.mixture_state.rhomass()
                cp = self.mixture_state.cpmass()

                # For high water content (refprop only gets transport properties if x_H2O< 5%),
                # use correlation for viscosity and conductivity
                if self.x_H2O > 0.05 and self.dry_state and self.wet_state:
                    correlation_vectors = self._get_correlation_vectors(T, P)

                    # Update component states at partial pressures
                    self.dry_state.update(self.CP.PT_INPUTS, P * (1 - self.x_H2O), T)
                    self.wet_state.update(self.CP.PT_INPUTS, P * self.x_H2O, T)

                    # Calculate properties using correlation
                    mu_dry = self.dry_state.viscosity()
                    mu_wet = self.wet_state.viscosity()
                    k_dry = self.dry_state.conductivity()
                    k_wet = self.wet_state.conductivity()

                    mu = (self.x_H2O * mu_wet + (1 - self.x_H2O) * mu_dry) * (
                        1 + (self.x_H2O - self.x_H2O**2) / correlation_vectors[0]
                    )
                    k = (self.x_H2O * k_wet + (1 - self.x_H2O) * k_dry) * (
                        1 + (self.x_H2O - self.x_H2O**2) / correlation_vectors[1]
                    )
                else:
                    # Use direct mixture properties
                    mu = self.mixture_state.viscosity()
                    k = self.mixture_state.conductivity()

                return rho, cp, mu, k

            except Exception as e:
                raise RuntimeError(f"REFPROP calculation failed: {e}") from e
                # Fall through to CoolProp

        # CoolProp fallback
        return self._get_coolprop_properties(T, P)

    def _get_coolprop_properties(self, T: float, P: float) -> tuple:
        """Get properties using CoolProp fallback approach"""
        if self.x_H2O > 0 and T < 623.15:
            # Use HAPropsSI for humid air at low temperatures
            try:
                v_ha = self.CP.HAPropsSI("Vha", "T", T, "P", P, "W", self.w_H2O)
                rho = (1.0 + self.w_H2O) / v_ha

                cp_ha = self.CP.HAPropsSI("C", "T", T, "P", P, "W", self.w_H2O)
                cp = cp_ha / (1 + self.w_H2O)

                mu = self.CP.HAPropsSI("M", "T", T, "P", P, "W", self.w_H2O)
                k = self.CP.HAPropsSI("K", "T", T, "P", P, "W", self.w_H2O)

                return rho, cp, mu, k
            except Exception as e:
                raise RuntimeError(f"HAPropsSI failed: {e}") from e

        # Use dry air as final fallback
        air = self.CP.AbstractState("HEOS", "Air")
        air.update(self.CP.PT_INPUTS, P, T)
        return air.rhomass(), air.cpmass(), air.viscosity(), air.conductivity()

    def get_density(self, T: float, P: float) -> float:
        rho, _, _, _ = self.get_transport_properties(T, P)
        return rho

    def get_cp(self, T: float, P: float) -> float:
        _, cp, _, _ = self.get_transport_properties(T, P)
        return cp

    def get_viscosity(self, T: float, P: float) -> float:
        _, _, mu, _ = self.get_transport_properties(T, P)
        return mu

    def get_thermal_conductivity(self, T: float, P: float) -> float:
        _, _, _, k = self.get_transport_properties(T, P)
        return k

    def get_specific_enthalpy(self, T: float, P: float) -> float:
        """Get specific enthalpy using low-level interface calls"""
        if self.refprop_available:
            try:
                self.mixture_state.update(self.CP.PT_INPUTS, P, T)
                return self.mixture_state.hmass()
            except Exception as e:
                raise RuntimeError(f"REFPROP enthalpy calculation failed: {e}") from e
                # Fall through to CoolProp

        # CoolProp fallback
        if self.x_H2O > 0 and T < 623.15:
            try:
                # Use HAPropsSI for humid air
                h_ha = self.CP.HAPropsSI("Hha", "T", T, "P", P, "W", self.w_H2O)
                return h_ha / (1 + self.w_H2O)  # Convert to per kg mixture
            except Exception as e:
                raise RuntimeError(f"HAPropsSI enthalpy calculation failed: {e}") from e

        # Final fallback to dry air
        air = self.CP.AbstractState("HEOS", "Air")
        air.update(self.CP.PT_INPUTS, P, T)
        return air.hmass()

    def get_specific_entropy(self, T: float, P: float) -> float:
        """Get specific entropy using low-level interface calls"""
        if self.refprop_available:
            try:
                self.mixture_state.update(self.CP.PT_INPUTS, P, T)
                return self.mixture_state.smass()
            except Exception as e:
                logger.debug(f"REFPROP entropy calculation failed: {e}")
                # Fall through to CoolProp

        # CoolProp fallback
        if self.x_H2O > 0 and T < 623.15:
            try:
                # Use HAPropsSI for humid air
                s_ha = self.CP.HAPropsSI("Sha", "T", T, "P", P, "W", self.w_H2O)
                return s_ha / (1 + self.w_H2O)  # Convert to per kg mixture
            except Exception as e:
                logger.debug(f"HAPropsSI entropy calculation failed: {e}")

        # Final fallback to dry air
        air = self.CP.AbstractState("HEOS", "Air")
        air.update(self.CP.PT_INPUTS, P, T)
        return air.smass()


class CombustionProductsProperties(FluidPropertiesStrategy):
    """
    Combustion products fluid properties calculated from fuel-air ratio.
    Supports hydrogen and hydrocarbon (C0.92H2) fuels.
    """

    def __init__(self, fuel_type: str, FAR_mass: float, prefer_refprop: bool = True):
        """
        Initialize combustion products properties.

        Args:
            fuel_type: 'H2' for hydrogen or 'C092H2' for hydrocarbon
            FAR_mass: Fuel-to-air mass ratio (mdot_fuel/mdot_dry_air)
            prefer_refprop: If True, prefer REFPROP over CoolProp

        If FAR < 1e-6 just model as 21 % vol O2 and 79 % vol N2
        Else consider combustion balance for one mole of fuel.
        Pass on to MixtureProperties object to actually calculate properties.
        """
        self.fuel_type = fuel_type
        self.FAR_mass = FAR_mass
        self.prefer_refprop = prefer_refprop

        # Calculate composition
        self.components, self.mole_fractions = self._calculate_composition()

        # Create underlying mixture properties object
        self.mixture = MixtureProperties(self.components, self.mole_fractions, prefer_refprop)
        self.mixture_state = getattr(self.mixture, "mixture_state", None)

    def _calculate_composition(self):
        """Calculate mole fractions from fuel-air ratio"""
        if self.FAR_mass < 1e-6:
            # Dry air
            return ["N2", "O2"], [0.79, 0.21]

        # Molar masses
        M = {
            "O2": 31.999,
            "N2": 28.013,
            "H2O": 18.015,
            "H2": 2.016,
            "CO2": 44.01,
            "C092H2": 12.011 * 0.92 + 1.008 * 2,
        }

        M_air = 0.21 * M["O2"] + 0.79 * M["N2"]

        if self.fuel_type == "H2":
            return self._h2_combustion_products(M, M_air)
        elif self.fuel_type == "C092H2":
            return self._hydrocarbon_combustion_products(M, M_air)
        else:
            raise ValueError(f"Unsupported fuel type: {self.fuel_type}")

    def _h2_combustion_products(self, M, M_air):
        """Calculate H2 combustion products: H2 + 0.5 O2 -> H2O"""
        # For 1 mole of H2
        mass_air = M["H2"] / self.FAR_mass
        moles_O2_air = 0.21 * mass_air / M_air
        moles_N2 = 0.79 * mass_air / M_air

        # Combustion: 1 H2 + 0.5 O2 -> 1 H2O
        moles_H2O = 1
        moles_O2_remaining = moles_O2_air - 0.5

        total_moles = moles_H2O + moles_N2 + moles_O2_remaining

        components = ["H2O", "N2", "O2"]
        mole_fractions = [
            moles_H2O / total_moles,
            moles_N2 / total_moles,
            moles_O2_remaining / total_moles,
        ]

        return components, mole_fractions

    def _hydrocarbon_combustion_products(self, M, M_air):
        """Calculate C0.92H2 combustion products: C0.92H2 + 1.42 O2 -> 0.92 CO2 + H2O"""
        # For 1 mole of C0.92H2
        mass_air = M["C092H2"] / self.FAR_mass
        moles_O2_air = 0.21 * mass_air / M_air
        moles_N2 = 0.79 * mass_air / M_air

        # Combustion: 1 C0.92H2 + 1.42 O2 -> 0.92 CO2 + 1 H2O
        moles_CO2 = 0.92
        moles_H2O = 1.0
        moles_O2_remaining = moles_O2_air - 1.42

        total_moles = moles_CO2 + moles_H2O + moles_N2 + moles_O2_remaining

        components = ["CO2", "H2O", "N2", "O2"]
        mole_fractions = [
            moles_CO2 / total_moles,
            moles_H2O / total_moles,
            moles_N2 / total_moles,
            moles_O2_remaining / total_moles,
        ]

        return components, mole_fractions

    def get_transport_properties(self, T: float, P: float) -> tuple:
        return self.mixture.get_transport_properties(T, P)

    def get_density(self, T: float, P: float) -> float:
        return self.mixture.get_density(T, P)

    def get_cp(self, T: float, P: float) -> float:
        return self.mixture.get_cp(T, P)

    def get_viscosity(self, T: float, P: float) -> float:
        return self.mixture.get_viscosity(T, P)

    def get_thermal_conductivity(self, T: float, P: float) -> float:
        return self.mixture.get_thermal_conductivity(T, P)

    def get_specific_enthalpy(self, T: float, P: float) -> float:
        return self.mixture.get_specific_enthalpy(T, P)

    def get_specific_entropy(self, T: float, P: float) -> float:
        return self.mixture.get_specific_entropy(T, P)
