from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from functools import cached_property
from typing import ClassVar, Protocol

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
                    r"C:\Program Files\REFPROP",
                    r"C:\Program Files (x86)\REFPROP",
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


class FluidState(Protocol):
    """
    Thermodynamic state for a fluid at a specified temperature (in Kelvin) and pressure (in Pascals).

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


class FluidModel(Protocol):
    """
    Fluid model capable of producing a state at (T, P) (temperature in Kelvin and pressure in Pascals).

    For now, states are defined strictly by temperature and pressure; additional
    state definitions (e.g., (p, h)) can be added later without changing callers
    that use this interface.
    """

    def state(self, T: float, P: float) -> FluidState: ...


class FluidInputs(Protocol):
    """
    Inputs container for a two-stream heat exchanger.

    Contains the fluid models for the hot and cold streams, their mass flow rates,
    and inlet boundary values. The hot-side pressure can be provided either at
    inlet (Ph_in) or as outlet (Ph_out); implementations should ensure exactly
    one of these is provided.

    Units:
      - Temperatures [K]
      - Pressures [Pa]
      - Mass flows [kg/s]
    """

    # Fluids
    hot: FluidModel
    cold: FluidModel

    # Mass flow rates
    m_dot_hot: float
    m_dot_cold: float

    # Boundary values
    Tc_in: float
    Pc_in: float
    Th_in: float

    # Hot-side pressure: provide either inlet or outlet
    Ph_in: float | None
    Ph_out: float | None


class PerfectGasFluid:
    # region Docstring
    """
    Perfect gas model with constant specific heat and Sutherland viscosity.
    Default values are for Air.

    Parameters
    ----------
    M : float, default 28.97
        Molecular weight [kg/kmol] used to compute R_specific.
    S : float, default 111.0
        Sutherland's constant [K] (Air value).
    T_ref : float, default 273.0
        Reference temperature [K] for Sutherland law (Air value).
    mu_ref : float, default 1.716e-5
        Reference dynamic viscosity [Pa·s] at T_ref (Sutherland law, Air value).
    gamma : float, default 1.4
        Ratio of specific heats (cp/cv). Model-level parameter.
    Pr : float, default 0.7
        Prandtl number (assumed constant). Model-level parameter.

    Notes
    -----
    - Model parameters (mu_ref, T_ref, S, M, gamma, Pr, R_specific) are properties of the
      model and are not stored on the state; the state exposes only thermodynamic
      properties at (T, P) and uses cached properties to avoid redundant computation.
    - Entropy reference is s = 0 at T0 = 300 K and P0 = 101325 Pa.
    - Sutherland's law constants for common gases:

      Gas      S (K)     T_ref (K)    mu_ref (Pa.s)   Gamma   Pr
    ---------------------------------------------------------------
      Air      111       273          1.716e-5      1.4     0.7  (Default)
      N2       107       273          1.663e-5      1.4     0.7
      H2        97       273          8.411e-6      1.41    0.7
      Steam   1064       350          1.12e-5       1.3     0.99
      He        79.4     273          1.9e-5        1.67    0.7

    References:
    Sutherland constants https://doc.comsol.com/6.2/doc/com.comsol.help.cfd/cfd_ug_fluidflow_high_mach.08.41.html
    except Helium Sutherland constants from German Wikipedia: https://de.wikipedia.org/wiki/Sutherland-Modell
    Gamma from CUED Engineering Databook
    """

    # endregion

    R_UNIVERSAL = 8314.462618  # J/(kmol·K)
    # region Presets
    _PRESET_LIBRARY: ClassVar[dict[str, dict[str, float]]] = {
        "air": {
            "M": 28.97,
            "S": 111.0,
            "T_ref": 273.15,
            "mu_ref": 1.716e-5,
            "gamma": 1.4,
            "Pr": 0.70,
        },
        "hydrogen": {
            "M": 2.016,
            "S": 97.0,
            "T_ref": 273.15,
            "mu_ref": 8.411e-6,
            "gamma": 1.41,
            "Pr": 0.70,
        },
        "nitrogen": {
            "M": 28.013,
            "S": 107.0,
            "T_ref": 273.15,
            "mu_ref": 1.663e-5,
            "gamma": 1.40,
            "Pr": 0.70,
        },
        "steam": {
            "M": 18.015,
            "S": 1064.0,
            "T_ref": 350.0,
            "mu_ref": 1.12e-5,
            "gamma": 1.30,
            "Pr": 0.99,
        },
        "helium": {
            "M": 4.0026,
            "S": 79.4,
            "T_ref": 273.15,
            "mu_ref": 1.9e-5,
            "gamma": 1.66,
            "Pr": 0.70,
        },
    }
    _PRESET_ALIASES: ClassVar[dict[str, str]] = {}
    _PRESET_DISPLAY_NAMES: ClassVar[dict[str, str]] = {
        "air": "Air",
        "hydrogen": "Hydrogen",
        "nitrogen": "Nitrogen",
        "steam": "Steam",
        "helium": "Helium",
    }

    _PRESET_SYNONYMS: ClassVar[dict[str, tuple[str, ...]]] = {
        "air": ("air", "Air", "AIR"),
        "hydrogen": ("hydrogen", "Hydrogen", "HYDROGEN", "h2", "H2"),
        "nitrogen": ("nitrogen", "Nitrogen", "NITROGEN", "n2", "N2"),
        "steam": ("steam", "Steam", "STEAM", "water", "Water", "WATER", "h2o", "H2O"),
        "helium": ("helium", "Helium", "HELIUM", "he", "He", "HE"),
    }
    for _canonical, _aliases in _PRESET_SYNONYMS.items():
        for _alias in _aliases:
            _PRESET_ALIASES[_alias.lower()] = _canonical

    # endregion
    def __init__(
        self,
        M: float = 28.97,
        S: float = 111.0,
        T_ref: float = 273.0,
        mu_ref: float = 1.716e-5,
        *,
        gamma: float = 1.4,
        Pr: float = 0.7,
        label: str | None = None,
    ) -> None:
        self.R_specific = self.R_UNIVERSAL / M  # J/(kg·K)
        self.mu_ref = mu_ref
        self.T_ref = T_ref
        self.S = S
        self.gamma = gamma
        self.Pr = Pr
        self.cp_const = self.R_specific * gamma / (gamma - 1.0)
        self._preset_name = label

    def state(self, T: float, P: float) -> FluidState:
        return _PerfectGasState(T=T, P=P, _model=self)

    # region Preset properties / methods
    @property
    def preset_name(self) -> str | None:
        """Return the preset identifier if constructed via :meth:`from_name`."""

        return self._preset_name

    @classmethod
    def available_presets(cls) -> tuple[str, ...]:
        """Tuple of canonical preset names (title case)."""

        return tuple(cls._PRESET_DISPLAY_NAMES[name] for name in sorted(cls._PRESET_LIBRARY))

    @classmethod
    def from_name(cls, fluid: str) -> PerfectGasFluid:
        """Instantiate using one of the built-in presets (Air, Hydrogen, Nitrogen, Steam, Helium).

        The lookup accepts CoolProp/REFPROP style names (case-insensitive) such as ``"Air"``,
        ``"NITROGEN"``, ``"H2"``, ``"Water"`` and ``"He"``.
        """

        key = fluid.strip().lower()
        canonical = cls._PRESET_ALIASES.get(key)
        if canonical is None:
            available = ", ".join(cls.available_presets())
            logger.error(
                "Unknown PerfectGasFluid preset '%s'. Available presets: %s",
                fluid,
                available,
            )
            raise ValueError(f"Unknown perfect-gas preset '{fluid}'. Available: {available}")

        params = dict(cls._PRESET_LIBRARY[canonical])
        label = cls._PRESET_DISPLAY_NAMES.get(canonical, canonical.title())
        return cls(label=label, **params)

    # endregion


@dataclass(frozen=True)
class _PerfectGasState:
    """
    Perfect gas thermodynamic state at (T, P) (temperature in Kelvin and pressure in Pascals).

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
        return self._model.cp_const

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
    def s(self, Td: float = 300.0, Pd: float = 101325.0) -> float:
        """
        Specific entropy s [J/(kg·K)] with reference s=0. Default reference state is Td=300 K, Pd=101325 Pa:
            s = cp · ln(T/Td) - R_specific · ln(P/Pd)
        """
        m = self._model
        return self.cp * np.log(self.T / Td) - m.R_specific * np.log(self.P / Pd)

    @cached_property
    def a(self) -> float:
        """Speed of sound a [m/s] for ideal gas: a = sqrt(gamma · R_specific · T)."""
        m = self._model
        return np.sqrt(m.gamma * m.R_specific * self.T)


class CoolPropFluid:
    """
    Real fluid model using CoolProp's high-accuracy equations of state (HEOS backend).

    Parameters
    ----------
    fluid_name : str
        CoolProp fluid identifier (e.g., "Air", "Water", "Nitrogen", "CO2").
        See http://www.coolprop.org/fluid_properties/PurePseudoPure.html for available fluids.
    backend : str, default "HEOS"
        CoolProp backend to use. Options include:
        - "HEOS": Helmholtz Energy Equation of State (high accuracy) --> default
        - "INCOMP": Incompressible fluids

    Notes
    -----
    - Uses CoolProp's low-level AbstractState interface for efficient property calculations.
    - All thermodynamic properties are calculated on-demand using cached properties.
    - Entropy reference is defined by CoolProp's internal reference state for each fluid.

    Examples
    --------
    >>> air = CoolPropFluid("Air")
    >>> state = air.state(T=300.0, P=101325.0)
    >>> state.rho
    1.1769...
    """

    def __init__(self, fluid_name: str, backend: str = "HEOS") -> None:
        self.fluid_name = fluid_name
        self.backend = backend
        try:
            # Test that we can create an AbstractState with this fluid
            test_state = CP.AbstractState(backend, fluid_name)
            del test_state
        except Exception as e:
            logger.error(
                "Failed to initialize CoolProp fluid '%s' with backend '%s': %s",
                fluid_name,
                backend,
                e,
            )
            raise ValueError(
                f"Cannot create CoolProp fluid '{fluid_name}' with backend '{backend}': {e}"
            ) from e

    def state(self, T: float, P: float) -> FluidState:
        return _CoolPropState(T=T, P=P, _model=self)


@dataclass(frozen=True)
class _CoolPropState:
    """
    CoolProp thermodynamic state at (T, P) (temperature in Kelvin and pressure in Pascals).

    Properties are provided as cached properties and use CoolProp's low-level
    AbstractState interface for efficient calculations.
    """

    T: float  # K
    P: float  # Pa
    _model: CoolPropFluid

    def _get_abstract_state(self) -> CP.AbstractState:
        """Create and update an AbstractState for this T, P."""
        AS = CP.AbstractState(self._model.backend, self._model.fluid_name)
        AS.update(CP.PT_INPUTS, self.P, self.T)
        return AS

    @cached_property
    def _AS(self) -> CP.AbstractState:
        """Cached AbstractState for this thermodynamic point."""
        return self._get_abstract_state()

    @cached_property
    def rho(self) -> float:
        """Density [kg/m^3] from CoolProp."""
        return self._AS.rhomass()

    @cached_property
    def cp(self) -> float:
        """Specific heat at constant pressure cp [J/(kg·K)] from CoolProp."""
        return self._AS.cpmass()

    @cached_property
    def mu(self) -> float:
        """Dynamic viscosity mu [Pa·s] from CoolProp."""
        return self._AS.viscosity()

    @cached_property
    def k(self) -> float:
        """Thermal conductivity k [W/(m·K)] from CoolProp."""
        return self._AS.conductivity()

    @cached_property
    def h(self) -> float:
        """Specific enthalpy h [J/kg] from CoolProp."""
        return self._AS.hmass()

    @cached_property
    def s(self) -> float:
        """Specific entropy s [J/(kg·K)] from CoolProp (reference defined by CoolProp)."""
        return self._AS.smass()

    @cached_property
    def a(self) -> float:
        """Speed of sound a [m/s] from CoolProp."""
        return self._AS.speed_sound()


class RefPropFluid:
    """
    Real fluid model using REFPROP's high-accuracy equations of state via CoolProp's REFPROP backend.

    Parameters
    ----------
    fluid_name : str
        REFPROP fluid identifier (e.g., "WATER", "NITROGEN", "HYDROGEN", "CO2").
        Can also use short forms like "H2O", "N2", "H2" which will be mapped to REFPROP names.
        See REFPROP documentation for available fluids.

    Notes
    -----
    - Requires REFPROP to be installed on the system.
    - Uses CoolProp's low-level AbstractState interface with REFPROP backend.
    - All thermodynamic properties are calculated on-demand using cached properties.
    - Entropy reference is defined by REFPROP's internal reference state for each fluid.
    - REFPROP path is automatically configured from environment variables or default locations.

    Examples
    --------
    >>> water = RefPropFluid("WATER")
    >>> state = water.state(T=400.0, P=101325.0)
    >>> state.rho
    0.5977...
    """

    # Mapping from common names to REFPROP standard names
    _FLUID_NAME_MAP: ClassVar[dict[str, str]] = {
        "H2O": "WATER",
        "N2": "NITROGEN",
        "O2": "OXYGEN",
        "H2": "HYDROGEN",
        "CO2": "CO2",
        "He": "HELIUM",
        "Ar": "ARGON",
        "Air": "AIR",
    }

    def __init__(self, fluid_name: str) -> None:
        configure_refprop()
        self.fluid_name = fluid_name

        # Try to map common names to REFPROP names
        refprop_name = self._FLUID_NAME_MAP.get(fluid_name, fluid_name)

        try:
            # Try to create REFPROP AbstractState
            test_state = CP.AbstractState("REFPROP", refprop_name)
            del test_state
            self._refprop_name = refprop_name
        except Exception as e:
            # If mapping failed and we used a mapped name, try the original
            if refprop_name != fluid_name:
                try:
                    test_state = CP.AbstractState("REFPROP", fluid_name)
                    del test_state
                    self._refprop_name = fluid_name
                except Exception:
                    # Both failed, raise informative error
                    error_msg = [
                        f"Error initializing REFPROP for fluid '{fluid_name}':",
                        f"Original error: {str(e)}",
                        f"Current REFPROP path: {CP.get_config_string(CP.ALTERNATIVE_REFPROP_PATH)}",
                        "Check that REFPROP is installed and the fluid name is correct.",
                    ]
                    logger.error(" ".join(error_msg))
                    raise ValueError(f"Cannot create REFPROP fluid '{fluid_name}': {e}") from e
            else:
                error_msg = [
                    f"Error initializing REFPROP for fluid '{fluid_name}':",
                    f"Original error: {str(e)}",
                    f"Current REFPROP path: {CP.get_config_string(CP.ALTERNATIVE_REFPROP_PATH)}",
                    "Check that REFPROP is installed and the fluid name is correct.",
                ]
                logger.error(" ".join(error_msg))
                raise ValueError(f"Cannot create REFPROP fluid '{fluid_name}': {e}") from e

    def state(self, T: float, P: float) -> FluidState:
        return _RefPropState(T=T, P=P, _model=self)


@dataclass(frozen=True)
class _RefPropState:
    """
    REFPROP thermodynamic state at (T, P) (temperature in Kelvin and pressure in Pascals).

    Properties are provided as cached properties and use CoolProp's REFPROP backend
    low-level AbstractState interface for efficient calculations.
    """

    T: float  # K
    P: float  # Pa
    _model: RefPropFluid

    def _get_abstract_state(self) -> CP.AbstractState:
        """Create and update an AbstractState for this T, P."""
        AS = CP.AbstractState("REFPROP", self._model._refprop_name)
        AS.update(CP.PT_INPUTS, self.P, self.T)
        return AS

    @cached_property
    def _AS(self) -> CP.AbstractState:
        """Cached AbstractState for this thermodynamic point."""
        return self._get_abstract_state()

    @cached_property
    def rho(self) -> float:
        """Density [kg/m^3] from REFPROP."""
        return self._AS.rhomass()

    @cached_property
    def cp(self) -> float:
        """Specific heat at constant pressure cp [J/(kg·K)] from REFPROP."""
        return self._AS.cpmass()

    @cached_property
    def mu(self) -> float:
        """Dynamic viscosity mu [Pa·s] from REFPROP."""
        return self._AS.viscosity()

    @cached_property
    def k(self) -> float:
        """Thermal conductivity k [W/(m·K)] from REFPROP."""
        return self._AS.conductivity()

    @cached_property
    def h(self) -> float:
        """Specific enthalpy h [J/kg] from REFPROP."""
        return self._AS.hmass()

    @cached_property
    def s(self) -> float:
        """Specific entropy s [J/(kg·K)] from REFPROP (reference defined by REFPROP)."""
        return self._AS.smass()

    @cached_property
    def a(self) -> float:
        """Speed of sound a [m/s] from REFPROP."""
        return self._AS.speed_sound()


__all__ = [
    "FluidState",
    "FluidModel",
    "FluidInputs",
    "PerfectGasFluid",
    "CoolPropFluid",
    "RefPropFluid",
]
