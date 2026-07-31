"""Pull REC_*/GEOM_* for bar charts from fig8/9 optima into cycle_assumptions.py."""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_OLD = Path(__file__).resolve().parent / "Old_figs" / "old_scripts"
if _OLD.is_dir() and str(_OLD) not in sys.path:
    sys.path.insert(0, str(_OLD))

import fig8_no_cycle_model as f8  # noqa: E402
import fig9_w_cycle_model as f9  # noqa: E402

CYCLE_ASSUMPTIONS_PATH = Path(__file__).resolve().parent / "cycle_assumptions.py"


@dataclass(frozen=True)
class RecupSyncCase:
    stem: str
    eps: float
    dp_hot_frac: float
    dp_cold_frac: float
    a_over_a_ref: float
    ao_over_ao_ref: float


def extract_recup_cases_from_fig89() -> tuple[RecupSyncCase, RecupSyncCase, RecupSyncCase]:
    """
    Reference + cycle-model (vary BC & mdot) fixed-mass and global optima.

    eps / dp match fig8_red_only CYCLE MODEL columns; geometry from fig9 red line.
    """
    data8 = f8.get_line_data()
    data9 = f9.get_line_data()
    if data8 is None or data9 is None:
        raise RuntimeError("fig8/9 get_line_data() returned no valid data")

    ntu_match = float(f8.NTU_MATCH)
    a_r_ref = float(f8._a_over_a_ref(1.0, ntu_match))
    pdr9 = data9["pressure_drop_ratio"]

    ao_fix = float(np.interp(a_r_ref, data9["red"]["a"], data9["red"]["ao"]))
    ntu_fix = float(np.interp(a_r_ref, data9["red"]["a"], data9["red"]["ntu"]))
    id_cyc = int(data9["id_min_red"])
    ao_glob = float(data9["red"]["ao"][id_cyc])
    ntu_glob = float(data9["red"]["ntu"][id_cyc])
    a_glob = float(data9["red"]["a"][id_cyc])

    def _cycle(ao: float, ntu: float) -> tuple[float, float, float]:
        out = f9.solve_mdot_at_constant_power(
            ao,
            ntu,
            pdr9,
            data9["mdot_at_ref"],
            data9["T_hot_in_ref"],
            data9["P_shaft_ref"],
        )
        eps, dp_h, dp_c = out[3], out[4], out[5]
        if not (np.isfinite(eps) and np.isfinite(dp_h) and np.isfinite(dp_c)):
            raise RuntimeError(f"cycle solve failed at ao={ao}, ntu={ntu}")
        return float(eps), float(dp_h), float(dp_c)

    e_ref, dh_ref, dc_ref = _cycle(1.0, ntu_match)
    e_fix, dh_fix, dc_fix = _cycle(ao_fix, ntu_fix)
    e_glob, dh_glob, dc_glob = _cycle(ao_glob, ntu_glob)

    return (
        RecupSyncCase("rec_ref", e_ref, dh_ref, dc_ref, 1.0, 1.0),
        RecupSyncCase("rec_fix", e_fix, dh_fix, dc_fix, a_r_ref, ao_fix),
        RecupSyncCase("rec_glob", e_glob, dh_glob, dc_glob, a_glob, ao_glob),
    )


def _fmt(x: float) -> str:
    return repr(float(x))


def format_cycle_assumptions_block(cases: tuple[RecupSyncCase, RecupSyncCase, RecupSyncCase]) -> str:
    ref, fix, glob = cases
    return f"""# Recuperator cases for conf PPT bar charts (eps, dp_h/p_hi, dp_c/p_ci).
# Synced from fig8/9 cycle-model optima (sync_recup_from_fig89.py):
#   REC_REF  = Ao=1, NTU_MATCH; REC_FIX = fixed-mass red-line; REC_GLOB = global red-line.
REC_REF = RecuperatorInputs(
    eps={_fmt(ref.eps)},
    dp_hot_frac={_fmt(ref.dp_hot_frac)},
    dp_cold_frac={_fmt(ref.dp_cold_frac)},
)
REC_FIX = RecuperatorInputs(
    eps={_fmt(fix.eps)},
    dp_hot_frac={_fmt(fix.dp_hot_frac)},
    dp_cold_frac={_fmt(fix.dp_cold_frac)},
)
REC_GLOB = RecuperatorInputs(
    eps={_fmt(glob.eps)},
    dp_hot_frac={_fmt(glob.dp_hot_frac)},
    dp_cold_frac={_fmt(glob.dp_cold_frac)},
)

# Geometry from fig9 red-line coupled optima (get_line_data).
GEOM_REF = RecupHexGeometry(a_over_a_ref=1.0, ao_over_ao_ref=1.0)
GEOM_FIX = RecupHexGeometry(a_over_a_ref={_fmt(fix.a_over_a_ref)}, ao_over_ao_ref={_fmt(fix.ao_over_ao_ref)})  # fixed mass, A/A_ref = 1
GEOM_GLOB = RecupHexGeometry(a_over_a_ref={_fmt(glob.a_over_a_ref)}, ao_over_ao_ref={_fmt(glob.ao_over_ao_ref)})  # global aircraft-mass opt
"""


def write_cycle_assumptions(cases: tuple[RecupSyncCase, RecupSyncCase, RecupSyncCase] | None = None) -> Path:
    """Replace REC_*/GEOM_* block in cycle_assumptions.py."""
    if cases is None:
        cases = extract_recup_cases_from_fig89()
    text = CYCLE_ASSUMPTIONS_PATH.read_text(encoding="utf-8")
    block = format_cycle_assumptions_block(cases)
    pattern = re.compile(
        r"# Recuperator cases for conf PPT bar charts.*?"
        r"GEOM_GLOB = RecupHexGeometry\([^\n]+\)[^\n]*\n",
        re.DOTALL,
    )
    if not pattern.search(text):
        raise RuntimeError(f"could not find REC_*/GEOM_* block in {CYCLE_ASSUMPTIONS_PATH}")
    new_text = pattern.sub(block, text, count=1)
    CYCLE_ASSUMPTIONS_PATH.write_text(new_text, encoding="utf-8")
    return CYCLE_ASSUMPTIONS_PATH


def print_cases(cases: tuple[RecupSyncCase, RecupSyncCase, RecupSyncCase]) -> None:
    print("\n" + "=" * 72)
    print("  fig8/9 -> cycle_assumptions (bar chart REC_*/GEOM_*)")
    print("=" * 72)
    for c, label in zip(cases, ("REC_REF", "REC_FIX", "REC_GLOB"), strict=True):
        print(
            f"  {label}: eps={c.eps:.10g}, dp_h={c.dp_hot_frac:.10g}, dp_c={c.dp_cold_frac:.10g}, "
            f"A/A_ref={c.a_over_a_ref:.6g}, Ao/Ao_ref={c.ao_over_ao_ref:.6g}"
        )
    print("=" * 72 + "\n")


def main(*, write: bool = True) -> None:
    cases = extract_recup_cases_from_fig89()
    print_cases(cases)
    if write:
        path = write_cycle_assumptions(cases)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main(write=True)
