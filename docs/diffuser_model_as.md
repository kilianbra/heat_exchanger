## Diffuser + “dump diffuser” + HEX model (as implemented in `Hex_model.hex_diff`)

This note explains:

- The **theory** the diffuser section is trying to represent (pressure recovery vs loss).
- The **three-step area change** used in this simple model: $A_{\mathrm{fr},\mathrm{in}}\rightarrow A_{\mathrm{fr},\mathrm{diff}}\rightarrow A_{\mathrm{fr},\mathrm{hex}}$.
- How the diffuser pieces are **actually used** in the codebase, and how they feed into the “simple hex model” and the cycle (`exhaust_model.py`).

This is written to match the intent captured in the KB comments around `Hex_model.py` lines ~70–114.

---

## Where this lives in the code

- **Top-level model**: `Hex_model.py`
  - `hex_diff(params)`: wraps “diffuser + incidence/dump + HEX core” and returns `params` augmented with losses, heat transfer, and bookkeeping.
  - `hex_model_gen(x)`: computes the **HEX core** heat transfer and hot-side pressure drop for a simplified tube bank / crossflow model.
- **Cycle integration**:
  - `exhaust_model.py`: creates a `HEX` dict from cycle state, calls `hex_diff(HEX)`, then applies `dPo_tot` to reduce `P04ho` which reduces nozzle performance.
  - `kb_exhaust_model.py`: same concept, plus alternative HEX solvers; `hex_mod == 1` uses `hex_diff`.
- **Design tools / plotting**:
  - `Hex_app.py` and `Hex_design_plots.py`: compute `a_hex` from an *area ratio target* (`AR_hex`) and sweep diffuser parameters to visualize loss split.

---

## The “three-step” diffusion concept (what the model is assuming)

The geometry and loss modeling is built around three conceptual steps:

### Step 0 — define inlet flow area $A_{\mathrm{fr},\mathrm{in}}$

The diffuser “inlet” area is not a geometric input; it is inferred from continuity at the diffuser inlet:

$$
A_{\mathrm{fr},\mathrm{in}} = \frac{\dot m}{\rho_{\mathrm{in}} V_{\mathrm{in}}}
$$

In `hex_diff()`:

- `V_in` is an input (`params["V_in"]`).
- $\rho_{\mathrm{in}}$ is computed from (`T01`, `P01`) and `V_in` using a simple compressible/incompressible mix:
  - it converts stagnation to static using a Mach estimate, then uses $\rho = P/(RT)$.
- The flow is treated as an **annulus with mean radius** `r_in = params["r"]`, so inlet annulus “height” is:

$$
h_{\mathrm{in}} = \frac{A_{\mathrm{fr},\mathrm{in}}}{2\pi r}
$$

### Step 1 — attached diffuser: $A_{\mathrm{fr},\mathrm{in}}\rightarrow A_{\mathrm{fr},\mathrm{diff}}$

The code treats the diffuser as a **2-wall symmetric opening** in annulus height over an axial length `L_diff` with a (half-)angle `diff_ang`:

$$
h_{\mathrm{diff}} = h_{\mathrm{in}} + 2 L_{\mathrm{diff}}\tan(\theta)
$$

$$
A_{\mathrm{fr},\mathrm{diff}} = 2\pi r h_{\mathrm{diff}}
$$

So the diffuser area ratio is:

$$
\mathrm{AR}*{\mathrm{diff}} = \frac{A*{\mathrm{fr},\mathrm{diff}}}{A_{\mathrm{fr},\mathrm{in}}}
$$

This is the “do as much diffusion as you can” part. In the KB comments, the implied practical constraint is:

- **limit $\mathrm{AR}_{\mathrm{diff}}\lesssim 1.4$** (typical “combustor-style”/short diffuser constraints due to separation / boundary layer growth).

In this codebase, that limit is not enforced explicitly; it’s managed by selecting reasonable `L_diff` and `diff_ang` ranges in the plotting tools and by the angle effectiveness penalty discussed next.

### Step 2 — dump diffuser / incidence into the HEX: $A_{\mathrm{fr},\mathrm{diff}}\rightarrow A_{\mathrm{fr},\mathrm{hex}}$

If the required HEX frontal area `a_hex` is much larger than the “attached diffuser” can reach (while staying efficient/attached), the model assumes the remaining area increase happens as a **dump / sudden expansion** into the HEX face:

$$
A_{\mathrm{fr},\mathrm{hex}} \gg A_{\mathrm{fr},\mathrm{diff}}
$$

The intent (per KB comments) is:

- Diffuse “as much as practical” to `a_diff`
- Then take the remaining expansion as a **dump loss** (called “incidence” in the code)

This loss is intentionally the *dominant* diffuser-related term in this simplified model.

---

## Diffuser theory used (pressure recovery vs loss)

There are two distinct physical ideas in diffuser modeling:

1. **Pressure recovery** (ideal, inviscid): converting dynamic pressure into static pressure as velocity decreases.
2. **Total pressure loss** (real, viscous/separated): loss due to wall shear, separation, mixing, and non-uniformity.

### Ideal static pressure recovery coefficient $C_{p,\mathrm{i}}$

For an incompressible, uniform, 1D diffuser with no losses:

$$
C_{p,\mathrm{i}} \equiv \frac{p_2-p_1}{\tfrac12\rho V_1^2}
     = 1-\left(\frac{V_2}{V_1}\right)^2
     = 1-\left(\frac{A_1}{A_2}\right)^2
     = 1-\frac{1}{\mathrm{AR}^2}
$$

The code uses exactly this expression:

- `CPi = 1 - 1/(AR_diff**2)`

### “Effectiveness” vs angle (empirical penalty)

Real diffusers do not achieve the ideal recovery, especially at larger diffuser angles where separation becomes likely. The code applies an empirical multiplier based on diffuser angle:

- `ang = [2.0, 4.0, 7.5, 10.6]` degrees
- `cpi = [0.93, 0.826, 0.667, 0.444]` (dimensionless)
- `cp = fcp(diff_ang) * CPi`

Despite the variable name `cpi`, these values behave like a **diffuser effectiveness** $\eta_{\mathrm{diff}}$ such that:

$$
C_p \approx \eta_{\mathrm{diff}}(\theta)C_{p,\mathrm{i}}
$$

Interpretation:

- Small `diff_ang` ⇒ closer to ideal recovery
- Larger `diff_ang` ⇒ worse recovery (separation / non-uniformity), hence lower effective $C_p$

### “Effective area” implied by achieved $C_p$

Given $C_p = 1-(A_1/A_{2,\mathrm{eff}})^2$, you can rearrange:

$$
A_{2,\mathrm{eff}} = \frac{A_1}{\sqrt{1-C_p}}
$$

The code computes this as:

- $A_{\mathrm{fr},\mathrm{diff},\mathrm{eff}} = A_{\mathrm{fr},\mathrm{in}}\sqrt{1/(1-C_p)}$, i.e. `a_diff_eff = a_in * sqrt(1/(1 - cp))`

This is a **diagnostic** for “how much diffusion you effectively got” after accounting for angle effectiveness (e.g., boundary-layer/separation reducing the usable area gain). In the current implementation it is not used downstream in the incidence loss term (see next section), but it is stored for plotting:

- `params["AR_diff_eff"]` stores $A_{\mathrm{fr},\mathrm{diff},\mathrm{eff}}/A_{\mathrm{fr},\mathrm{in}}$

---

## Losses used in the implementation (what actually contributes to `dPo_tot`)

`hex_diff()` produces three hot-side total pressure loss contributions:

$$
\Delta P_{0,\mathrm{tot}} = \Delta P_{0,\mathrm{diff}} + \Delta P_{0,\mathrm{inc}} + \Delta P_{0,\mathrm{hex}}
$$

and stores them as:

- `params["dPo_diff"]` (attached diffuser wall loss)
- `params["dPo_inc"]` (dump/incidence loss into the HEX face)
- `params["dPo_hex"]` (HEX core hot-side loss from `hex_model_gen`)

### 1) Attached diffuser loss: `dPo_diff`

The code includes a small wall-loss style term:

- `Cd = 0.003`
- `loss_diff = (0.5*Cd*ρ*V_in^3) * (a_in/tanθ) * (1 - (a_in/a_diff)^2)`
- `dPo_diff = ρ * loss_diff / m_dot`

Notes / interpretation:

- This looks like an integrated shear-power / dissipation estimate mapped back into a pressure loss via $ \Delta p = \dot{E}/\dot{V} $ style reasoning.
- It uses **inlet** velocity `V_in` and **inlet** density `ro_in` throughout.
- Per the KB comments, this term is expected to be **small** relative to the dump/incidence loss for this short diffuser + big area jump problem.

### 2) Dump / incidence loss: `dPo_inc` (dominant term)

This is the “important term” in the KB comments. As implemented:

$$
\Delta P_{0,\mathrm{inc}}
= f_{\mathrm{loss}}\left(1-\left(\frac{A_{\mathrm{fr},\mathrm{diff}}}{A_{\mathrm{fr},\mathrm{hex}}}\right)^2\right)
\frac12\rho_{\mathrm{in}}V_{\mathrm{diff}}^2
$$

where $V_{\mathrm{diff}}$ is the velocity at diffuser exit based on `a_diff`:

$$
V_{\mathrm{diff}} = \frac{\dot m}{\rho_{\mathrm{in}}A_{\mathrm{fr},\mathrm{diff}}}
$$

In code:

- `dPo_inc = diff_loss_frac * (1 - (a_diff/a_hex)**2) * 0.5 * ((m_dot/a_diff)**2) / ro_in`

Notes:

- `diff_loss_frac` is a **tuning multiplier** (the comment says it “should be 1”).
- The form $\left(1-(A_{\mathrm{fr},\mathrm{diff}}/A_{\mathrm{fr},\mathrm{hex}})^2\right)$ is a simplified “incidence/dump severity” factor driven by how mismatched the areas are.
- This is the place where the **three-step** idea matters:
  - Increasing `a_diff` (via `L_diff` or `diff_ang`) reduces $V_{\mathrm{diff}}$ and reduces the mismatch term, cutting `dPo_inc`.
  - Increasing `a_hex` at fixed `a_diff` increases mismatch and increases `dPo_inc`.

### 3) HEX core hot-side loss: `dPo_hex`

`hex_model_gen()` computes heat transfer and pressure drop through a simplified tube bank model. The hot-side pressure loss returned as `out[3]` is used:

- `dPo_hex = out[3]`

This loss scales strongly with hot-side mass velocity through the minimum/free-flow area in the tube bank model (see `hex_model_gen()` around `Aff_h`, `Re_h`, and `dPo_h`).

---

## How `a_hex` is chosen / interpreted in the wider code

`a_hex` is the **frontal (face) flow area** that the diffuser must feed.

Two common patterns in this repo:

### In `Hex_app.py` / `Hex_design_plots.py` (design sweeps)

They treat `AR_hex` as a design variable and set:

$$
A_{\mathrm{fr},\mathrm{hex}} = \mathrm{AR}*{\mathrm{hex}}A*{\mathrm{fr},\mathrm{in}}
$$

So `a_hex` is directly coupled to inlet flow conditions through $A_{\mathrm{fr},\mathrm{in}}$.

### In `exhaust_model.py` / `kb_exhaust_model.py` (cycle integration)

`a_hex` is passed in as `dp["a_hex"]` (a cycle/design input), then forwarded into `HEX["a_hex"]`.

So, from the cycle model’s perspective:

- `a_hex` is “what you have available to dump into” (installation / packaging / design choice)
- The diffuser parameters (`L_diff`, `diff_ang`) decide how much of that you can reach with an attached diffuser (`a_diff`), and the remaining mismatch drives `dPo_inc`.

---

## How diffuser losses affect thrust/TSFC in the cycle

In `exhaust_model.cycle()` when `hex_mod == 1`:

- `HEX = hex_diff(HEX)` returns `HEX["dPo_tot"]` (Pa)
- Cycle applies:
  - `dp["P04ho"] = dp["P04hi"] - HEX["dPo_tot"]`
  - `dp["T04ho"] = dp["T04hi"] - HEX["dT"]`

Then downstream nozzle work/velocity (`Vh`) and thrust are computed from this reduced stagnation pressure.

So the diffuser model matters because:

- **Heat transfer** reduces fuel required (good for TSFC)
- **Pressure loss** reduces available nozzle pressure ratio / jet velocity (bad for thrust, worsens TSFC)

The plotting scripts often show the loss split:

- diffuser wall loss (`dPo_diff`)
- dump/incidence loss (`dPo_inc`)
- HEX core loss (`dPo_hex`)

and the KB comments emphasize that, for this architecture, **dump/incidence is usually the controlling diffuser-related penalty**.

---

## Practical reading of the KB comments (how to use this model)

If you read the diffuser section as a design heuristic:

- **Choose `L_diff` and `diff_ang`** to get a reasonable `AR_diff = a_diff/a_in` without making `diff_ang` so large that the recovery effectiveness collapses (and, physically, separation would occur).
- Treat **$\mathrm{AR}_{\mathrm{diff}}\sim 1.2$–$1.4$** as “maximum practical attached diffusion” in this simplified short-annular-diffuser context.
- Any additional required face area `a_hex` beyond what you can reach with `a_diff` is paid for as **dump/incidence loss** via `dPo_inc`.

Important implementation detail:

- The **recovery coefficient `cp` is mostly diagnostic right now** (used to compute `AR_diff_eff` and stored for plots), while `dPo_inc` is the term that actually enforces the “dump diffuser penalty”.

