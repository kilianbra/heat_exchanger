# Figure / BC audit (2026-08-02) — issues only; scripts not edited

## Folder cleanup done (`Figs_current/final_journal_paper/`)

Deleted unwanted variants (also from `for_latex/` where present):

- `fig6a/b/c_NTU_w_plus.*`
- `fig7b/c_Ao_Aoref.*`
- `fig7c_homog_yaxis.*`
- `fig8_combined.*`

**Kept (intended journal set):**

- `fig4a_bar_c`, `fig4b_bar_p`
- `fig6a/b/c_oneMach_th_n_visc`
- `fig7a_Ao_Aoref`, `fig7b_homog_yaxis`
- `fig8_red_only`

(`for_latex/` now only has `fig7a_Ao_Aoref.pdf` and `fig8_cycle_red_only.png` — no oneMach/fig4 copies there yet.)

## Script defaults still point at journal (will re-pollute if re-run)

Do **not** re-run these as-is if you want the cleaned journal folder to stay clean:

| Script | Default save | Produces |
|---|---|---|
| `fig6a_lengthening.py` | `JOURNAL_PLOTS` | `fig6a_NTU_w_plus` (has hot+cold Δp) |
| `fig6b_lengthening.py` | `JOURNAL_PLOTS` | `fig6b_NTU_w_plus` (triple Mach) |
| `fig6c_lengthening.py` | `JOURNAL_PLOTS` | `fig6c_NTU_w_plus` (triple Mach) |
| `fig7b_aspect_ratio.py` | `JOURNAL_PLOTS` | `fig7b_Ao_Aoref` |
| `fig7c_aspect_ratio.py` | `JOURNAL_PLOTS` | `fig7c_Ao_Aoref` |
| `fig7bc_homog_yaxis.py` | `JOURNAL_PLOTS` | `fig7b_homog_yaxis` **and** `fig7c_homog_yaxis` |

Suggested later fixes (not done): point lengthening / non-homog Ao scripts at `EXPLORE_IDEAS`; make homog script save only `fig7b_homog_yaxis` to journal (or drop 7c call).

`fig8_combined.py` already redirects to `fig8_red_only` — only stale output files were the problem.

## Content issue: `fig6a_oneMach_th_n_visc` still missing cold Δp

`fig6_1line_th_n_visc.py` → `save_fig6a()`:

- forces `pressure_drop_percent_ratio_cold_over_hot=0.0`
- plots **only** hot Δp
- right-axis label is `Hot Pressure Drop (...)`

The cold line / dual legend you added lives in `fig6a_lengthening.py` (`fig6a_NTU_w_plus`), not in the oneMach script. Needed edit before regenerating journal 6a: port hot+cold Δp styling from lengthening (dotted/dashed, legend “Hot side / Cold side”, ylabel → `Pressure Drop (...)`) into `save_fig6a()`, and use the inlet-density `pressure_drop_ratio` (same helper already used by 6b/6c in that file).

fig6b/c oneMach availability curves **do** include cold Δp via `pressure_drop_ratio` (viscous band is combined hot+cold, coloured as one “Viscous” patch).

## BC consistency vs modelling section (fixed-BC figures 4, 6, 7; fig8 red = cycle-coupled)

Shared fixed-BC stack used by fig4 / fig6 oneMach / fig7 / fig8 practical limb is consistent with the paper’s intended baseline (and with the latex comment exact values):

| Quantity | Paper claim | Scripts | Match? |
|---|---|---|---|
| PR, TIT, η_poly,c / η_poly,t | 9, 1500 K, 88% / 84% | same in `cycle_assumptions` / fig9 | yes |
| Ambient | 1.0 bar, 288 K | same | yes |
| Cold inlet (fixed) | 9.0 bar, 588 K | 9.0 bar; T_c used as 588 | yes |
| Hot inlet (fixed) | 1.06 bar, 908 K | 1.064 bar; T_h = 907 (ratio 907/588); fig4 uses 908.0 | yes (rounding) |
| cp, γ | 1.07 kJ/kg/K, 1.4 | 1070 J/kg/K, 1.4 | yes |
| St/f, St_c=St_h | 0.40; equal St | 0.4; `f_c/f_h=1` | yes |
| A_r, o_r | 0.92, 0.23 | A_r=0.92, d_r=0.25 → o_r=d_r·A_r=0.23 | yes |
| M_h,in (baseline) | table 0.14 | 0.1362 | yes (table rounded) |
| M_c,in (baseline) | table 0.06 | ≈0.0564 from geometry | yes (table rounded) |
| ε, Δp_h, Δp_c at NTU_MATCH=1.479 | table 60% / 6.0% / 4.1%; body also says “4%” once | ≈59.66% / 6.00% / 4.11% (`REC_REF`) | yes; body “4%” is looser than table |

Checked numerically with the shared model at NTU=1.479, M_h=0.1362, inlet-density Δp ratio ≈0.685 → ε≈0.5966, Δp_h≈0.0600, Δp_c≈0.0411.

**Fig8 red line** uses the cycle-coupled path (varying hot inlet / ṁ at fixed 700 kW), which the paper explicitly allows; cold inlet stays on the PR=9 / 588 K compressor exit. Not the same as fixed-BC fig4/6/7, by design.

## Minor / non-blocking notes

- Paper prose once says fixed-BC Δp “6% and 4%”; table + model are 6.0% / 4.1%. Prefer table wording.
- Fig4 uses stagnation T_h=908 and Mach-corrected statics for availability; fig6/7 use T ratio 907/588 without that static correction in the availability calls — small, historically accepted difference.
- `Reports_latex_github/.../mypaper_journal/Figures/` still has old `fig6*_NTU_w_plus`, `fig7b/c_Ao_Aoref`, `fig8_combined_new_legend` copies; not touched here.
- Regenerating journal 6a is blocked on the cold-Δp port above; 6b/c/7a/7b/8 look OK to keep as-is once save-path defaults are fixed.

---

<!--
# Figure / BC audit (fig4, 6, 7, 8) — do not edit scripts yet

Audit date: 2026-08-02. Compared figure scripts under `analysis/paper_figures/`
against the modelling claims in this file. Scripts were **not** changed.

## Desired journal outputs (`Figs_current/final_journal_paper/`)

| Keep | Drop / do not regenerate into journal by default |
|------|--------------------------------------------------|
| `fig4a_bar_c`, `fig4b_bar_p` | — |
| `fig6a/b/c_oneMach_th_n_visc` | `fig6a/b/c_NTU_w_plus` |
| `fig7a_Ao_Aoref`, `fig7b_homog_yaxis` | `fig7b/c_Ao_Aoref`, `fig7c_homog_yaxis` |
| `fig8_red_only` | `fig8_combined` |

Folder currently matches the keep set. Unwanted variants are gone, but several
scripts still default `save_dir = JOURNAL_PLOTS` and will recreate them if re-run:
- `fig6a_lengthening.py` → `fig6a_NTU_w_plus` (this is the one with both hot+cold Δp lines)
- `fig6b_lengthening.py` / `fig6c_lengthening.py` → `fig6b/c_NTU_w_plus`
- `fig7b_aspect_ratio.py` / `fig7c_aspect_ratio.py` → `fig7b/c_Ao_Aoref`
- `fig7bc_homog_yaxis.py` also writes `fig7c_homog_yaxis` into journal (only `7b` wanted)
- `fig8_combined.py` already redirects to `fig8_red_only` (safe); old `fig8_combined.*` were leftovers

Also: `Reports_latex_github/ASME_2026/mypaper_journal/Figures/` still has old
`fig6*_NTU_w_plus`, `fig7b/c_Ao_Aoref`, `fig8_combined_new_legend` copies — not the
`oneMach_th_n_visc` / `7b_homog` / `red_only` set.

## Issue 1 (real gap): `fig6a_oneMach_th_n_visc` missing cold Δp

`fig6_1line_th_n_visc.py` `save_fig6a()` forces
`pressure_drop_percent_ratio_cold_over_hot=0.0` and labels the twin axis
**“Hot Pressure Drop”**. So the journal fig6a candidate is hot-only.

The cold line + shared legend (“Pressure Drop … Hot side / Cold side”) currently
lives only in `fig6a_lengthening.py` → `fig6a_NTU_w_plus`, which should not be the
journal default. When scripts are edited: port cold Δp + ylabel from lengthening
into `th_n_visc` fig6a (and stop writing `NTU_w_plus` into `JOURNAL_PLOTS`).

`fig6b/c_oneMach_th_n_visc` already include cold via `calculate_pressure_drop_ratio`
inside the availability breakdown — only the conventional 6a panel is incomplete.

## Issue 2: BC consistency vs paper claims — mostly OK

Paper (this file) claims for **fixed** HEx BCs:
- cold in: 9.0 bar, 588 K (PR=9, η_poly,c=88%, ambient 1 bar / 288 K)
- hot in: ~1.06 bar, ~908 K; mdot ~2.24 kg/s
- baseline performance: ε≈60%, (Δp/p)_h≈6%, (Δp/p)_c≈4% (table 4.1%)
- table Mach: M_h=0.14, M_c=0.06
- latex comment exacts: M_h=0.1362, M_c=0.0563, dp_c/p=0.04105, ε=0.5966,
  T_h=907.59, T_c=587.78, p_h=1.0638, mdot=2.2356
- geometry: o_r=0.23, A_r=0.92 → d_h,c/d_h,h=0.25; St/f=0.40; γ=1.4; c_p equal; C_r=1

Scripts (fig4 / fig6 th_n_visc / fig7a / fig7 homog / fig8–9) share the same core:
- `M_h=0.1362`, `T=907/588`, `p_h=1.064`, `p_c=9`, `A_r=0.92`, `d_r=0.25` (⇒ o_r=0.23),
  `St/f=0.4`, `f_c/f_h=1`, `γ=1.4`, `C_r=1`, `NTU_MATCH=1.479`
- inlet_density dp ratio ≈ **0.685** → at 6.00% hot gives **4.11%** cold, matching
  `REC_REF` and the latex exact comment (table rounds Mach to 0.14 / 0.06)

So: **yes — fig4/6/7/8 run on the same fixed-BC family the model claims**, aligned
with the exact comment values rather than the rounded table.

### Small inconsistencies (probably not plot-breaking)

1. **fig4 T_h = 908 K** vs **907 K** elsewhere (`907/588`). Tiny.
2. **fig4 practical** converts p,T to **static** via Mach before ratios;
   fig6/7/8 pass **stagnation** ratios (`9/1.064`, `288/588`). Magnitude:
   p_c/p_h ≈ 8.55 (static) vs 8.46 (stag); t0/tc ≈ 0.4901 vs 0.4898.
   Classical fig4 uses stag T ratio for t but static for t_dead — mixed.
3. Prose says cold dp “4%” while table/scripts use **4.1%**.
4. **fig8_red_only** intentionally uses **cycle-coupled** BCs (varying T_hot_in, mdot
   at fixed 700 kW) for the red lines — that matches §cycle_assumptions, not a bug;
   fixed-BC practical optima still use the same reference point as fig4/6/7.
5. Availability y-labels on fig6b/c vs fig7b are consistent in form; no BC issue.

## Recommended script edits (later)

1. `fig6_1line_th_n_visc.save_fig6a`: add cold Δp (same ratio as lengthening/fig7a);
   ylabel → shared Pressure Drop; hot dotted / cold dashed + legend.
2. Point `fig6*_lengthening.py` and `fig7b/c_aspect_ratio.py` defaults away from
   `JOURNAL_PLOTS` (or gate journal saves), so only keep-set names land there.
3. `fig7bc_homog_yaxis.py`: only save `fig7b_homog_yaxis` to journal (drop 7c).
4. Optionally unify fig4 on 907 K and document static-vs-stag choice for practical.

-->

\section{Modelling method}\label{sec:modelling_method}

In this section, the equations and assumptions behind the general heat exchanger model used to demonstrate the frameworks of the paper are presented. It is inspired by a similar model by McDonald~\cite{McDonald1972}. The goal is to consider geometric scaling decisions of a baseline heat exchanger that involve a trade-off between heat transfer and pressure drop, rather than to provide a high-fidelity model for a specific problem. The baseline heat exchanger is loosely based on a mixture of two reference heat exchangers: the retrofit bolt-on heat exchanger depicted in Fig.~\ref{fig:helicopter_helicopter}~\cite{McDonald1972}, and a clean sheet recuperated turboshaft heat exchanger design~\cite{McDonald1970}. %See Section~\ref{sec:cycle_assumptions} and Table~\ref{tab:baseline_parameters} for more information on these three designs.

\begin{figure}[ht]
\begin{subfigure}[b]{0.55\columnwidth}% subfigure is basically the same as minipage
\centering{
\includegraphics[height=3.4cm, alt={Recuperated Helicopter Turboshaft Engine}]{Figures/fig5a.png}%
}
\subcaption{Recuperated Engine \label{fig:helicopter_engine}}
\end{subfigure}%
\hspace\*{0.01\columnsep}% reduced horizontal space between subfigures
%%%%%%%%%%%%% no line break between these two subfigures
\begin{subfigure}[b]{0.45\columnwidth}
\centering{%
\includegraphics[height=3.4cm, alt={Annular tubular heat exchanger}]{Figures/fig5b.png}%
}%
\subcaption{Heat Exchanger Core \label{fig:helicopter_hex}}
\end{subfigure}
\caption{Bolt-on retrofit recuperator for helicopter engine \cite{McDonald1972}\label{fig:helicopter_helicopter}}
\end{figure}

\subsection{Heat exchanger geometry}

The case studies in the present work will be based on aligned counterflow heat exchangers (plate-fin style). For the sake of generality, the geometric parametrisation was chosen to be independent of the relative flow orientation of the two fluids, enabling the scaling to be easily adapted to (multi-pass) crossflow (as in Fig.~\ref{fig:helicopter_helicopter}) or radial spiral counterflow heat exchangers (as in Fig.~\ref{fig:fig1_radial_inv_big_ideal_machines}) which are common for microtube-based aviation heat exchangers, but not considered in this study\footnote{Crossflow and spiral configurations introduce geometric coupling between the flow length on one side and the free-flow area on the other, making it harder to isolate individual design choices. The practical availability framework would still apply with more complicated equations.}.

\subsubsection{Independent parameters}

The primary independent parameters chosen to be varied are the heat transfer area $A$ and free-flow area $A_{o}$ for both fluid sides. To reduce the number of independent variables, the following geometric ratios are considered:

\vspace{-1em}
\noindent
\begin{minipage}[t]{0.32\columnwidth}
\begin{equation}\label{eq:A*r_definition}
A*{\mathrm{r}} \triangleq \frac{A*{\mathrm{cold}}}{A*{\mathrm{hot}}},
\end{equation}
\end{minipage}%
\hfill
\begin{minipage}[t]{0.32\columnwidth}
\begin{equation}\label{eq:V*r_definition}
V*{\mathrm{r}} \triangleq \frac{V*{\mathrm{cold}}}{V*{\mathrm{hot}}},
\end{equation}
\end{minipage}%
\hfill
\begin{minipage}[t]{0.32\columnwidth}
\begin{equation}\label{eq:sigma*r_definition}
o*{\mathrm{r}} \triangleq \frac{A*{o,{\mathrm{cold}}}}{A*{o,{\mathrm{hot}}}}.
\end{equation}
\end{minipage}
\medskip

For aligned flow heat exchangers, the core flow lengths $L$ of both fluids are equal. The volume occupied by each fluid $V$ can be related to the free-flow area if the latter is uniform along the flow: $V = A_{o} \cdot L$. If both these conditions are fulfilled, then $V_{\mathrm{r}} = o_{\mathrm{r}}$. These ratios are inspired by Miltén et al. \cite{Milten2024}.

The area ratio $A_{\mathrm{r}}$ can be primarily varied by using fins to provide more heat transfer (and friction) area to one fluid. It is near unity for surfaces with no fins. The volume and free-flow area ratios are related to the void fractions and influence the ratio of mass velocities for given mass flow rates.

The relationship between area and volume is controlled by the hydraulic diameter $d_{\mathrm{h}}$ as defined by Kays and London \cite{Kays1984}. To maintain a general treatment including crossflow tube banks, the true volume based hydraulic diameter $d_{\mathrm{v}}$ as defined by Gunter and Shaw \cite{GunterShaw1945} is also introduced.

\begin{equation}\label{eq:dh*definition}
d*{\mathrm{h,}i} \triangleq \frac{4 A*{o,i} L_i}{A*{i}} \leq d*{\mathrm{v},i} \triangleq \frac{4 V*{i}}{A\_{i}}, \quad i \in \{\mathrm{cold},\, \mathrm{hot}\}.
\end{equation}

By definition, $d_{\mathrm{v,cold}}/d_{\mathrm{v,hot}}=V_{\mathrm{r}}/A_{\mathrm{r}}$ and for equal flow lengths, $d_{\mathrm{h,cold}}/d_{\mathrm{h,hot}}=o_{\mathrm{r}}/A_{\mathrm{r}}$. In this study, the ratios are fixed throughout to the values of the reference redesign~\cite{McDonald1970}: $o_{\mathrm{r}} = 0.23$ and $A_{\mathrm{r}} = 0.92$. As mentioned previously, the reference designs are multi-pass crossflow, but all designs in this work are counterflow for simplicity. Hence, $V_{\mathrm{r}} = o_{\mathrm{r}}$ in this work.

\subsubsection{Quantities that scale with the heat transfer area}

The heat exchanger core is assumed to have a metal or wall volume $V_{\mathrm{metal}}$ proportional to the heat transfer area. The constant of proportionality is defined as follows:

\begin{equation}\label{eq:t*def}
t*{\mathrm{hot}} \triangleq \frac{V*{\mathrm{metal}}}{A*{\mathrm{hot}}}.
\end{equation}

As the heat transfer area can potentially include fins, this combines the fin thicknesses on both sides and the wall thickness into one parameter.
The total mass of the heat exchanger core is then given by the metal material density $\rho_{\mathrm{metal}}$:

\begin{equation}\label{eq:m*HEx}
m*{\mathrm{HEx}} = \rho*{\mathrm{metal}} \cdot t*{\mathrm{hot}} \cdot A\_{\mathrm{hot}}.
\end{equation}

The volume of the core, $V_{\mathrm{HEx}}$, can be broken down into three volumes occupied by the metal walls and the two fluids and rearranged using the volume ratio Eq.~\eqref{eq:V_r_definition} and the volume based hydraulic diameter Eq.~\eqref{eq:dh_definition}.

\begin{equation}\label{eq:V*core_breakdown}
V*{\mathrm{HEx}} = A*{\mathrm{hot}} \left[ t*{\mathrm{hot}} + \frac{d*{\mathrm{v,hot}} }{4} \left( 1 + V*{\mathrm{r}} \right) \right].
\end{equation}

\subsubsection{Quantities that scale with free-flow area}

For a given heat transfer area and core volume, the choice of free-flow area is a key driver of the velocity of the fluids. For aligned flow heat exchangers, the frontal area of the core $A_{\mathrm{fr}}$ can be broken down into the free-flow areas of the two fluids (assumed at the same axial location) and metal area $A_{{\mathrm{fr,metal}}}$ at the free-flow cross section.

\begin{equation}\label{eq:A*fr_breakdown}
A*{\mathrm{fr}} = A*{o,{\mathrm{cold}}} + A*{o,{\mathrm{hot}}} + A\_{\mathrm{fr,metal}}.
\end{equation}

In heat exchangers with uniform cross section, as is assumed in this work \footnote{Which is not the case for general flow over tube banks or heat exchangers with radial flow. Both of these apply to the heat exchanger in Fig.~\ref{fig:helicopter*helicopter}, in which case an adequate representative free flow area must be chosen for first order modelling.}, the cross sectional area occupied by metal is related to the metal volume as follows $A*{\mathrm{fr,metal}} = V\_{\mathrm{metal}} / L$. Equations~\eqref{eq:sigma_r_definition}, \eqref{eq:dh_definition}, \eqref{eq:t_def}, and \eqref{eq:A_fr_breakdown} can be combined to give:

\begin{equation}\label{eq:A*fr_propto_Ao_hot}
A*{\mathrm{fr}} = A*{o,{\mathrm{hot}}} \left( 1 + o*{\mathrm{r}} + \frac{4 t*{\mathrm{hot}}}{d*{\mathrm{h,hot}}} \right).
\end{equation}

For the rest of this work, changing the free-flow area and frontal area will be used interchangeably.

\subsection{Heat exchanger performance correlations}\label{sec:perf_correl}

The dimensionless coefficient used for modelling pressure drop is the friction factor $f$ as defined by Kays and London~\cite{Kays1984} (Fanning-like). In general, this includes dissipation originating from wall shear and in wakes downstream of flow interruptions.

The friction factor is related to the dimensionless heat transfer coefficient by an analogy assumption. Various analogies relate heat and momentum transfer, which rely on similar physical mechanisms. The Reynolds analogy states that for Prandtl number $\Pr \approx 1$, the ratio of the Stanton number can be approximated by half of the friction factor: $\St/f \approx 0.5$. A generalisation of this over a larger range of $\Pr$ is the Chilton-Colburn analogy based on the Colburn factor $j \triangleq \St \Pr^{2/3}$ where $j/f \approx 0.5$ \cite{Shah2024}. In practice, real heat exchanger surfaces do not achieve such high heat transfer coefficients and $j/f < 0.5$ is typically observed \cite{Kays1984}. In this work, it will be assumed that:

\begin{equation}\label{eq:j_f_relation}
\frac{j}{f} = 0.32 \quad \text{for } \Pr = 0.7 \quad \Longleftrightarrow \quad \frac{\St}{f} = 0.40.
\end{equation}
The reference redesign has analogy ratios \num{0.29} (hot, over tube bank) and \num{0.43} (cold, in tubes)~\cite{McDonald1970}. For simplicity, both were modelled as the same constant value. The chosen value is similar to that of laminar flow in smooth circular tubes (where the constant heat flux boundary condition case has $\St/f = 4.36 /16/0.7=0.39$).

Furthermore, the Stanton numbers (and hence friction factors) are assumed to have the following proportionality with the Reynolds number \cite{Milten2024}:

\noindent
\begin{minipage}[c]{0.48\columnwidth}
\begin{equation}\label{eq:St*Re_relation}
\St \propto \Re^{-0.413},
\end{equation}
\end{minipage}
\hfill
\begin{minipage}[c]{0.48\columnwidth}
\begin{equation}\label{eq:Re_side}
\Re \triangleq G d*{\text{h}}/\mu,
\end{equation}
\end{minipage}%

\medskip

where $G=\dot{m}/A_o$ is the mass velocity at the minimum free-flow area. The correlations apply to both fluids, but it will be assumed that $\St_{\mathrm{cold}} = \St_{\mathrm{hot}}$.
% Note: this was done so that to approximately achieve a similar pressure drop ratio as in the retrofit design.
%Both this and the analogy ratio were found to be reasonable values given the reference redesigned heat exchanger of Table \ref{tab:baseline_parameters}, which has a value of \num{0.26}.

\subsection{Heat exchanger model}
The geometric parameters, correlations and fluid mass flow rate and properties (see next section) are combined to form the heat exchanger model of this study.

The heat transfer is modelled via the number of heat transfer units $N_{\text{tu}}$. The mean thermal conductance $(hA)_i$ of side $i$ can be related to the heat transfer to free-flow area ratio from the definition of the Stanton number and the heat capacity flow rate $C_i = \dot{m}_i c_{p,i}$ ($C_{\min}\triangleq \min_i(C_i)$) as follows:

\begin{equation}
(hA)_i = \St_i \cdot G_i c_{p,i} A*i = C_i \cdot \frac{\St_i A_i}{A*{o,i}}.
\end{equation}

Assuming negligible wall thermal resistance and ideal fin efficiencies, $N_{\text{tu}}$ is then given by:

\begin{equation}\label{eq:NTU*overall}
N*{\mathrm{tu}} = \frac{ \St*{\mathrm{hot}} A*{\mathrm{hot}}}{A*{o,\mathrm{hot}}} \cdot \frac{C*{\mathrm{hot}}}{C*{\min}} \left( 1 + \frac{C*{\mathrm{hot}}}{C*{\mathrm{cold}}} \cdot \frac{\St*{\mathrm{hot}}}{\St*{\mathrm{cold}}} \cdot \frac{o*{\mathrm{r}}}{A\_{\mathrm{r}}} \right)^{-1}.
\end{equation}

The thermal effectiveness $\varepsilon$, defined as the ratio of actual heat transfer to the maximum thermodynamically possible heat transfer between the streams given the inlet temperatures, can be related to $N_{\text{tu}}$ for various heat exchanger configurations such as single pass crossflow or counterflow.
The traditional counterflow expression $C_i$ only depends on their ratio \cite{Kays1984} and can be rewritten as follows\footnote{The expressions is symmetric with respect to the heat capacity flow rates, which can be seen by multiplying numerator and denominator by $-e^{-\Gamma}$. The relevant case for this work, where $C_{\mathrm{hot}} = C_{\mathrm{cold}}$, can be obtained by using limits or found in heat exchanger textbooks. The general expression is presented for completeness.}, explicitly referring to the hot and cold side heat capacity flow rates $C_{\mathrm{hot}}$ and $C_{\mathrm{cold}}$:

\begin{equation}
\varepsilon = \frac{C*{\mathrm{hot}}}{C*{\mathrm{min}}} \cdot \frac{e^{\Gamma} - 1}{e^{\Gamma} - C*{\mathrm{hot}}/C*{\mathrm{cold}}},
\label{eq:counterflow*effectiveness}
\end{equation}
where
\begin{equation} %\stackrel{\text{def}}{=} or triangleq
\Gamma \triangleq N*{\text{tu}} \cdot \frac{C*{\mathrm{min}}}{C*{\mathrm{hot}}} \left(1- \frac{C*{\mathrm{hot}}}{C*{\mathrm{cold}}}\right).
\end{equation}

The hot and cold stagnation pressure losses, $\Delta p_{\mathrm{t}}$ are modelled as dominated by viscous dissipation in the core:

\begin{equation}\label{eq:dP*side}
\frac{\Delta p*{\mathrm{t}}}{p*{\mathrm{t,in}}} \approx\frac{1}{2} \overbrace{\gamma M*{\mathrm{in}}^2}^{\frac{(\dot{m}/A*o)^2}{p*{\mathrm{in}} \rho*{\mathrm{in}}}} \cdot \frac{fA}{A_o} \cdot \overbrace{\left(\frac{\rho*{\mathrm{in}}}{\rho}\right)\_{\mathrm{mean}}}^{\mathrm{neglected} \approx 1}.
\end{equation}

The variations of density $\rho$ and Mach number $M$ through the heat exchanger are neglected when modelling the pressure loss. Expressing the pressure loss as a function of the Mach number bridges two formulations: the compressible flow with friction and heat addition of Shapiro~\cite{Shapiro1953} relevant for gas turbines, and the compact heat exchanger static pressure drop of Kays and London~\cite{Kays1984} where the mass velocity $\dot{m}/A_o$ is used\footnote{Shapiro's influence coefficients show that the friction contribution to the stagnation pressure is exactly $-\tfrac{1}{2}\gamma M^2 (4f\,\mathrm{d}x/D)$, whereas the static contribution carries an additional factor $[1+(\gamma-1)M^2]/(1-M^2)$, arising from the acceleration of the flow. Equation~\eqref{eq:dP_side} is therefore taken as a stagnation pressure loss, for which it is exact to leading order $\mathcal{O}(M^4)$ at subsonic Mach numbers. The neglected heat-driven (Rayleigh) contributions correspond to the density change terms of heat exchanger textbooks.}.

The baseline heat exchanger used in this work has an effectiveness and pressure drop shown in Table~\ref{tab:baseline_parameters}, based on the bolt-on retrofit design of McDonald \cite{McDonald1972}.

\subsection{Cycle assumptions and boundary conditions}\label{sec:cycle_assumptions}

Table \ref{tab:baseline_parameters} summarises the cycle parameters of the reference cycles that informed the selection of the baseline used in this study.

\begin{table}[ht]
\centering
\caption{Baseline Cycle and Heat Exchanger parameters}
\label{tab:baseline*parameters}
\renewcommand{\arraystretch}{1.1}
\begin{tabular}{lccc}
\toprule
& \textbf{Retrofit \cite{McDonald1972}} & \textbf{Redesign \cite{McDonald1970}} & \textbf{Baseline} \\
\midrule
$\dot{P}$ (\unit{\kilo\watt}) & $209$ & $718$ & $700$ \\
$p*{\mathrm{ci}}/p*{\mathrm{ho}}$ & 6.2 & 9.0 & 9.0 \\
TIT (\unit{\kelvin}) & 1239 & 1533 & 1500 \\
$\eta$ (cycle) & $23.6\%$ & $37.9\%$ & $40.6\%$ \\
\midrule
$\varepsilon$ & 60\% & 65\% & 60\% \\
$(\Delta p*{\mathrm{t}}/p*{\mathrm{t,in}})*{\mathrm{h}}$ & 6.0\% & 3.6\% & 6.0\% \\
$(\Delta p_{\mathrm{t}}/p_{\mathrm{t,in}})_{\mathrm{c}}$ & 4.0\% & 2.2\% & 4.1\% \\
$M_{\mathrm{in,h}}$ & 0.04 & 0.06 & 0.14 \\
$M_{\mathrm{in,c}}$ & 0.03 & 0.04 & 0.06 \\
\bottomrule
\end{tabular}
\end{table}

The inlet conditions to the heat exchanger are thus based on the reference clean-sheet helicopter turboshaft cycle of McDonald \cite{McDonald1970}. The cycle of this study has been modelled with a compressor of pressure ratio 9:1 and polytropic efficiency 88\%. Continuous design power is modelled, so the stagnation inlet properties are those of the ambient sea-level conditions at $p_0=\qty{1.0}{\bar}$ and $T_0=\qty{288}{\K}$. Therefore, the heat exchanger cold stream stagnation inlet condition, which is situated at compressor exit, is \qty{9.0}{\bar} and \qty{588}{\K} for the entirety of this work.

The hot side inlet condition is modelled either as fixed or the heat exchanger model is coupled with a simple cycle model. The coupling results in hot side inlet conditions varying with the heat exchanger performance. The cycle has been modelled as having a fixed turbine inlet temperature of \qty{1500}{\K}, a turbine polytropic efficiency\footnote{The turbine polytropic efficiency was obtained from the turbine inlet and exit temperatures and pressures of \cite{McDonald1970} assuming a hot side $\gamma=1.33$.} of $84\%$ and an exhaust pressure (heat exchanger hot outlet) of \qty{1.0}{\bar}. Therefore, cold-side pressure drop reduces the combustor and turbine inlet pressures, while the hot-side pressure drop raises the required turbine exit pressure. Both effects reduce the available expansion ratio and hence turbine work. To achieve a constant net power, the cycle and heat exchanger mass flow is adjusted to achieve the required $\dot{P}=\qty{700}{kW}$ of design power.

When heat exchanger boundary conditions and mass flows are fixed, the hot side inlet properties of \qty{1.06}{\bar} and \qty{908}{\K} and mass flows of \qty{2.24}{\kg\per\s} are used, as obtained from the cycle model using the baseline heat exchanger performance of $60\%$ effectiveness and hot and cold side pressure drops of $6\%$ and $4\%$ respectively. To achieve this performance with the simple pressure drop model described in Section~\ref{sec:modelling_method}, it was necessary to assume substantially higher inlet Mach numbers than in either of the reference designs (as shown in Table~\ref{tab:baseline_parameters}).

For ease of reproducibility, both fluids are modelled as perfect gases with equal specific heat capacities, $c_p = \qty{1.07}{\kilo\joule\per\kilogram\per\kelvin}$ and $\gamma = 1.4$. This assumption explains the higher recuperated cycle efficiency shown in Table \ref{tab:baseline_parameters} for the baseline case relative to the redesign, despite the heat exchanger being worse on all metrics.

% The values that are not exact here are M_h_in = 0.1362, M_c_in = 0.0563, dp_c/p_cin = 0.04105, eps = 0.5966, T_h_in_fixed = 907.59, T_c_in_fixed = 587.78, p_h_in_fixed = 1.0638, mdot_fixed = 2.2356

\subsection{Aircraft and fuel burn model}\label{sec:fuel_burn_model}

The benefit of recuperation is modelled as a reduction in helicopter take-off mass relative to an unrecuperated baseline, comprising two elements: the change in fuel mass $\Delta m_{\mathrm{f}}$, and the addition of the heat exchanger mass. Both the recuperated and unrecuperated helicopters operate at design power $\dot{P}$ for an effective mission duration of $\tau = \qty{2}{\hour}$, with a fuel lower heating value of $H_{\mathrm{\ell}} = \qty{12}{\kWh\per\kg}$ (\qty{43.2}{\mega\joule\per\kg}). The fuel mass saving is related to the difference in thermal efficiencies of the recuperated and unrecuperated cycles, $\eta$ and $\eta_{\mathrm{u}}=34.5\%$:

\begin{equation}\label{eq:fuel*savings_cycle}
\Delta\dot{Q}*{\mathrm{in}} = \dot{P} \left( \frac{1}{\eta\_{\mathrm{u}}} - \frac{1}{\eta} \right)
\end{equation}

To maintain the design power $\dot{P}$ as cycle specific work varies, the mass flow rate is adjusted.
