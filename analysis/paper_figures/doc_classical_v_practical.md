# Exergy vs Euergy: Relationship and T-s Diagram Visualization

## Summary

Both **exergy** and **euergy** quantify unavailable work or work potential destruction in thermal systems. They use different formulations but share the same dead state \((T_d, p_d)\). For ideal gases, the key distinction is **exergy uses entropy change** while **euergy uses an isentropic-equivalent temperature change** at dead-state pressure. Plotting their difference on a T-s diagram emphasises temperature, which is central to heat transfer.

---

## Definitions (Ideal Gas, Dead State \(T_d\), \(p_d\))

### Exergy (entropy-based)

- **Unavailable work created** (destruction): \(\dot{m} T_d \, \Delta s\)
- For a stream: \(\Delta s = c_p \ln(T_2/T_1) - R \ln(P_2/P_1)\)
- Exergy destruction in a heat exchanger: \(T*d (s*{\mathrm{hot,out}} - s*{\mathrm{hot,in}} + s*{\mathrm{cold,out}} - s\_{\mathrm{cold,in}})\)

Exergy uses the total entropy change of both streams. It mixes thermal effects (driven by \(\Delta T\)) and pressure effects (driven by \(\Delta P\)).

### Euergy (temperature-based at \(p_d\))

- **Isentropic-equivalent temperature** at dead-state pressure:
  \[
  T\_{\mathrm{se}} = T \left( \frac{p_d}{P} \right)^{(\gamma-1)/\gamma}
  \]
  This is the temperature the fluid would have at \(p_d\) if brought there isentropically from \((T,P)\).

- **Unavailable work created**: \(\dot{m} c*p \, \Delta T*{\mathrm{se}}\)
- Euergy destruction in a heat exchanger: \(c*p \left[ (T*{\mathrm{se,h,out}} - T*{\mathrm{se,h,in}}) + (T*{\mathrm{se,c,out}} - T\_{\mathrm{se,c,in}}) \right]\) per unit mass

Euergy focuses on temperature changes along the **\(p_d\) isobar**, so it isolates the thermal (temperature) contribution and makes pressure-loss effects explicit via the \(T \to T\_{\mathrm{se}}\) mapping.

---

## Relationship Between Exergy and Euergy

- Both use the same dead state \((T_d, p_d)\).
- **Exergy**: \(T_d \cdot \Delta s\) — uses entropy change along the real \((T,P)\) path.
- **Euergy**: \(c*p \cdot \Delta T*{\mathrm{se}}\) — uses temperature change along the \(p_d\) isobar.

For an ideal gas, \(s\) and \(T\_{\mathrm{se}}\) are linked:

- \((T,P)\) and \((T\_{\mathrm{se}}, p_d)\) lie on the same isentrope, so they share the same \(s\).
- The projection \((T,P) \mapsto (T\_{\mathrm{se}}, p_d)\) moves each state onto the \(p_d\) isobar at the same \(s\).

The gap \(T - T*{\mathrm{se}}\) reflects the pressure penalty: as pressure drops, \(T*{\mathrm{se}} < T\), and euergy treats this as a loss of temperature potential at \(p_d\).

---

## Why Use \(T\) and Ideal Gases

1. **\(\Delta T\) drives heat transfer**: Heat flux scales with \(\Delta T\). Using \(T\) keeps the representation consistent with heat-exchanger physics.
2. **Ideal gas**: \(c_p\), \(\gamma\), and \(s = c_p \ln T - R \ln P + \mathrm{const}\) give simple, closed forms for both exergy and euergy.
3. **\(T\_{\mathrm{se}}\) is well-defined**: The isentropic relation \(T/T\_{\mathrm{se}} = (P/p_d)^{(\gamma-1)/\gamma}\) is exact for ideal gases.
4. **Interpretability**: On a T-s diagram, temperature is the vertical axis, so thermal driving forces and losses are visual.

---

## Plotting the Exergy–Euergy Difference on a T-s Diagram

### What to Plot

1. **Both streams of the heat exchanger**
   - Hot stream: \((s*{\mathrm{hot}}, T*{\mathrm{hot}})\) from inlet to outlet (including pressure drop).
   - Cold stream: \((s*{\mathrm{cold}}, T*{\mathrm{cold}})\) from inlet to outlet.

2. **Dead-state isobar \(p_d\)**
   - \(s = c*p \ln(T/T_d) - R \ln(p_d/p_d) + s*{\mathrm{ref}}\) → \(s = c_p \ln(T/T_d) + \mathrm{const}\)
   - Plot \(T\) vs \(s\) for this isobar (e.g. \(T = T*d \exp((s - s*{\mathrm{ref}})/c_p)\)).

3. **\(T\_{\mathrm{se}}\) points**
   - For each state \((T,P)\), compute \(T\_{\mathrm{se}} = T (p_d/P)^{(\gamma-1)/\gamma}\).
   - \((s, T\_{\mathrm{se}})\) lies on the \(p_d\) isobar (same \(s\), same isentrope).
   - Mark these points for both streams.

4. **Visualizing the difference**
   - At each entropy \(s\) along a stream, plot:
     - **Actual state**: \((s, T)\) — real process path.
     - **Euergy-equivalent state**: \((s, T\_{\mathrm{se}})\) — on the \(p_d\) isobar.
   - **Vertical segments** between \(T\) and \(T*{\mathrm{se}}\) at fixed \(s\) show the “pressure penalty” \(T - T*{\mathrm{se}}\).
   - Shaded bands or arrows along these segments make the difference clear.

### Implementation Steps

1. **State data**
   - For each stream: arrays of \(T\), \(P\), \(s\) (inlet → outlet), e.g. from cycle or HEx model.
   - Dead state: \(T_d\), \(p_d\); ideal gas: \(\gamma\), \(R\), \(c_p\).

2. **Entropy**
   - \(s = c_p \ln(T/T_d) - R \ln(P/p_d)\) (or any consistent reference).

3. **\(T\_{\mathrm{se}}\)**
   - \(T\_{\mathrm{se}} = T \cdot (p_d/P)^{(\gamma-1)/\gamma}\) for each \((T,P)\).

4. **T-s curves**
   - Hot and cold streams: `ax.plot(s_hot, T_hot)` and `ax.plot(s_cold, T_cold)`.
   - \(p_d\) isobar: `ax.plot(s_isobar, T_isobar)`.
   - \(T\_{\mathrm{se}}\) paths: `ax.plot(s_hot, T_se_hot)` and `ax.plot(s_cold, T_se_cold)` (or markers).

5. **Difference**
   - Option A: Vertical lines from \((s, T\_{\mathrm{se}})\) to \((s, T)\) at selected points.
   - Option B: Fill between \(T(s)\) and \(T\_{\mathrm{se}}(s)\) for each stream.
   - Option C: A second axis or annotation for \(\Delta W*{\mathrm{pot,Ex}}\) vs \(\Delta W*{\mathrm{pot,Eu}}\) at the HEx.

6. **Legend**
   - Actual paths (solid).
   - \(T\_{\mathrm{se}}\) paths or markers on \(p_d\) isobar.
   - \(p_d\) isobar.
   - Optional: “\(T - T\_{\mathrm{se}}\)” shaded region.

---

## Physical Interpretation on the T-s Diagram

- **Exergy destruction** \(\propto T_d \cdot \Delta s\): roughly the area under a horizontal line at \(T = T_d\) associated with total \(\Delta s\) of both streams.
- **Euergy destruction** \(\propto c*p \cdot \Delta T*{\mathrm{se}}\): change in temperature along the \(p_d\) isobar.
- **Vertical gap \(T - T\_{\mathrm{se}}\)**: reflects loss of temperature potential when pressure is below \(p_d\); larger gaps where pressure drop is large.
- For heat exchangers, \(\Delta T\) between streams drives heat transfer; plotting \(T\) and \(T\_{\mathrm{se}}\) makes it clear how pressure losses reduce the effective temperature difference from an euergy viewpoint.
