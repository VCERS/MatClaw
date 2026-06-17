# Parameterizing the Recipe for Optimization

The first guess fixes a *starting point*; the active-learning loop improves on it
by varying conditions across experiments. Your job here is to expose the right
**knobs** with sensible **initial values** and **plausible bounds**, so the
Bayesian-optimization skill has a well-posed search space. A recipe with no
parameter space turns the autonomous lab into a single one-shot experiment —
which defeats the loop.

## What makes a good knob
A variable belongs in the search space if it (a) plausibly affects the outcome
(phase purity, yield, crystallinity, morphology), (b) is controllable by the
synthesis hardware, and (c) has a physically reasonable range. Keep the set small
(≈3–6) — every extra dimension costs experiments. Prefer the knobs most likely to
matter for *this* chemistry over an exhaustive list.

## Initial value vs. bounds
- **Initial value:** the route's nominal setting (a literature value, or a reasoned
  estimate from analogous systems). This is where the optimizer starts.
- **Bounds:** the range worth exploring — wide enough to contain the likely
  optimum, narrow enough to stay physical and safe. Anchor bounds to the evidence:
  for a literature route, bracket the value seen in the literature; for a reasoned
  route, use the typical range seen in analogous systems (and widen a little to
  reflect that the value is an estimate, not a measurement).
- Mark each variable `continuous` or `categorical`, and include `choices` for
  categoricals.

## Common knobs by method

**Solid-state (calcination):**
| Knob | Typical initial | Typical bounds | Notes |
|------|-----------------|----------------|-------|
| `calcination_temperature_c` | route value | ±100–150 °C around it | dominant variable |
| `calcination_time_h` | 8–12 | 2–24 | dwell at temperature |
| `ramp_rate_c_per_min` | 5 | 1–10 | affects phase formation/cracking |
| `atmosphere` | air (oxides) | {air, O2, Ar, N2, forming gas} | categorical; redox-sensitive phases |
| `<A>_excess_pct` | 0 | 0–15 | excess of a **volatile** element (Li, Na, K, Pb) to offset loss |
| `n_intermediate_regrinds` | 0–1 | 0–3 | homogeneity for multi-cation oxides |

**Hydrothermal / solvothermal:**
temperature (100–250 °C), time (6–48 h), fill fraction, pH / mineralizer
concentration, precursor concentration, surfactant/template amount (morphology).

**Sol-gel:**
chelating-agent : metal ratio, gelation pH, aging time, calcination temperature/time.

## Picking the volatile-element knob
If the target contains a volatile cation (Li, Na, K, Pb, Bi, P), a small **excess**
of its precursor is one of the highest-value knobs — loss during firing is the most
common reason a solid-state route misses the target phase. Include an
`<element>_excess_pct` variable (initial 0, bounds 0–15%) when such an element is
present.

## Objective
State what the loop is optimizing — usually **phase purity** measured by XRD
(the active-learning skill's `xrd_analyze_pattern` characterization), sometimes
yield, crystallite size, or a target property. The objective belongs in the
handoff so the optimizer and the characterization step agree on the target.

## Example (solid-state oxide)
For a literature LiCoO2 route (850 °C / 12 h / air):
```
"optimization": {
  "objective": "phase purity by XRD (maximize)",
  "variables": [
    {"name": "calcination_temperature_c", "initial": 850, "bounds": [750, 950],  "type": "continuous"},
    {"name": "calcination_time_h",        "initial": 12,  "bounds": [4, 24],     "type": "continuous"},
    {"name": "Li_excess_pct",             "initial": 0,   "bounds": [0, 12],     "type": "continuous"},
    {"name": "atmosphere", "initial": "air", "choices": ["air", "O2"], "type": "categorical"}
  ]
}
```
Bounds bracket the literature value; `Li_excess_pct` is included because Li is
volatile at calcination temperatures.
