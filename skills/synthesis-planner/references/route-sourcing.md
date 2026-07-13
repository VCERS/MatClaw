# Sourcing the Route

Pick the synthesis route from the strongest evidence available. The reasoning: a
published recipe has been shown to work, so it comes first; absent that, a route
reasoned from solid-state chemistry and close analogues is a defensible starting
point. Confidence — and the human-review depth you recommend — follows from which
tier produced the route.

## Tier 1 — Literature (try first, always)

```
mp_search_recipe(
    target_formula="LiCoO2",     # reduced formula
    format_routes=True,          # return standardized, execution-ready routes
    limit=10,
    # optional constraint filters applied during the search:
    temperature_max=None, heating_time_max=None, synthesis_type=None,
    keywords=None, precursor_formulas=None,
)
```

- `format_routes=True` returns ready-to-use routes (`routes`, `n_routes`,
  `original_count`); `format_routes=False` returns raw recipe records (`recipes`,
  `count`) — use that only when you need to analyze or compare literature rather
  than plan.
- If `n_routes > 0`, you're on the high-confidence path. Use the top route as the
  recipe and the others as alternatives. Carry the `doi`/`original_count` into
  provenance.
- Apply the user's constraints as search filters (`temperature_max`,
  `synthesis_type`, `keywords`) rather than filtering after the fact.

If literature returns nothing, reason the route out (Tier 2).

## Tier 2 — Reasoned route (when there is no literature)

With no published recipe, design the route yourself from materials-chemistry
principles and analogous known systems. This is your core competency here — careful
chemical reasoning over close analogues is more reliable than a black-box predictor,
so invest in it:

- **Analyze the target** — element types and oxidation states, volatility, air/moisture
  sensitivity, and (most useful) the closest well-characterized compounds you know.
- **Choose a method** suited to the desired phase/morphology: conventional solid-state
  (mix–grind–calcine) for most bulk oxides and chalcogenides; sol-gel for homogeneous
  multi-cation oxides; hydrothermal for nanostructures or metastable phases.
- **Pick precursors** that decompose cleanly to the product — typically carbonates,
  oxides, nitrates, or hydroxides for the cations, mirroring what works for the
  analogues (e.g. Li2CO3 / La2O3 / ZrO2 for a Li–La–Zr garnet oxide). For a solid-state
  route, make sure the chosen precursors together supply every product element.
- **Set conditions** — temperature, atmosphere, dwell, ramp — bracketed from those
  analogous systems (e.g. ~900–1100 °C in air for many complex oxides; inert or sealed
  ampoule for sulfides and reduced phases).
- **Calibrate confidence to the analogues.** When the chemistry is well-precedented
  (a layered oxide, garnet, spinel, etc. near known members), this is a *medium*-
  confidence route. When the composition or method is genuinely novel or exotic, mark
  it *low*. Either way, state plainly that it is unvalidated for this exact composition,
  name the analogues you reasoned from, and recommend a small test batch (and, for low
  confidence, a literature search for analogues first).

## Decision summary

```
mp_search_recipe(format_routes=True)
        │ n_routes > 0 ?
   yes ─┴─▶ Tier 1: literature route   (confidence high,       review minimal)
    no
        └──▶ Tier 2: reasoned route     (confidence medium/low, review recommended/required)
                     medium if grounded in close, well-precedented analogues;
                     low if the composition or method is novel/exotic.
```

## Worked examples

**Well-studied (Tier 1).** `LiCoO2` → `mp_search_recipe` returns ~200 recipes →
top route: Li2CO3 + Co3O4, calcine 850 °C / 12 h / air, confidence high. Report
the route + alternatives + DOI.

**Novel solid-state, close analogues (Tier 2, medium).** `Li7La3Zr2O12` → no literature
→ reason from the garnet / complex-oxide family: cation precursors Li2CO3 + La2O3 + ZrO2
(a carbonate and oxides that fire cleanly to the oxide), mix–grind–calcine in air at
~900–1000 °C with a regrind, plus a Li-excess knob for Li volatility. Medium confidence;
note it is unvalidated for this exact composition and suggest a small test batch.

**Novel, weak analogues (Tier 2, low).** "Hydrothermal NiO nanowires" → no literature and
no close precedent for the exact morphology → reasoned route: Ni(NO3)2·6H2O + NaOH,
150–180 °C, 12–24 h autoclave, surfactant/template for morphology, pH-sensitive. Low
confidence; review required; recommend a literature search for analogous nanostructures
first.
