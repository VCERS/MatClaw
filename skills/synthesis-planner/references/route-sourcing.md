# Sourcing the Route

Pick the synthesis route from the strongest evidence available, falling back
deliberately and labeling which tier you used. The reasoning: a published recipe
has been shown to work; an ML prediction is statistically plausible but unproven
for this exact composition; a chemistry-principles guess is only a starting
point. Confidence — and the human-review depth you recommend — follows directly
from which tier produced the route.

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

If literature returns nothing, fall through.

## Tier 2 — ML prediction (solid-state inorganic only)

Only valid when the route is, or can reasonably be, **solid-state**. The models
are trained on solid-state literature; using them for hydrothermal/sol-gel/CVD
will produce confident nonsense. If the user explicitly asked for a non-solid-state
method, skip to Tier 3.

```
prec = er_predict_precursors(target_formula="Li7La3Zr2O12", top_k=5)
# → {"top_prediction": {"precursors": [...], "confidence": 0.61},
#    "precursor_sets": [ {precursors, confidence}, ... ]}

temp = er_predict_temperature(
    target_formula="Li7La3Zr2O12",
    precursors=prec["top_prediction"]["precursors"],
)
# → {"temperature_celsius": 908.3, ...}
```

- If `top_prediction.confidence < ~0.2`, treat the prediction as unreliable and
  fall through to Tier 3 (or present it only as a weak suggestion).
- `er_predict_temperature` validates that precursor elements cover the target's
  elements and raises `ValueError` on mismatch — if the top precursor set errors,
  try the next set in `precursor_sets` before giving up.
- Build a standard solid-state route: mix & grind → calcine at the predicted
  temperature (air for oxides) → (optional regrind/re-fire). Medium confidence;
  attach the precursor confidence and the ±50–100 °C temperature uncertainty as
  warnings.

## Tier 3 — Reasoned default (last resort)

When there's no literature and ML doesn't apply (non-solid-state target, or
ML failed/low-confidence), propose a route from materials-chemistry principles:

- Analyze the target — element types, oxidation states, volatility, moisture/air
  sensitivity, analogous known compounds.
- Choose a method appropriate to the morphology/phase (e.g. hydrothermal for
  nanostructures, sol-gel for homogeneous multi-cation oxides).
- Name plausible precursors (common salts/oxides/hydroxides), a temperature range
  from analogous systems, and the key processing steps.
- Mark confidence **low** and review **required**. Be explicit that it is
  unvalidated for this exact composition.

## Decision summary

```
mp_search_recipe(format_routes=True)
        │ n_routes > 0 ?
   yes ─┴─▶ Tier 1: literature route        (confidence high,   review minimal)
    no
        │ solid-state AND ML applicable ?
   yes ─┴─▶ er_predict_precursors → er_predict_temperature
        │        confidence ≥ ~0.2 ?
   yes ─┴─▶ Tier 2: ML route                (confidence medium, review recommended)
    no
        └──▶ Tier 3: reasoned default       (confidence low,    review required)
```

## Worked examples

**Well-studied (Tier 1).** `LiCoO2` → `mp_search_recipe` returns ~200 recipes →
top route: Li2CO3 + Co3O4, calcine 850 °C / 12 h / air, confidence high. Report
the route + alternatives + DOI.

**Novel solid-state (Tier 2).** `Li7La3Zr2O12` → no literature → `er_predict_precursors`
→ Li2CO3 + La2O3 + ZrO2 (0.61) → `er_predict_temperature` → ~908 °C. Medium
confidence; warn on ±50–100 °C and unproven composition; suggest a small test batch.

**Non-solid-state, novel (Tier 3).** "Hydrothermal NiO nanowires" → no literature,
ML inapplicable → reasoned route: Ni(NO3)2·6H2O + NaOH, 150–180 °C, 12–24 h
autoclave, surfactant/template for morphology, pH-sensitive. Low confidence; review
required; recommend literature search for analogous nanostructures first.
