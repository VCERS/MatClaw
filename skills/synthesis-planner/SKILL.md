---
name: synthesis-planner
description: >
  Generate a good first-guess synthesis recipe for an inorganic material as the seed of a lab campaign. Sources the route by best available evidence: literature-validated recipes (Materials Project) first, then a route reasoned from solid-state chemistry and analogous known systems. Produces a structured, robot-middleware-ready recipe (precursors with forms, ordered steps with parameters), defines the tunable parameter search space for downstream Bayesian-optimization refinement, and can quantify precursor masses for a target batch size. Use this skill WHENEVER the user wants a synthesis route, recipe, or procedure; asks how to make / prepare / synthesize a compound or formula; needs precursors, calcination/annealing conditions, or a solid-state route; or when a DFT-validated candidate needs a first synthesis plan to hand to a robotic lab. It does NOT execute synthesis or approve autonomous runs — a separate experiment-orchestration skill plus a human gate own that — and it does NOT run the optimization loop itself (the active-learning skill does).
---

# Synthesis Planner Skill

Produce a **first-guess synthesis recipe** for an inorganic target — the precursors,
method, and conditions most likely to work — packaged so two downstream skills can
take it forward:

- an **experiment-orchestration** skill that turns the recipe into
  machine protocols and runs it (with a human gate), and
- an **active-learning** skill that refines the conditions via Bayesian optimization
  over successive experiments.

So your output is not a final answer — it is the **seed of an optimization campaign**.
That framing drives every design choice below: a route a robot can execute, plus an
explicit parameter search space the optimizer can move in.

**Input:** a target formula (often a high-confidence candidate handed over after DFT),
with optional constraints (method, temperature ceiling, batch size).
**Output:** a structured recipe + a tunable parameter space + provenance and confidence.

---

## Scope: where this skill stops

Be clear about the boundary — it keeps the pipeline clean and safe:

- **You plan; you do not execute.** Never mark a recipe "approved for autonomous
  execution." Emit a recipe + a confidence level + a recommended human-review depth,
  and hand off. The orchestration skill plus a human decide whether to run it.
- **You seed; you do not optimize.** Don't iterate conditions yourself. Define the
  initial point and the bounds; the active-learning skill explores them.

This separation is *why* the output has both a concrete recipe (for the robot) and a
parameter space (for the optimizer), rather than a single take-it-or-leave-it route.

## Tools

| Tool | Role |
|------|------|
| `mp_search_recipe` | Literature-validated recipes from Materials Project. **Try this first** — it is the only sourcing tool. If it returns nothing, you reason the route out yourself (no tool for that tier — it's your chemistry judgment). |
| `synthesis_recipe_quantifier` | Convert stoichiometry → precursor **masses (g)** for a target batch size. |

Upstream context often comes from `candidate-screener` / `vasp` / `orca` (a DFT-validated
formula to make). Downstream you hand to the experiment-orchestration and `active-learning`
skills.

---

## Workflow

### 1. Intake
Establish the target and the campaign context:
- the **target formula** (normalize to a reduced formula, e.g. `LiCoO2`, not `Li1Co1O2`);
- any **constraints** the user gave: method (solid-state / hydrothermal / sol-gel …),
  temperature ceiling, time, atmosphere, batch size;
- the **downstream intent** — is this headed for a robotic run and a BO loop? (Usually
  yes in this pipeline; it tells you to populate the parameter space carefully.)

### 2. Source the route — by best available evidence
Always start from literature; if there is none, reason the route out yourself. The
principle: *a published recipe is proven; a route reasoned from solid-state chemistry
and close analogues is a strong, defensible starting point* — so prefer the strongest
evidence available and be honest about which tier you landed on. Full algorithm,
tool arguments, and worked examples in
[references/route-sourcing.md](references/route-sourcing.md). In brief:

1. **Literature first** — `mp_search_recipe(target_formula, format_routes=True)`. If it
   returns routes, use them; this is the high-confidence path.
2. **Reasoned route** — if there's no literature, design the route from
   materials-chemistry principles and analogous known systems: pick precursors (common
   carbonates, oxides, nitrates, or hydroxides that decompose cleanly to the product
   and together supply every product element), a method suited to the phase, and
   conditions (temperature, atmosphere, time) bracketed from compounds known to behave
   similarly. Calibrate confidence to the analogues — *medium* when the chemistry is
   well-precedented (e.g. a layered oxide or garnet near known ones), *low* when it is
   novel or exotic. Name the analogues you reasoned from and require human review.

Careful chemical reasoning over close analogues is the heart of this tier — it is more
reliable than a black-box predictor, so invest in it rather than reaching for a shortcut.

### 3. Parameterize for optimization
A first guess that's a single fixed recipe wastes the lab loop. Identify the **knobs the
active-learning skill should tune**, each with an **initial value** and **plausible
bounds**, derived from the route you sourced. Typical knobs: calcination temperature,
dwell time, ramp rate, atmosphere, and stoichiometry offsets for volatile elements
(e.g. Li/Na excess). How to choose values and bounds per method is in
[references/parameterization.md](references/parameterization.md). This parameter space
is the single most important thing you add on top of the raw route.

### 4. Quantify (optional, when a batch size is known)
If the user or the orchestration layer specifies a batch size, call
`synthesis_recipe_quantifier(recipes, target_batch_size_grams, target_formula)` to turn
stoichiometry into **masses to weigh out**. Treat the masses as deterministic arithmetic
to be re-checked at the bench (verify hydrate/water-of-crystallization handling for
precursors like `Ni(NO3)2·6H2O`).

### 5. Hand off
Return the recipe in the structured shape below so the next skills can consume it
directly. Contracts and field meanings in
[references/handoff-contracts.md](references/handoff-contracts.md).

---

## Output shape

Return a single object with these parts (fill the fields; this describes intent and
structure, not a rigid literal — include what you have, omit what you don't):

```
{
  "target_formula": "...",
  "source": "literature | reasoned",             # which evidence tier you used
  "confidence": "high | medium | low",
  "method": "solid_state | hydrothermal | sol-gel | ...",

  "recipe": {                                      # → experiment-orchestration skill
    "precursors": [
      {"compound": "Li2CO3", "role": "Li source", "form": "carbonate",
       "stoichiometric_amount": 0.5, "mass_grams": null}   # mass set if quantified
    ],
    "steps": [
      {"step": 1, "action": "mix_grind", "parameters": {}},
      {"step": 2, "action": "calcine",
       "parameters": {"temperature_c": 850, "time_h": 12, "atmosphere": "air",
                      "ramp_c_per_min": 5}}
    ]
  },

  "optimization": {                                # → active-learning (BO) skill
    "objective": "phase purity by XRD (maximize)",
    "variables": [
      {"name": "calcination_temperature_c", "initial": 850, "bounds": [750, 1000], "type": "continuous"},
      {"name": "calcination_time_h",        "initial": 12,  "bounds": [4, 24],     "type": "continuous"},
      {"name": "Li_excess_pct",             "initial": 0,   "bounds": [0, 15],     "type": "continuous"},
      {"name": "atmosphere", "initial": "air", "choices": ["air", "O2", "Ar"], "type": "categorical"}
    ]
  },

  "provenance": {"tool": "mp_search_recipe", "doi": "...", "n_literature_recipes": 206},
  "review": "minimal | recommended | required",    # human-review depth, NOT an execution approval
  "warnings": ["..."]
}
```

Keep it honest: the `confidence`, `review`, and `warnings` fields must reflect which tier
sourced the route. See [references/safety.md](references/safety.md) for the confidence
tiers, the review depth each implies, and the red flags that should raise caution.

## Reference files
- [references/route-sourcing.md](references/route-sourcing.md) — the tiered algorithm, tool arguments, decision points, and worked examples (literature / reasoned).
- [references/parameterization.md](references/parameterization.md) — choosing tunable knobs, initial values, and bounds per synthesis method; building the BO search space.
- [references/handoff-contracts.md](references/handoff-contracts.md) — the recipe artifact for the robot-middleware layer and the search-space schema for the active-learning skill.
- [references/safety.md](references/safety.md) — confidence tiers, human-review depth, and red-flag conditions.
