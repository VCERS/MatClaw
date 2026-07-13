# Handoff Contracts

This skill produces an artifact consumed by two downstream skills. Keeping the
shapes stable and engine-agnostic is what lets the pipeline compose without glue
code. You don't call those skills — you produce the structure they read.

## 1 → Experiment-orchestration / robot-middleware skill

This layer translates the recipe into machine protocols and runs it (behind a
human gate). Give it a **method-agnostic, fully specified procedure**: it should
not have to infer chemistry, only map your `action` + `parameters` onto hardware.

```
"recipe": {
  "precursors": [
    {
      "compound": "Li2CO3",          # exact formula (incl. hydrate if relevant)
      "role": "Li source",
      "form": "powder",              # powder | solution | pellet | ...
      "stoichiometric_amount": 0.5,  # per formula unit of target
      "mass_grams": 3.69,            # absolute mass IF quantified (else null)
      "hazards": []                  # optional, surfaced for handling
    }
  ],
  "steps": [
    {"step": 1, "action": "mix_grind",  "parameters": {"duration_min": 20, "medium": "agate"}},
    {"step": 2, "action": "pelletize",  "parameters": {"pressure_mpa": 200}},
    {"step": 3, "action": "calcine",
       "parameters": {"temperature_c": 850, "time_h": 12, "atmosphere": "air", "ramp_c_per_min": 5}},
    {"step": 4, "action": "characterize", "parameters": {"technique": "XRD"}}
  ],
  "product_target": {"formula": "LiCoO2", "batch_size_grams": 10.0}
}
```

Guidelines:
- Use **verbs the middleware can map**: `mix_grind`, `pelletize`, `calcine`,
  `anneal`, `dissolve`, `stir`, `heat_autoclave`, `dry`, `wash`, `characterize`.
  Keep parameters as flat key/values with explicit units in the key
  (`temperature_c`, `time_h`, `ramp_c_per_min`).
- Set `mass_grams` only when a batch size is known and you ran
  `synthesis_recipe_quantifier`; otherwise leave it `null` and let the
  orchestration layer quantify at its chosen batch size.
- Do **not** include execution approval. Confidence and review depth live at the
  top level (see safety.md); the human gate at the orchestration layer decides.

## 2 → Active-learning (Bayesian optimization) skill

This skill iterates conditions to maximize an objective measured after each run
(typically XRD phase purity). Give it the **initial point** and the **search
space** — exactly what an optimizer needs to propose the next experiment.

```
"optimization": {
  "objective": "phase purity by XRD (maximize)",
  "variables": [
    {"name": "calcination_temperature_c", "initial": 850, "bounds": [750, 950], "type": "continuous"},
    {"name": "calcination_time_h",        "initial": 12,  "bounds": [4, 24],    "type": "continuous"},
    {"name": "Li_excess_pct",             "initial": 0,   "bounds": [0, 12],    "type": "continuous"},
    {"name": "atmosphere", "initial": "air", "choices": ["air", "O2"], "type": "categorical"}
  ]
}
```

Guidelines:
- Each `variable.name` should match a tunable field used in `recipe.steps[*].parameters`
  (or a stoichiometry offset), so the optimizer's suggestion maps cleanly back onto a
  concrete next recipe.
- `initial` is the recipe's nominal value; `bounds`/`choices` define the explorable
  range (see parameterization.md for how to set them).
- Keep the variable set small (≈3–6). State the `objective` explicitly so the
  optimizer and the characterization step agree on the target.

## Closing the loop
The orchestration skill runs the recipe → characterizes → the active-learning skill
reads the result and proposes the next point within `bounds` → the orchestration
skill runs that. Your artifact seeds round 1 of that loop; you are not part of the
loop itself.
