# Confidence, Review Depth, and Red Flags

Your recipe carries a confidence level and a recommended human-review depth. These
are advisory signals for the human gate at the orchestration layer — this skill
never authorizes autonomous execution. Be honest: the level must match the evidence
tier that produced the route (see route-sourcing.md), because a downstream human or
robot will calibrate trust on it.

## Confidence tiers and the review they imply

| Source tier | Confidence | `review` | What it means |
|-------------|-----------|----------|---------------|
| Literature (Materials Project) | high | minimal | A published, community-validated procedure. Still: confirm precursor purity/availability and run a small test batch. |
| ML prediction (solid-state) | medium | recommended | Statistically plausible, not validated for this composition. ±50–100 °C on temperature. Small-scale test and a literature cross-check before scale-up. |
| Reasoned default | low | required | Chemistry-principles starting point only. Expert review, a literature search for analogues, and a small test are prerequisites. |

`review` is a *recommended depth of human scrutiny*, not an execution verdict. Even
"minimal" assumes a human signs off at the orchestration layer.

## Red flags — raise caution (add a warning, bump review depth)

**Composition / chemistry:**
- > 4 elements (more competing phases, harder to hit the target).
- Volatile elements (Li, Na, K, Pb, Bi, P) — loss during firing; include an excess knob.
- Air/moisture-sensitive species — handling and atmosphere constraints.
- Mixed/uncertain oxidation states (many transition metals, lanthanides).
- Known competing or intermediate phases in the system.

**Route confidence:**
- ML precursor confidence < 0.4, or elements poorly represented in training data.
- Temperatures, times, or precursor combinations far outside what analogous
  literature uses.
- Conflicting or unusual atmosphere requirements.

**Always include in warnings when relevant:**
- the evidence tier and its limitation ("ML-predicted, not validated for this
  composition"),
- quantitative uncertainty (e.g. "temperature ±50–100 °C"),
- a recommendation to test at small scale before committing a full batch,
- any precursor hazards worth surfacing to the handling layer.

## Quantification caveat
When `synthesis_recipe_quantifier` sets precursor masses, treat them as arithmetic
to be re-checked at the bench — in particular verify water-of-crystallization is
handled for hydrated precursors (e.g. `Ni(NO3)2·6H2O`), since an unaccounted hydrate
changes the molar mass and therefore the weighed amount.
