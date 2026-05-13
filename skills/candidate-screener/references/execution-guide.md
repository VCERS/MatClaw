# Screening Execution Guide

Use this reference when the main skill is not enough to decide execution order, recover from failures, or scale a screening run.

---

## When to Read This

Open this file when you need any of the following:
- The full step order for a screening run
- Decision logic for relaxation, source selection, or polymorph handling
- Failure handling that preserves candidates instead of silently dropping them
- Checkpointing guidance for screening more than 20 candidates

If you only need disorder handling, read `preprocessing-guide.md` instead.
If you only need property-model selection, read `ml-calculations-guide.md` instead.

---

## Canonical Workflow

Follow this order for each run:

1. **Preprocess disorder when present**
   - Ordered structures continue directly.
   - Disordered structures go through majority, enumeration, or SQS ordering based on the metadata or concentration heuristic.

2. **Validate before enrichment**
   - Run structural validation and composition analysis first.
   - Reject only when the structure is genuinely unusable.
   - Record rejection reasons so criteria can be refined later.

3. **Retrieve properties hierarchically**
   - Try Materials Project first for highest-confidence DFT-quality data.
   - Fall back to ASE cache when equivalent properties were already computed.
   - Run ML calculations only when the higher-confidence sources are unavailable.

4. **Filter against screening criteria**
   - Apply thresholds only after the candidate has the required properties.
   - Keep explicit failure reasons for every rejected candidate.

5. **Rank surviving candidates**
   - Use multi-objective ranking.
   - Weight scores by source confidence so DFT-backed values outrank approximations.
   - Flag strong ML-only hits for DFT verification before expensive downstream work.

6. **Persist outputs**
   - Store retrieved properties in ASE.
   - Preserve both original and relaxed structures when relaxation was used.
   - Emit a screening report with counts, provenance, rankings, and failure reasons.

---

## Core Decisions

### Relaxation Before Prediction

Use relaxed structures for ML predictions unless the structure already comes from a trusted optimized source such as DFT, Materials Project, or validated experimental geometry. This matters because the ML models are generally trained on near-equilibrium geometries.

### Source Priority

Prefer this order:
- Materials Project
- ASE cache
- ML prediction

This keeps the workflow calibrated toward higher-confidence data and reduces unnecessary recomputation.

### Multiple Materials Project Matches

If several polymorphs match the same formula:
- Keep all of them when metastability or polymorph competition is part of the screening question.
- Otherwise prefer the lowest-energy structure and record that the choice was made.

### ML Failure Handling

If an ML step fails:
- Try a fallback model if one is available for that property.
- If no fallback succeeds, retain the candidate with an error flag.
- Mark it for DFT or manual follow-up rather than dropping it.

The key behavior is transparency: failures should remain visible in the report.

---

## Large-Batch Execution

For more than 20 candidates, switch from an ad hoc loop to a checkpointed run.

### What to Track

Track at least these fields per candidate:
- Validation status
- Property source for each retrieved value
- Relaxation status
- Filter pass or fail
- Ranking inputs and final score
- Errors or follow-up flags

### Why Checkpointing Matters

Checkpointing helps because:
- ML relaxations and matcalc jobs are the slowest part of the workflow
- A partial run is still useful if the user wants to refine criteria midstream
- Cached results should survive interruptions so the run can resume cheaply

### Batch Pattern

Use this operational pattern:
- Build a tracking JSON before execution
- Save after each candidate completes a major stage
- Reuse cached properties when criteria change
- Re-run only filtering and ranking when the underlying property set is unchanged

---

## Failure Modes Worth Preserving

These cases should stay visible in outputs rather than disappearing from the run:
- Invalid structures rejected during validation
- ML property calculations that failed
- Approximate ordered representations generated from disorder heuristics
- High-scoring ML-only candidates that still need DFT confirmation
- Candidates excluded by thresholds that the user may want to relax later

This makes the workflow auditable and supports iteration instead of forcing a full rerun.

---

## Practical Defaults

- Use Materials Project as the first lookup whenever possible.
- Use MatGL for rapid formation-energy and band-gap screening.
- Use matcalc only for properties that need its structure-based calculations.
- Save relaxed structures whenever relaxation occurs.
- Treat screening as enrichment plus ranking, not as a lossy filter.