# Large-Scale Screening (>20 Candidates)

When screening **many candidates** (>20 structures), **ALWAYS create a screening tracking file FIRST** before executing the workflow. This enables:
- **Checkpointing** after each candidate (property retrieval can fail/timeout)
- **Resume capability** (ML relaxations take 5-10s per structure = 8-15 min for 100)
- **Progress monitoring** (track validation, property sources, screening results)
- **Iterative refinement** (adjust screening criteria based on partial results)
- **Audit trail** (which properties came from MP vs ASE vs ML)

## When to Use Screening Tracking

**Trigger conditions:**
- User provides >20 candidates for screening
- Screening requires expensive ML calculations (elasticity, phonons, surfaces)
- Screening criteria may need adjustment after initial results
- User requests comprehensive property enrichment

**Skip tracking for:**
- Quick screenings (<10 candidates)
- All properties available in Materials Project (fast lookups)
- Single-round screening with fixed criteria

---

## Screening Tracking Workflow

### Step 1: Create Screening Plan

Generate a structured JSON tracking file that records:
- Input candidates and screening criteria
- Per-candidate validation, property retrieval, and screening results
- Status tracking for resume capability
- Source attribution (MP/ASE/ML) for confidence assessment

**Template screening plan:**
```json
{
  "metadata": {
    "screening_date": "2026-04-28",
    "input_source": "lanthanide_niobate_candidates_100.json",
    "ase_database": "screening_ln_niobates_20260428.db",
    "total_candidates": 100,
    "status": "planning",
    "screening_criteria": {
      "essential": {
        "structure_valid": true,
        "composition_valid": true,
        "energy_above_hull_max": 0.3,
        "formation_energy_required": true
      },
      "application_specific": {
        "band_gap_min": 3.0,
        "band_gap_max": 5.0,
        "mechanical_stability_required": false,
        "phonon_stability_required": false
      }
    },
    "property_hierarchy": ["Materials_Project", "ASE_cached", "ML_calculated"],
    "ml_settings": {
      "relaxation_fmax": 0.1,
      "relaxation_max_steps": 500,
      "eform_model": "M3GNet-MP-2018.6.1-Eform",
      "bandgap_model": "MEGNet-MP-2019.4.1-BandGap-mfi"
    }
  },
  "candidates": [
    {
      "id": "CAND-001",
      "formula": "Sr0.95Eu0.05Nb2O6",
      "input_structure": {
        "cif": "...",
        "ase_db_source": "lanthanide_niobate_candidates.db",
        "ase_db_id": 1
      },
      "status": "not_started",
      "validation": {
        "structure_valid": null,
        "composition_valid": null,
        "validation_issues": [],
        "duplicate_of": null,
        "timestamp": null
      },
      "properties": {
        "formation_energy_per_atom": null,
        "energy_above_hull": null,
        "band_gap": null,
        "space_group": null,
        "is_stable": null,
        "property_sources": {},
        "timestamp": null
      },
      "screening_result": {
        "passed": null,
        "failed_criteria": [],
        "rejection_reason": null,
        "requires_dft": false
      },
      "ranking": {
        "rank": null,
        "composite_score": null,
        "objective_scores": {}
      }
    }
    // ... 99 more candidates
  ],
  "execution_log": [],
  "summary_statistics": {
    "not_started": 100,
    "validation_in_progress": 0,
    "validation_passed": 0,
    "validation_failed": 0,
    "properties_in_progress": 0,
    "properties_complete": 0,
    "properties_failed": 0,
    "screening_passed": 0,
    "screening_failed": 0,
    "ranked": 0,
    "property_source_breakdown": {
      "Materials_Project": 0,
      "ASE_cached": 0,
      "ML_calculated": 0
    }
  }
}
```

**Key tracking fields:**
- `status`: `"not_started"` → `"validating"` → `"validated"` → `"retrieving_properties"` → `"properties_complete"` → `"screening_complete"` → `"ranked"` (or `"rejected"` at any stage)
- `validation`: Structure and composition validation results
- `properties.property_sources`: Maps each property to its source (`{"formation_energy_per_atom": "Materials_Project", "band_gap": "ML_calculated"}`)
- `screening_result.failed_criteria`: List of criteria that failed (e.g., `["band_gap_too_low", "unstable"]`)
- `requires_dft`: Flag for high-priority candidates needing DFT verification
- `execution_log`: Timestamped events (validation failed, MP lookup timeout, ML prediction succeeded, etc.)

---

### Step 2: Present Screening Plan to User

**ALWAYS show the user:**
1. Total candidates and screening criteria
2. Expected property sources (estimate MP vs ML percentages)
3. Estimated runtime based on property retrieval needs
4. Option to adjust criteria before execution

**Example output:**
```
Generated screening tracking file: screening_ln_niobates_100.json

Screening Plan Summary:
───────────────────────────────────────────────────
Input: 100 lanthanide-doped niobate candidates
Database: screening_ln_niobates_20260428.db

Essential Criteria:
  ✓ Structure validation (reject invalid)
  ✓ Composition analysis (reject unbalanced)
  ✓ Energy above hull ≤ 0.3 eV/atom
  ✓ Formation energy required

Application-Specific Criteria (Phosphor Screening):
  ✓ Band gap: 3.0 - 5.0 eV (for UV excitation)
  ✗ Mechanical properties: not required
  ✗ Phonon stability: not required

Property Retrieval Strategy:
  1st priority: Materials Project (DFT quality)
     → Estimated: ~15% of candidates (common niobates)
  2nd priority: ASE cache (previous screenings)
     → Estimated: ~5% (if rerun)
  3rd priority: ML calculations (MatGL + matcalc)
     → Estimated: ~80% (novel doped compositions)

Estimated Runtime:
  - Validation (100 candidates): ~1 minute
  - MP lookups (15 hits): ~5 seconds
  - ML calculations (80 candidates):
    * Structure relaxation: ~10 minutes
    * Property predictions: ~2 minutes
  - Ranking: ~10 seconds
  TOTAL: ~15 minutes (with checkpointing enabled)

Review screening_ln_niobates_100.json and confirm to proceed.
Type 'proceed' to start screening.
```

**Wait for user approval.** User may:
- Approve as-is
- Adjust criteria (e.g., relax band gap range to 2.5-5.5 eV)
- Skip expensive calculations (e.g., disable phonon checks)
- Abort if criteria don't match application needs

---

### Step 3: Execute with Checkpointing

**Production Implementation: Use Batch Client Utility**

For production execution, use the reusable batch client utility instead of manual loops:

```python
from MatClaw.skills.utils.batch_client import MCPBatchClient, setup_logging
from pathlib import Path
from datetime import datetime
import asyncio

setup_logging()

class CandidateScreeningClient(MCPBatchClient):
    """Screening client with multi-phase workflow: validation → properties → screening → ranking"""
    
    def __init__(self, plan_file: Path, **kwargs):
        super().__init__(**kwargs)
        self.plan_file = plan_file
        self.plan = None
        self.ase_db = None
        
    def load_plan(self):
        """Load screening plan from JSON file."""
        with open(self.plan_file) as f:
            self.plan = json.load(f)
        self.ase_db = self.plan["metadata"]["ase_database"]
    
    def save_plan(self):
        """Save updated plan to JSON file."""
        with open(self.plan_file, 'w') as f:
            json.dump(self.plan, f, indent=2)
    
    async def process_item(self, candidate: dict, context: dict) -> bool:
        """
        Process one candidate through all screening phases.
        Returns True if processing succeeded, False if failed/rejected.
        """
        try:
            # PHASE 1: Validation (if not already validated)
            if candidate["status"] == "not_started":
                if not await self._validate_candidate(candidate):
                    return False  # Rejected during validation
            
            # PHASE 2: Property Retrieval (if not already complete)
            if candidate["status"] == "validated":
                if not await self._retrieve_properties(candidate):
                    return False  # Failed to retrieve properties
            
            # PHASE 3: Screening (if properties complete)
            if candidate["status"] == "properties_complete":
                await self._apply_screening_criteria(candidate)
            
            return True
            
        except Exception as e:
            candidate["status"] = "rejected"
            candidate["screening_result"]["rejection_reason"] = f"Processing error: {str(e)}"
            self.plan["execution_log"].append({
                "timestamp": datetime.now().isoformat(),
                "candidate_id": candidate["id"],
                "event": "error",
                "message": str(e)
            })
            return False
    
    async def _validate_candidate(self, candidate: dict) -> bool:
        """Phase 1: Validate structure and composition."""
        candidate["status"] = "validating"
        
        # Structure validation
        val_result = await self.call_tool(
            "structure_validator",
            {"input_structure": candidate["input_structure"]["cif"]}
        )
        
        candidate["validation"]["structure_valid"] = val_result["is_valid"]
        
        if not val_result["is_valid"]:
            candidate["status"] = "rejected"
            candidate["validation"]["validation_issues"] = val_result["issues"]
            candidate["screening_result"]["rejection_reason"] = f"Invalid structure: {val_result['issues']}"
            self.plan["summary_statistics"]["validation_failed"] += 1
            return False
        
        # Composition analysis
        comp_result = await self.call_tool(
            "composition_analyzer",
            {"input_structure": candidate["input_structure"]["cif"]}
        )
        
        candidate["validation"]["composition_valid"] = not comp_result.get("errors", False)
        candidate["properties"]["space_group"] = comp_result.get("spacegroup", "unknown")
        
        if comp_result.get("errors"):
            candidate["status"] = "rejected"
            candidate["validation"]["validation_issues"].append("composition_invalid")
            candidate["screening_result"]["rejection_reason"] = "Invalid composition"
            self.plan["summary_statistics"]["validation_failed"] += 1
            return False
        
        # Validation passed
        candidate["status"] = "validated"
        candidate["validation"]["timestamp"] = datetime.now().isoformat()
        self.plan["summary_statistics"]["validation_passed"] += 1
        return True
    
    async def _retrieve_properties(self, candidate: dict) -> bool:
        """Phase 2: Hierarchical property retrieval (MP → ASE → ML)."""
        candidate["status"] = "retrieving_properties"
        
        # Try Materials Project first
        mp_result = await self.call_tool(
            "mp_search_materials",
            {"formula": candidate["formula"], "limit": 5}
        )
        
        if mp_result.get("success") and mp_result.get("count", 0) > 0:
            # Found in MP - retrieve detailed properties
            mp_id = mp_result["materials"][0]["material_id"]
            props = await self.call_tool(
                "mp_get_material_properties",
                {
                    "material_ids": [mp_id],
                    "properties": ["formation_energy_per_atom", "band_gap", "energy_above_hull", "is_stable"]
                }
            )
            
            self._store_properties(candidate, props[0], "Materials_Project")
            candidate["status"] = "properties_complete"
            self.plan["summary_statistics"]["properties_complete"] += 1
            self.plan["summary_statistics"]["property_source_breakdown"]["Materials_Project"] += 1
            return True
        
        # Try ASE cache
        ase_result = await self.call_tool(
            "ase_query",
            {"db_path": self.ase_db, "formula": candidate["formula"]}
        )
        
        if ase_result.get("count", 0) > 0:
            entry = ase_result["entries"][0]
            candidate["properties"]["formation_energy_per_atom"] = entry.get("energy")
            candidate["properties"]["band_gap"] = entry.get("band_gap")
            candidate["properties"]["property_sources"] = {
                "formation_energy_per_atom": "ASE_cached",
                "band_gap": "ASE_cached"
            }
            candidate["status"] = "properties_complete"
            self.plan["summary_statistics"]["properties_complete"] += 1
            self.plan["summary_statistics"]["property_source_breakdown"]["ASE_cached"] += 1
            return True
        
        # ML calculation (last resort)
        try:
            # Relax structure
            relax_result = await self.call_tool(
                "matgl_relax_structure",
                {
                    "input_structure": candidate["input_structure"]["cif"],
                    "fmax": self.plan["metadata"]["ml_settings"]["relaxation_fmax"],
                    "max_steps": self.plan["metadata"]["ml_settings"]["relaxation_max_steps"]
                }
            )
            
            if not relax_result["converged"]:
                raise Exception("Structure relaxation failed to converge")
            
            relaxed = relax_result["final_structure"]
            
            # Formation energy prediction
            eform = await self.call_tool(
                "matgl_predict_eform",
                {"input_structure": relaxed}
            )
            
            # Band gap prediction
            bandgap = await self.call_tool(
                "matgl_predict_bandgap",
                {"input_structure": relaxed}
            )
            
            # Stability analysis
            stability = await self.call_tool(
                "stability_analyzer",
                {"input_structure": relaxed, "hull_tolerance": 0.1}
            )
            
            candidate["properties"]["formation_energy_per_atom"] = eform["formation_energy_eV_per_atom"]
            candidate["properties"]["band_gap"] = bandgap["band_gap_eV"]
            candidate["properties"]["energy_above_hull"] = stability.get("energy_above_hull")
            candidate["properties"]["is_stable"] = stability["stability_category"] == "stable"
            candidate["properties"]["property_sources"] = {
                "formation_energy_per_atom": "ML_calculated",
                "band_gap": "ML_calculated"
            }
            candidate["screening_result"]["requires_dft"] = True
            candidate["status"] = "properties_complete"
            
            # Cache in ASE
            await self.call_tool(
                "ase_store_result",
                {
                    "db_path": self.ase_db,
                    "atoms_dict": relaxed,
                    "key_value_pairs": {"source": "ML_calculated", "candidate_id": candidate["id"]}
                }
            )
            
            self.plan["summary_statistics"]["properties_complete"] += 1
            self.plan["summary_statistics"]["property_source_breakdown"]["ML_calculated"] += 1
            return True
            
        except Exception as e:
            candidate["status"] = "properties_failed"
            candidate["screening_result"]["rejection_reason"] = f"Property retrieval failed: {str(e)}"
            self.plan["summary_statistics"]["properties_failed"] += 1
            return False
    
    async def _apply_screening_criteria(self, candidate: dict):
        """Phase 3: Apply screening criteria."""
        failed_criteria = []
        criteria = self.plan["metadata"]["screening_criteria"]
        
        # Essential criteria
        if candidate["properties"]["energy_above_hull"] is None:
            failed_criteria.append("energy_above_hull_missing")
        elif candidate["properties"]["energy_above_hull"] > criteria["essential"]["energy_above_hull_max"]:
            failed_criteria.append(f"energy_above_hull_too_high")
        
        # Application-specific criteria
        bg = candidate["properties"]["band_gap"]
        bg_min = criteria["application_specific"]["band_gap_min"]
        bg_max = criteria["application_specific"]["band_gap_max"]
        
        if bg is None:
            failed_criteria.append("band_gap_missing")
        elif bg < bg_min:
            failed_criteria.append(f"band_gap_too_low ({bg:.2f} < {bg_min})")
        elif bg > bg_max:
            failed_criteria.append(f"band_gap_too_high ({bg:.2f} > {bg_max})")
        
        # Record result
        if failed_criteria:
            candidate["screening_result"]["passed"] = False
            candidate["screening_result"]["failed_criteria"] = failed_criteria
            candidate["screening_result"]["rejection_reason"] = "; ".join(failed_criteria)
            candidate["status"] = "rejected"
            self.plan["summary_statistics"]["screening_failed"] += 1
        else:
            candidate["screening_result"]["passed"] = True
            candidate["status"] = "screening_complete"
            self.plan["summary_statistics"]["screening_passed"] += 1
    
    def _store_properties(self, candidate: dict, props: dict, source: str):
        """Helper to store properties with source attribution."""
        candidate["properties"]["formation_energy_per_atom"] = props.get("formation_energy_per_atom")
        candidate["properties"]["band_gap"] = props.get("band_gap")
        candidate["properties"]["energy_above_hull"] = props.get("energy_above_hull")
        candidate["properties"]["is_stable"] = props.get("is_stable")
        candidate["properties"]["property_sources"] = {
            "formation_energy_per_atom": source,
            "band_gap": source,
            "energy_above_hull": source
        }
        candidate["properties"]["timestamp"] = datetime.now().isoformat()


async def main():
    plan_file = Path("screening_ln_niobates_100.json")
    
    # Load and start screening
    client = CandidateScreeningClient(
        plan_file=plan_file,
        server_command="python",
        server_args=[Path("MatClaw/mcp/server.py").absolute()],
        checkpoint_frequency=1  # Save after every candidate
    )
    
    client.load_plan()
    candidates = client.plan["candidates"]
    
    async with client:
        # Process all candidates through phases 1-3
        summary = await client.batch_process(
            items=candidates,
            checkpoint_file=plan_file  # Automatic checkpointing
        )
        
        # Phase 4: Ranking (only if candidates passed screening)
        passed = [c for c in candidates if c["screening_result"].get("passed")]
        
        if passed:
            ranking = await client.call_tool(
                "multi_objective_ranker",
                {
                    "candidates": [
                        {
                            "id": c["id"],
                            "formation_energy_per_atom": c["properties"]["formation_energy_per_atom"],
                            "band_gap": c["properties"]["band_gap"],
                            "energy_above_hull": c["properties"]["energy_above_hull"]
                        }
                        for c in passed
                    ],
                    "objectives": {
                        "formation_energy_per_atom": {"weight": 0.3, "direction": "minimize"},
                        "energy_above_hull": {"weight": 0.4, "direction": "minimize"},
                        "band_gap_deviation": {"weight": 0.3, "direction": "minimize", "target": 4.0}
                    }
                }
            )
            
            for ranked in ranking["ranked_candidates"]:
                candidate = next(c for c in candidates if c["id"] == ranked["id"])
                candidate["ranking"]["rank"] = ranked["rank"]
                candidate["ranking"]["composite_score"] = ranked["composite_score"]
                candidate["status"] = "ranked"
            
            client.plan["summary_statistics"]["ranked"] = len(passed)
            client.save_plan()
    
    print(f"\n{'='*60}")
    print("Screening Complete!")
    print(f"{'='*60}")
    print(f"Validated:        {client.plan['summary_statistics']['validation_passed']}")
    print(f"Properties:       {client.plan['summary_statistics']['properties_complete']}")
    print(f"  - MP:           {client.plan['summary_statistics']['property_source_breakdown']['Materials_Project']}")
    print(f"  - ASE:          {client.plan['summary_statistics']['property_source_breakdown']['ASE_cached']}")
    print(f"  - ML:           {client.plan['summary_statistics']['property_source_breakdown']['ML_calculated']}")
    print(f"Passed:           {client.plan['summary_statistics']['screening_passed']}")
    print(f"Ranked:           {client.plan['summary_statistics']['ranked']}")


if __name__ == "__main__":
    asyncio.run(main())
```

**Why use batch_client?**
- ✅ Automatic checkpointing after every candidate
- ✅ Resume from last checkpoint on interruption
- ✅ Multi-phase workflow (validation → properties → screening) in single item processor
- ✅ Error handling without stopping batch
- ✅ Progress tracking and logging
- ✅ Property source tracking (MP/ASE/ML)
- ✅ Production-ready MCP client (no server source access)

**Critical checkpointing features:**
- Batch client saves plan after EVERY candidate automatically via `checkpoint_file` parameter
- Candidates with `status != "not_started"` are processed based on current phase
- Errors captured per-candidate without stopping batch
- All failures logged with timestamps in plan JSON
- ASE database queries prevent duplicate calculations

---

### Step 4: Handling Interruptions and Resume

**If screening is interrupted:**

1. **Check tracking file status:**
   ```json
   "metadata": {"status": "in_progress"}
   "summary_statistics": {
     "validation_passed": 85,
     "properties_complete": 42,
     "screening_passed": 38,
     "ranked": 0
   }
   ```
   
   Status shows 85 validated, 42 with properties, 38 passed screening → resume at property retrieval for remaining candidates

2. **Resume from checkpoint:**
   ```python
   plan = load_json("screening_ln_niobates_100.json")
   
   # Phase 1: Validation - resume for candidates with status "not_started"
   # Phase 2: Property retrieval - resume for candidates with status "validated" (not "properties_complete")
   # Phase 3: Screening - resume for candidates with status "properties_complete" (not "screening_complete")
   # Phase 4: Ranking - run if any candidates have status "screening_complete"
   ```

3. **Verify ASE database consistency:**
   ```python
   # Cross-check tracking file vs ASE database
   tracked_complete = [c for c in plan["candidates"] if c["status"] == "properties_complete"]
   
   ase_entries = ase_query(
       db_path=plan["metadata"]["ase_database"],
       property_filters={"source": {"$exists": True}}
   )
   
   if len(tracked_complete) != ase_entries["count"]:
       print("WARNING: Tracking file and ASE database out of sync!")
       # Reconcile by querying ASE for each candidate
       for candidate in plan["candidates"]:
           if candidate["status"] == "retrieving_properties":
               ase_result = ase_query(db_path=ase_db, formula=candidate["formula"])
               if ase_result["count"] > 0:
                   # Found in ASE - mark as complete
                   candidate["status"] = "properties_complete"
   ```

---

### Step 5: Iterative Criteria Refinement

**Common workflow:**
1. Run screening with initial criteria (e.g., band gap 3.0-5.0 eV)
2. Review partial results after 50% completion
3. Adjust criteria if too strict/loose (e.g., relax to 2.5-5.5 eV)
4. Rerun screening phase (properties already cached - fast)

**Implementation:**
```python
# Modify criteria in tracking file
plan["metadata"]["screening_criteria"]["application_specific"]["band_gap_min"] = 2.5
plan["metadata"]["screening_criteria"]["application_specific"]["band_gap_max"] = 5.5

# Reset screening results
for candidate in plan["candidates"]:
    if candidate["status"] in ["screening_complete", "rejected"] and candidate["properties"]["formation_energy_per_atom"] is not None:
        candidate["status"] = "properties_complete"  # Revert to pre-screening state
        candidate["screening_result"] = { 
            "passed": null,
            "failed_criteria": [],
            "rejection_reason": null,
            "requires_dft": candidate["screening_result"].get("requires_dft", False)
        }

# Rerun Phase 3 and 4 (validation and property retrieval already done)
run_screening_phase(plan, phase=3)  # Screening
run_screening_phase(plan, phase=4)  # Ranking
```

---

## Best Practices for Screening Tracking

1. **Checkpoint aggressively:**
   - Save after every candidate (not batches) - property retrieval can timeout
   - Enables fine-grained resume

2. **Track property sources explicitly:**
   - Essential for confidence assessment (MP > ASE > ML)
   - Flags ML predictions for DFT verification

3. **Log all errors with context:**
   - Timestamp, candidate ID, phase, error message
   - Enables debugging property retrieval failures

4. **Estimate property sources realistically:**
   - Novel compositions → mostly ML calculations
   - Common materials → mostly MP lookups
   - Sets user expectations for runtime

5. **Enable iterative refinement:**
   - Preserve all properties even if screening fails
   - User can adjust criteria without rerunning expensive ML calculations

6. **Cross-reference ASE database:**
   - Verify tracking file matches cached results
   - Detect orphaned entries or missing checkpoints

7. **Mark ML predictions for verification:**
   - Set `requires_dft=True` for top-ranked candidates from ML
   - Prioritize DFT calculations for experimental validation
