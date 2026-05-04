# Large-Scale Generation Planning

Planning workflow for generating >20 structures with checkpointing and progress tracking.

## When to Use Planning Workflow

### Trigger Conditions

Use planning workflow when:
- User requests N > 20 structures explicitly ("generate 100 candidates")
- User requests "all possible" or "comprehensive" screening
- Multi-batch generation involving different tools/templates
- Generation across multiple chemical systems

### Skip Planning For

- Quick explorations (<10 structures)
- Single-batch generation with one tool call
- User explicitly requests immediate generation

---

## Planning Workflow Steps

### Step 1: Create Generation Plan

Generate structured JSON that specifies:
- Total target count
- Batch organization (by tool, template, composition)
- Per-candidate metadata (formula, tool, parameters)
- Status tracking fields

**Plan template structure:**

```json
{
  "metadata": {
    "request_summary": "100 lanthanide-doped niobate phosphors for PC-LED",
    "total_planned": 100,
    "status": "planning",
    "created": "2026-04-28",
    "output_directory": "candidate-generation"
  },
  "batches": [
    {
      "batch_id": "batch_1",
      "description": "5% single Ln doping in SrNb2O6",
      "tool": "pymatgen_disorder_generator",
      "base_structure": {
        "mp_id": "mp-4591",
        "formula": "SrNb2O6"
      },
      "target_count": 14,
      "status": "not_started",
      "completed_count": 0,
      "candidates": [
        {
          "id": "CAND-001",
          "formula": "Sr0.95Eu0.05Nb2O6",
          "status": "not_started",
          "cif_file": null,
          "tool_parameters": {
            "site_substitutions": {"Sr": {"Eu": 0.05, "Sr": 0.95}}
          },
          "notes": "5% Eu3+ doping on Sr2+ site"
        }
      ]
    }
  ]
}
```

**Key fields:**
- `status`: `"planning"` → `"in_progress"` → `"completed"` (or `"paused"`)
- `batch_id`: Organize by tool, template, or scientific category
- `candidate.status`: `"not_started"` → `"completed"` → `"failed"`
- `cif_file`: Path to generated CIF file (set by script during execution)
- `tool_parameters`: Exact parameters for reproducibility
- `execution_log`: Timestamped events

---

### Step 2: Present Plan to User

**ALWAYS show:**
1. Total candidate count and batch breakdown
2. Scientific rationale for each batch
3. Which MCP tools will be used
4. Estimated resource requirements

**Example presentation:**

```
Generated planning file: generation_candidates.json

Plan Summary:
─────────────────────────────────────────────────
Total candidates: 100
Output directory: candidate-generation/
Batches: 7

Batch breakdown:
  1. Single Ln doping (SrNb₂O₆):    20 structures [disorder_generator]
  2. Single Ln doping (BaNb₂O₆):    15 structures [disorder_generator]
  3. Double perovskites (A₂LnNbO₆): 20 structures [substitution_generator]
  4. Co-doping (Ln₁+Ln₂):           15 structures [disorder_generator]
  5. Varied doping levels:          10 structures [disorder_generator]
  6. ZnNb₂O₆ host:                  10 structures [disorder_generator]
  7. Alternative hosts:             10 structures [disorder_generator]

Estimated: ~110 MCP tool calls, 15-30 minutes runtime

To execute:
  python batch_generation.py --plan generation_candidates.json

Review generation_candidates.json and confirm to proceed.
```

**Wait for user approval** before proceeding. User may:
- Approve as-is
- Request modifications (change doping, skip batches, add compositions)
- Abort if plan doesn't match intent

---

### Step 3: Execute with Checkpointing

**Production Implementation: Self-Contained Python Script**

For production execution, create a project-specific script based on the reference template.

**Reference Location:** `MatClaw/skills/candidate-generator/examples/batch_generation_example.py`

The reference script demonstrates how to use the **MCP client SDK** to call tools via `session.call_tool()`:

**Key Improvements in Updated Template:**

1. **✅ Dynamic Tool Selection** - No hardcoded tool names
2. **✅ Flexible Base Structure Resolution** - 5 fallback options
3. **✅ Generic Parameter Handling** - Works with any tool
4. **✅ Comprehensive Customization Guide** - Clear instructions for adaptation

### Dynamic Tool Selection

The template now reads tool names from the plan rather than hardcoding:

```python
async def generate_structure(self, session, candidate: dict, base_cif: str, batch: dict):
    """Generate structure using ANY MCP tool (not hardcoded)."""
    
    # Read tool name from candidate or batch
    tool_name = candidate.get('tool') or batch.get('tool')
    tool_params = candidate.get('tool_parameters', {}).copy()
    
    # Auto-inject base structure if not in params
    if 'input_structures' not in tool_params and base_cif:
        tool_params['input_structures'] = [base_cif]
    
    # Call tool dynamically (works for any tool)
    result = await session.call_tool(tool_name, tool_params)
```

**Supports:** `disorder_generator`, `substitution_generator`, `enumeration_generator`, `defect_generator`, etc.

### Flexible Base Structure Resolution

The template tries 5 sources in priority order:

```python
# Priority 1: Direct MP ID in tool_parameters
if 'base_mp_id' in candidate['tool_parameters']:
    base_cif = await self.get_base_structure(session, candidate['tool_parameters']['base_mp_id'])

# Priority 2: Direct MP ID at candidate level  
elif 'base_mp_id' in candidate:
    base_cif = await self.get_base_structure(session, candidate['base_mp_id'])

# Priority 3: Formula in candidate (searches MP)
elif 'base_composition' in candidate:
    base_id = await self.find_material_id(session, candidate['base_composition'])
    base_cif = await self.get_base_structure(session, base_id)

# Priority 4 & 5: Batch-level defaults (mp_id or formula)
elif 'base_structure_query' in batch:
    ...
```

**Extensible:** Easy to add support for local CIF files, prototype generation, etc.

### Plan Schema Flexibility

The template now supports multiple plan structures:

**Option A: Tool at batch level**
```json
{
  "batches": [
    {
      "batch_id": "batch_1",
      "tool": "pymatgen_disorder_generator",
      "base_structure_query": {"mp_id": "mp-4591"},
      "candidates": [
        {"id": "CAND-001", "tool_parameters": {...}}
      ]
    }
  ]
}
```

**Option B: Tool per candidate (mixed tools)**
```json
{
  "batches": [
    {
      "candidates": [
        {
          "id": "CAND-001",
          "tool": "pymatgen_disorder_generator",
          "base_mp_id": "mp-4591",
          "tool_parameters": {"site_substitutions": {...}}
        },
        {
          "id": "CAND-002",
          "tool": "pymatgen_substitution_generator",
          "base_composition": "SrMoO4",
          "tool_parameters": {"substitutions": {...}, "n_structures": 5}
        }
      ]
    }
  ]
}
```

### Core Components (Simplified Reference)

```python
class BatchGenerator:
    """Reference implementation with flexible tool/structure handling."""
    
    def __init__(self, plan_file: Path, output_dir: Path):
        self.plan_file = plan_file
        self.output_dir = output_dir
        self.base_cache = {}  # MP structure cache
        self.formula_to_id_cache = {}  # Formula → MP ID cache
    
    async def get_base_structure(self, session, material_id: str) -> str:
        """Fetch base from Materials Project via MCP (with caching)."""
        if material_id in self.base_cache:
            return self.base_cache[material_id]
        
        result = await session.call_tool(
            "mp_get_material_properties",
            {"material_ids": [material_id]}
        )
        cif_string = parse_tool_result(result)['properties'][0]['structure']['cif']
        self.base_cache[material_id] = cif_string
        return cif_string
    
    async def find_material_id(self, session, formula: str) -> str:
        """Search Materials Project for formula (with caching)."""
        if formula in self.formula_to_id_cache:
            return self.formula_to_id_cache[formula]
        
        result = await session.call_tool(
            "mp_search_materials",
            {"formula": formula, "num_results": 5}
        )
        materials = parse_tool_result(result)['materials']
        
        # Prefer stable materials
        stable = [m for m in materials if m.get('energy_above_hull') == 0]
        material_id = (stable[0] if stable else materials[0])['material_id']
        
        self.formula_to_id_cache[formula] = material_id
        return material_id
    
    async def generate_structure(self, session, candidate, base_cif, batch):
        """Call MCP tool dynamically (any tool, not hardcoded)."""
        tool_name = candidate.get('tool') or batch.get('tool')
        tool_params = candidate.get('tool_parameters', {}).copy()
        
        # Auto-inject base if not provided
        if 'input_structures' not in tool_params:
            tool_params['input_structures'] = [base_cif]
        if 'output_format' not in tool_params:
            tool_params['output_format'] = 'cif'
        
        result = await session.call_tool(tool_name, tool_params)
        return parse_tool_result(result)['structures'][0]
    
    async def process_candidate(self, session, candidate, batch):
        """Process with flexible base resolution (5 fallback options)."""
        # ... [flexible base resolution code - see template] ...
        base_cif = await self.resolve_base_structure(session, candidate, batch)
        
        # Generate with dynamic tool
        cif_content = await self.generate_structure(session, candidate, base_cif, batch)
        
        # Save and checkpoint
        cif_path = self.save_structure(candidate['id'], candidate['formula'], cif_content)
        candidate['status'] = 'completed'
        candidate['output_file'] = str(cif_path)
        self.save_plan()
```

**Full implementation:** See `examples/batch_generation_example.py` for complete code with:
- Connection management (stdio/SSE)
- Error handling & per-candidate error capture
- Progress tracking & real-time logging
- Automatic checkpoint/resume
- MP structure caching

---

## Step 4: Checkpointing and Resume

### Automatic Checkpointing

The script updates the plan file after EVERY candidate (atomic writes):

```python
async def process_candidate(self, session, candidate, batch):
    try:
        # ... generate structure ...
        
        # Save and update status
        candidate['status'] = 'completed'
        candidate['output_file'] = str(cif_path)
        self.save_plan()  # ← Automatic checkpoint
        
    except Exception as e:
        candidate['status'] = 'failed'
        candidate['error'] = str(e)
        self.save_plan()  # ← Checkpoint even on failure
```

**Checkpoint behavior:**
- ✅ Saves after every candidate (success or failure)
- ✅ Uses atomic writes (temp file → rename)
- ✅ On resume: skips candidates with `status == "completed"`
- ✅ Errors isolated per-candidate (batch continues)

### Resume from Interruption

If interrupted (Ctrl+C, crash, network loss), simply re-run the script:

```bash
python run_generation.py  # Automatically resumes where it left off
```

The script detects completed candidates and skips them:

```python
async def process_candidate(self, session, candidate, batch):
    if candidate.get('status') == 'completed':
        logger.info(f"{candidate['id']}: Already completed, skipping")
        return True
    # ... proceed with generation ...
```

**Progress tracking in plan file:**
```json
{
  "metadata": {
    "total_planned": 100,
    "last_updated": "2026-04-30T14:23:45"
  },
  "batches": [
    {
      "candidates": [
        {
          "id": "CAND-001",
          "status": "completed",
          "completed_at": "2026-04-30T14:15:23",
          "output_file": "candidates/CAND-001_Sr0.95Sm0.05MoO4.cif"
        },
        {
          "id": "CAND-002",
          "status": "failed",
          "failed_at": "2026-04-30T14:16:12",
          "error": "Could not find material ID for formula XYZ"
        },
        {
          "id": "CAND-003",
          "status": "pending"  # ← Will be processed on resume
        }
      ]
    }
  ]
}
```

---

## Step 4: Export Final Results

The batch script automatically saves structures as CIF files during execution:

```python
# During execution (automatic in project-specific script):
# Each candidate saved as: {candidate_id}_{formula}.cif
# Example: CAND-001_Sr0.95Eu0.05Nb2O6.cif
```

**Result structure:**
```
candidate-generation/
  ├── CAND-001_Sr0.95Eu0.05Nb2O6.cif
  ├── CAND-002_Sr0.95Sm0.05Nb2O6.cif
  ├── ...
  └── CAND-100_Ba0.9Eu0.05Sm0.05Nb2O6.cif
generation_candidates.json  # Contains status + paths to CIF files
```

**Optional: Generate summary JSON**

If you need a single JSON file with all results:

```python
import json
from pathlib import Path

def export_results(plan_file: Path, output_file: Path):
    """Export completed candidates with file paths."""
    with open(plan_file) as f:
        plan = json.load(f)
    
    candidates_output = []
    
    for batch in plan["batches"]:
        for candidate in batch["candidates"]:
            if candidate["status"] != "completed":
                continue
            
            candidates_output.append({
                "id": candidate["id"],
                "formula": candidate["formula"],
                "cif_file": candidate.get("cif_file"),
                "batch_id": batch["batch_id"],
                "base_template": batch["base_structure"]["mp_id"],
                "tool": batch["tool"]
            })
    
    output = {
        "metadata": {
            "total_candidates": len(candidates_output),
            "plan_file": str(plan_file)
        },
        "generated_candidates": candidates_output
    }
    
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

# Usage
export_results(
    plan_file=Path("generation_candidates.json"),
    output_file=Path("results_summary.json")
)
```

---

## Handling Interruptions

### Resume from Checkpoint

When execution is interrupted:

1. **Check plan status:**
```json
"metadata": {"status": "in_progress"}
"batches[0]": {"status": "completed", "completed_count": 20}
"batches[1]": {"status": "in_progress", "completed_count": 8}
```

2. **Resume from last checkpoint:**
- Batch 0 complete → skip
- Batch 1 has 8/15 done → resume at candidate 9
- `candidate["status"] == "completed"` → skip in loop

3. **Verify file consistency:**
```python
# Cross-check plan vs generated CIF files
from pathlib import Path

for batch in plan["batches"]:
    for candidate in batch["candidates"]:
        if candidate["status"] == "completed":
            # Verify CIF file exists
            cif_path = Path(candidate.get("cif_file", ""))
            if not cif_path.exists():
                print(f"WARNING: {candidate['id']} marked complete but CIF not found!")
                candidate["status"] = "not_started"
                candidate["cif_file"] = None
        
        elif candidate["status"] == "not_started" and candidate.get("cif_file"):
            # CIF exists but not marked complete
            cif_path = Path(candidate["cif_file"])
            if cif_path.exists():
                print(f"INFO: {candidate['id']} has CIF, marking as completed")
                candidate["status"] = "completed"
```

---

## Best Practices

### 1. Organize Batches Scientifically

**Good organization:**
- "Single Ln doping in Sr host"
- "Double perovskites A₂LnNbO₆"
- "Co-doping pairs for energy transfer"

**Bad organization:**
- "disorder_generator batch 1"
- "disorder_generator batch 2"
- "batch_3"

### 2. Include Scientific Rationale

Add `description` and `notes` explaining:
- Chemical logic
- Scientific hypothesis
- Expected properties

**Example:**
```json
{
  "description": "Co-doped SrNb₂O₆ with Sm+Eu pairs",
  "notes": "Sm→Eu energy transfer for white light emission"
}
```

### 3. Validate Before Execution

Check:
- Charge neutrality for ionic materials
- MP templates exist
- Site labels match structure
- No duplicate formulas

### 4. Size Batches Reasonably

**Good batch sizes:** 10-20 candidates
- Easy to review progress
- Logical checkpoints
- Clear error isolation

**Avoid:** >50 candidates per batch (hides issues)

### 5. Cross-Check Existing Structures

Before creating plan:
```python
existing = ase_query_db(property_filters={"compound": target_formula})
if existing["count"] > 0:
    print(f"WARNING: {target_formula} already exists in database")
```

### 6. Estimate Resources Honestly

**Timing estimates:**
- 1-2 seconds per `disorder_generator` call
- 2-5 seconds per `substitution_generator` call
- Add 20% buffer for API latency

**Example:**
```
90 disorder_generator calls × 1.5 sec = 135 seconds
25 substitution_generator calls × 3 sec = 75 seconds
Total: 210 seconds + 20% buffer = 252 seconds (~4 minutes)
```

---

## Example: 100 Lanthanide-Doped Niobates

**User request:**
> "Generate 100 lanthanide-doped niobate phosphor candidates based on MNb₂O₆ structure"

**Step 1: Create plan**

```python
plan = {
    "metadata": {
        "request_summary": "100 lanthanide-doped niobate phosphors for PC-LED",
        "total_planned": 100,
        "status": "planning",
        "output_directory": "./lanthanide_niobate_cifs"
    },
    "batches": [
        {
            "batch_id": "batch_1_sr_single_ln",
            "description": "5% single Ln³⁺ doping in SrNb₂O₆",
            "base_structure": {"mp_id": "mp-4591"},
            "lanthanides": ["La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", 
                           "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu"],
            "target_count": 14
        },
        {
            "batch_id": "batch_2_ba_single_ln",
            "description": "5% single Ln³⁺ doping in BaNb₂O₆",
            "base_structure": {"mp_id": "mp-28150"},
            "target_count": 14
        },
        {
            "batch_id": "batch_3_double_perovskite",
            "description": "Double perovskite Ba₂LnNbO₆",
            "tool": "pymatgen_substitution_generator",
            "base_structure": {"mp_id": "mp-553281", "formula": "Ba2LaNbO6"},
            "target_count": 10
        }
        # ... more batches to reach 100
    ]
}
```

**Step 2: Present to user**

```
Plan Summary:
─────────────────────────────────────────────────
Total candidates: 100 structures
Batches: 7

Scientific strategy:
  1. Single Ln doping (Sr, Ba, Ca hosts): 40 structures
     → Systematic screening of all Ln³⁺ dopants
  
  2. Double perovskites (A₂LnNbO₆): 20 structures
     → Ln on ordered B-site for higher concentrations
  
  3. Co-doping (energy transfer pairs): 15 structures
     → Sm+Eu, Nd+Yb pairs for tunable emission

Estimated: ~115 MCP tool calls, 20-35 minutes

Review and confirm to proceed.
```

**Step 3: Execute with checkpointing**

(See pseudocode in Step 3 above)

**Step 4: Export results**

```json
{
  "metadata": {
    "total_candidates": 100,
    "output_directory": "./lanthanide_niobate_cifs"
  },
  "generated_candidates": [
    {
      "id": "LNP-001",
      "formula": "Sr0.95La0.05Nb2O6",
      "structure": {
        "cif": "...",
        "natoms": 45
      }
    }
    // ... 99 more
  ]
}
```

---

## Troubleshooting

### Plan and Output Files Out of Sync

**Symptom:** `completed_count` doesn't match actual CIF files on disk

**Cause:** Checkpoint saved before file write, or file write failed

**Solution:**
```python
from pathlib import Path

# Reconcile from output directory
output_dir = Path(plan["output_directory"])

for batch in plan["batches"]:
    for candidate in batch["candidates"]:
        # Check if CIF file exists
        if "cif_file" in candidate:
            cif_path = Path(candidate["cif_file"])
        else:
            cif_path = output_dir / f"{candidate['id']}.cif"
        
        if cif_path.exists() and candidate["status"] != "completed":
            # Found file but not marked complete
            candidate["status"] = "completed"
            candidate["cif_file"] = str(cif_path)
        
        elif not cif_path.exists() and candidate["status"] == "completed":
            # Marked complete but file missing
            candidate["status"] = "not_started"
            candidate.pop("cif_file", None)

save_json(plan, "generation_candidates.json")
```

### Batch Stuck in Progress

**Symptom:** Batch shows "in_progress" but all candidates complete

**Cause:** Final batch status update didn't save

**Solution:**
```python
# Recount completed candidates
for batch in plan["batches"]:
    completed = sum(1 for c in batch["candidates"] if c["status"] == "completed")
    batch["completed_count"] = completed
    
    if completed == batch["target_count"]:
        batch["status"] = "completed"

save_json(plan, "generation_candidates.json")
```

### High Failure Rate

**Symptom:** Many candidates have `status: "failed"`

**Cause:** Systematic issue with tool parameters or input structures

**Solution:**
1. Check first failure in `execution_log`
2. Fix root cause (bad MP ID, invalid parameters)
3. Reset failed candidates: `status: "not_started"`
4. Re-run with corrected parameters

### Output Format Mismatch Between Plan and Script

**Symptom:** 
- Batch script fails with `TypeError: Expected CIF string, got <class 'dict'>`
- Or: Script fails when trying to write structure files with string operations on dict objects
- Affects all candidates systematically

**Cause:** 
The `generation_plan.json` file may specify `output_format: 'ase'` in `tool_parameters`, which causes generation tools to return Python dictionaries instead of CIF strings. This happens when:

1. The plan was created for ASE database storage (requires dict format)
2. The plan was copied from a template with different output requirements
3. The batch script assumes CIF strings for file writing operations

Different output formats return different types:
- `'cif'` or `'poscar'` → String (can write directly to file)
- `'ase'` → Dictionary `{numbers, positions, cell, pbc}` (requires conversion)
- `'json'` → JSON-encoded string with nested structure

**Solution:**

**Option 1 (Recommended):** Force output format in batch script

```python
# In batch generation script, after loading tool parameters:
tool_params = candidate.get("tool_parameters", {}).copy()

# Force CIF format regardless of what plan specifies
tool_params['output_format'] = 'cif'  # Override plan parameter

# Now call the tool with forced format
result = await session.call_tool(
    tool_name=candidate["generation_tool"],
    arguments=tool_params
)
```

This approach:
- ✅ Allows plan reuse for different purposes (scripts vs databases)
- ✅ Makes script requirements explicit
- ✅ Prevents type mismatch errors
- ✅ No need to manually edit large planning files

**Option 2:** Update planning file

```json
{
  "tool_parameters": {
    "base_mpid": "mp-18834",
    "substitutions": {"Sr": "Sm"},
    "concentrations": [0.03],
    "output_format": "cif"  // Change from 'ase' to 'cif'
  }
}
```

Only use this if you're certain all consumers of the plan need the same format.

**Prevention:**
- In batch scripts that write structure files, always force `output_format='cif'` 
- Document the expected format in script comments
- For plans used with multiple scripts, specify format per-use case rather than in plan
- See `references/gotchas.md` section "Batch Generation and Large-Scale Issues" for detailed examples
