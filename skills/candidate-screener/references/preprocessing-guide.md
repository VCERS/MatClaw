# Structure Ordering Preprocessing Guide

Guide for handling disordered structures before screening.

---

## Why Preprocessing Matters

### ASE Compatibility

MatGL and matcalc use ASE (Atomic Simulation Environment) as their structure representation backend. ASE **cannot represent partial site occupancies** — every atomic site must have occupancy = 1.0.

When pymatgen structures with fractional occupancy (e.g., Sr₀.₉₇Sm₀.₀₃Nb₂O₆) are passed to MatGL/matcalc tools, you'll see:

```
ERROR: Structure has partial site occupancies (disordered).
MatGL and ASE require fully ordered structures.
```

This means the structure needs to be ordered before these tools can work with it. The choice of which ordering method to use depends on the doping level and the research question.

### When Disordered Structures Appear

Disordered structures in screening workflows come from three sources:

1. **candidate-generator outputs** (most common, ~90% of cases)
   - `pymatgen_disorder_generator`: Explicit fractional substitution for dilute doping
   - `pymatgen_ion_exchange_generator`: Charge-balancing can create disorder
   - Should include `metadata.requires_ordering` flag

2. **Materials Project database** (rare, ~5% of cases)
   - Mineral structures with crystallographic disorder
   - Mixed-valence compounds
   - Metadata usually unavailable → use heuristic

3. **User-provided CIF files** (manual structures, ~5% of cases)
   - No metadata → must use doping concentration heuristic

---

## Choosing an ordering strategy

There's no single right answer — the best strategy depends on the research question.

```
IF structure.is_ordered → Skip preprocessing (no action needed)

ELIF metadata.requires_ordering is specified → Honor that choice

ELSE → Choose based on doping level and research goals:

    LOW DOPING (<5%):
    ─────────────────
    majority_orderer is a natural fit — the host lattice dominates,
    so keeping the majority species per site preserves the physics.
    
    HIGHER DOPING (≥5%):
    ────────────────────
    You have a genuine choice. Consider:
    
    - enumeration_orderer: Explores all symmetry-distinct arrangements
      of the dopant atoms, producing 10-50 structures ranked by Ewald
      energy. Use when site-specific ordering matters, or when you need
      the most stable ordering.
    
    - sqs_orderer: Creates one or a few supercells whose correlation
      functions approximate random disorder. More accurate for
      concentrated solid solutions. Use when you need random disorder
      representation.
    
    If you're unsure which is appropriate, consult the user.
```

**Key insight:** For low doping (<5%), majority ordering is a good approximation because dopant-dopant interactions are negligible. As doping increases, those interactions matter more, and you should decide with the user whether exhaustive enumeration or a representative SQS supercell better serves their question.

---

## Three Ordering Strategies

### Strategy 1: Majority-Species Orderer

**Tool:** `pymatgen_majority_orderer`

**Physical basis:** In the dilute limit (roughly <5% dopant), the host lattice structure and properties dominate. Dopants provide minor perturbations but don't fundamentally alter bonding or electronic structure. The majority-species approximation (keeping only the dominant species per site) is physically justified.

**When to use:**
- Low doping (roughly <5% dopant concentration)
- Fast high-throughput screening
- Dopant species has negligible structural impact
- `metadata.requires_ordering == "majority"`

**Algorithm:**
```
FOR each site in structure:
    IF site has multiple species:
        species_with_max_occupancy = max(site.species.items(), key=lambda x: x[1])
        ordered_site = create_site(species=species_with_max_occupancy[0])
        replace site with ordered_site
```

**Returns:** 1 ordered structure per input (no supercell expansion)

**Example:**
```python
# Input: Sr₀.₉₇Sm₀.₀₃Nb₂O₆ (disordered, 3% doping)
result = pymatgen_majority_orderer(
    input_structures=disordered_cif,
    check_ordered_input=True,  # Skip if already ordered
    output_format='cif'
)

# Output: SrNb₂O₆ (ordered)
# Metadata: {'was_disordered': True, 'lost_species': ['Sm'], ...}
```

**Physical guidance on validity:**
- **Well-suited (<5% doping):** Host lattice dominates, minority species produces minor perturbation
- **Potentially useful (5-20%):** Can still serve for initial screening, but consider discussing with user whether enumeration or SQS would better capture dopant interactions
- **Not recommended (>20%):** Dopant-dopant interactions are significant — enumeration or SQS is more appropriate

**Computational cost:** ~0.1s per structure (instant)

**Metadata output:**
```json
{
    "was_disordered": true,
    "sites_converted": 4,
    "lost_species": ["Sm"],
    "majority_species_used": ["Sr", "Nb", "O"],
    "source_formula": "Sr0.97Sm0.03Nb2O6",
    "ordered_formula": "SrNb2O6",
    "occupancy_threshold": 0.5
}
```

**Optional validation flags to attach:**
```python
candidate['validation_flags'] = {
    "preprocessing_applied": "majority_orderer",
    "approximation_valid": (doping_concentration < 0.05),
    "requires_dft_validation": (doping_concentration >= 0.05),
}
```

---

### Strategy 2: Enumeration Orderer

**Tool:** `pymatgen_enumeration_orderer`

**Physical basis:** At finite concentrations, the specific spatial arrangement of dopants matters. Different orderings have different electrostatic energies (Ewald energy). This tool exhaustively enumerates all symmetry-distinct orderings and ranks by stability.

**When to use:**
- Site-specific dopant studies (which site gives best properties?)
- Exhaustive configuration exploration
- Small supercells (< 20 atoms in supercell)
- Intermediate concentrations (10-30%)
- `metadata.requires_ordering == "enumeration"`

**Algorithm:**
```
1. Create supercell (2×2×2 typical)
2. Use enumlib to find all symmetry-distinct orderings
3. For each ordering:
   - Calculate Ewald energy (electrostatic stability)
   - Calculate energy above hull (thermodynamic stability)
4. Sort by energy (lowest = most stable)
5. Return top N structures
```

**Returns:** 10-50 ordered configurations per input (supercell expanded)

**Example:**
```python
# Input: Li₀.₅Na₀.₅Cl (disordered 50/50 rocksalt)
result = pymatgen_enumeration_orderer(
    input_structures=disordered_cif,
    supercell_size=2,  # 2×2×2 = 8× unit cell
    n_structures=10,  # Return top 10 by Ewald energy
    sort_by='ewald',  # Options: 'ewald', 'symmetry'
    output_format='cif'
)

# Output: 10 ordered configurations
# Example formulas: LiNaCl₂, Li₂Cl₂, Na₂Cl₂ (different arrangements)
```

**Physical validity:**
- ✅ All ordered configurations equally valid from enumeration perspective
- Use for screening: Test all, pick configuration with best properties
- Computational cost scales with number of configurations

**Computational cost:** ~2-10s per structure (depends on supercell size)

**When to use each output:**
- Screening: Test all 10-50 configurations, pick best performer
- DFT validation: Focus on top 3-5 by Ewald energy
- Property prediction: Weighted average over all configs (Boltzmann)

**Metadata output:**
```json
{
    "supercell_size": [2, 2, 2],
    "original_atoms": 8,
    "supercell_atoms": 64,
    "ewald_energy": -123.45,
    "configuration_rank": 1,
    "symmetry_group": "Pmm2"
}
```

---

### Strategy 3: SQS Orderer

**Tool:** `pymatgen_sqs_orderer`

**Physical basis:** At high concentrations (>20%), dopant-dopant interactions and local ordering become important. True random disorder is computationally intractable. Special Quasirandom Structures (SQS) are ordered supercells designed to match the Warren-Cowley correlation functions of true random disorder while remaining small enough for DFT.

**When to use:**
- High-concentration doping (> 20%)
- Solid solutions (random alloys)
- High-entropy materials (≥4 mixing species)
- Disorder is functionally important
- `metadata.requires_ordering == "sqs"`

**Algorithm:**
```
1. Create large supercell (50-200 atoms)
2. Monte Carlo optimization:
   - Random atom swaps
   - Calculate correlation functions vs target disorder
   - Accept/reject based on error metric
   - Iterate for n_mc_steps
3. Return best structure (lowest sqs_error)
```

**Returns:** 1-5 ordered supercells per input (large supercell)

**Example:**
```python
# Input: Li[Ni₀.₆Mn₀.₂Co₀.₂]O₂ (disordered ternary cathode)
result = pymatgen_sqs_orderer(
    input_structures=disordered_cif,
    supercell_size=4,  # 4×4×4 = 64× unit cell → ~192 atoms
    n_structures=5,  # Generate 5 different SQS
    n_mc_steps=100000,  # Longer for better convergence
    output_format='cif'
)

# Output: 5 ordered 192-atom supercells
# Best structure: result['structures'][0] (lowest sqs_error)
```

**Physical validity:**
- ✅ Most accurate for concentrated disorder
- ✅ Captures short-range order effects
- ⚠️ Still an approximation (not true disorder)

**Computational cost:** 
- Generation: ~10-60s (depends on n_mc_steps, supercell size)
- Downstream calculations: 10-50× more expensive (larger supercells)

**Convergence guidelines:**
```python
# Binary (2 mixing species): 50k-100k MC steps
n_mc_steps = 50000

# Ternary (3 mixing species): 100k-200k MC steps
n_mc_steps = 100000

# Quaternary+ (4+ mixing species): 200k-500k MC steps
n_mc_steps = 500000

# Check convergence: sqs_error should be < 0.01 for good approximation
if result['metadata'][0]['sqs_error'] > 0.01:
    # Rerun with more MC steps or larger supercell
```

**Metadata output:**
```json
{
    "supercell_size": [4, 4, 4],
    "original_atoms": 3,
    "supercell_atoms": 192,
    "sqs_error": 0.0047,  # Lower is better
    "n_mc_steps": 100000,
    "species_mixing": ["Ni", "Mn", "Co"],
    "target_occupancies": {"Ni": 0.6, "Mn": 0.2, "Co": 0.2}
}
```

---

## Implementation Examples

### Minimal Preprocessing (Single Candidate)

```python
from pymatgen.core import Structure

# Load candidate
candidate_cif = load_cif_file("candidate_001.cif")
structure = Structure.from_str(candidate_cif, fmt='cif')

# Check if preprocessing needed
if not structure.is_ordered:
    # Read metadata
    metadata = candidate.get('metadata', {})
    ordering_strategy = metadata.get('requires_ordering', 'majority')
    
    # Apply appropriate orderer
    if ordering_strategy == 'majority':
        result = pymatgen_majority_orderer(
            input_structures=candidate_cif,
            output_format='cif'
        )
        ordered_structure = result['structures'][0]
    
    # Continue to validation
    validator_result = structure_validator(ordered_structure)
else:
    # Already ordered, skip preprocessing
    ordered_structure = candidate_cif
```

### Complete Preprocessing Pipeline (Batch)

```python
def preprocess_candidates(raw_candidates):
    """
    Preprocess all candidates, handling disorder according to metadata.
    
    Returns: List of preprocessed candidates (possibly expanded from enumeration)
    """
    preprocessed = []
    
    for candidate in raw_candidates:
        structure = Structure.from_str(candidate['structure'], fmt='cif')
        
        # Already ordered - skip preprocessing
        if structure.is_ordered:
            preprocessed.append(candidate)
            continue
        
        # Disordered - determine strategy
        metadata = candidate.get('metadata', {})
        strategy = metadata.get('requires_ordering', 'majority')  # Default to majority
        
        if strategy == 'majority':
            # Fast dilute doping approximation
            result = pymatgen_majority_orderer(
                input_structures=candidate['structure'],
                output_format='cif'
            )
            
            ordered_candidate = {
                **candidate,
                'structure': result['structures'][0],
                'ordering_metadata': result['metadata'][0],
                'preprocessing_applied': 'majority_orderer',
                'approximation_valid': metadata.get('doping_concentration', 0) < 0.10
            }
            preprocessed.append(ordered_candidate)
        
        elif strategy == 'enumeration':
            # Exhaustive ordering - creates MULTIPLE candidates
            result = pymatgen_enumeration_orderer(
                input_structures=candidate['structure'],
                supercell_size=2,
                n_structures=10,
                sort_by='ewald',
                output_format='cif'
            )
            
            # Create separate candidate for each configuration
            for i, ordered_structure in enumerate(result['structures']):
                ordered_candidate = {
                    **candidate,
                    'candidate_id': f"{candidate['candidate_id']}_config{i+1:02d}",
                    'structure': ordered_structure,
                    'ordering_metadata': result['metadata'][i],
                    'preprocessing_applied': 'enumeration_orderer',
                    'configuration_rank': i + 1,
                    'ewald_energy': result['metadata'][i]['ewald_energy']
                }
                preprocessed.append(ordered_candidate)
        
        elif strategy == 'sqs':
            # Statistical disorder approximation
            result = pymatgen_sqs_orderer(
                input_structures=candidate['structure'],
                supercell_size=4,
                n_structures=1,  # Usually just best SQS
                n_mc_steps=100000,
                output_format='cif'
            )
            
            ordered_candidate = {
                **candidate,
                'structure': result['structures'][0],
                'ordering_metadata': result['metadata'][0],
                'preprocessing_applied': 'sqs_orderer',
                'sqs_error': result['metadata'][0]['sqs_error'],
                'approximation_valid': True  # SQS always valid for high conc
            }
            preprocessed.append(ordered_candidate)
        
        else:
            # Unknown strategy - apply heuristic
            doping_conc = estimate_doping_concentration(structure)
            
            if doping_conc < 0.10:
                # Use majority orderer as fallback
                result = pymatgen_majority_orderer(
                    input_structures=candidate['structure'],
                    output_format='cif'
                )
                ordered_candidate = {
                    **candidate,
                    'structure': result['structures'][0],
                    'ordering_metadata': result['metadata'][0],
                    'preprocessing_applied': 'majority_orderer',
                    'warning': 'No metadata, used heuristic'
                }
                preprocessed.append(ordered_candidate)
            else:
                # Reject - requires manual SQS decision
                print(f"REJECT {candidate['candidate_id']}: High-concentration disorder without metadata")
                continue
    
    return preprocessed

def estimate_doping_concentration(structure):
    """Estimate doping concentration from site occupancies."""
    min_occupancy = 1.0
    for site in structure:
        if len(site.species) > 1:
            # Find minority species occupancy
            occupancies = sorted(site.species.values())
            min_occupancy = min(min_occupancy, occupancies[0])
    return min_occupancy
```

### Large-Scale Preprocessing (>100 Candidates)

```python
import json
from tqdm import tqdm

def preprocess_batch(candidates_file, output_file, checkpoint_freq=10):
    """
    Preprocess large batch with checkpointing.
    """
    # Load candidates
    with open(candidates_file) as f:
        raw_candidates = json.load(f)
    
    # Check for checkpoint
    preprocessed = []
    start_index = 0
    
    if os.path.exists(output_file):
        with open(output_file) as f:
            preprocessed = json.load(f)
            start_index = len(preprocessed)
            print(f"Resuming from candidate {start_index}")
    
    # Process remaining candidates
    for i, candidate in enumerate(tqdm(raw_candidates[start_index:], 
                                      desc="Preprocessing")):
        try:
            # Preprocess single candidate
            result = preprocess_candidates([candidate])
            preprocessed.extend(result)
            
            # Checkpoint every N candidates
            if (i + 1) % checkpoint_freq == 0:
                with open(output_file, 'w') as f:
                    json.dump(preprocessed, f, indent=2)
        
        except Exception as e:
            print(f"ERROR processing {candidate.get('candidate_id')}: {e}")
            continue
    
    # Final save
    with open(output_file, 'w') as f:
        json.dump(preprocessed, f, indent=2)
    
    print(f"Preprocessing complete: {len(raw_candidates)} → {len(preprocessed)} candidates")
    return preprocessed

# Usage
preprocessed = preprocess_batch(
    'lanthanide_phosphors_raw.json',
    'lanthanide_phosphors_preprocessed.json',
    checkpoint_freq=10
)
```

---

## Validation Flags After Preprocessing

After preprocessing, attach validation metadata to track approximation validity:

```python
def add_validation_flags(candidate):
    """Add validation flags based on preprocessing method and doping."""
    preprocessing = candidate.get('preprocessing_applied', 'none')
    doping_conc = candidate.get('metadata', {}).get('doping_concentration', 0)
    
    if preprocessing == 'majority_orderer':
        # Check if approximation is valid
        approximation_valid = doping_conc < 0.10
        requires_dft = doping_conc >= 0.10
        confidence_penalty = 0.0 if approximation_valid else 0.2
        
        candidate['validation_flags'] = {
            "preprocessing_applied": "majority_orderer",
            "approximation_valid": approximation_valid,
            "requires_dft_validation": requires_dft,
            "confidence_penalty": confidence_penalty,
            "doping_concentration": doping_conc
        }
    
    elif preprocessing == 'enumeration_orderer':
        # All configurations valid, no approximation
        candidate['validation_flags'] = {
            "preprocessing_applied": "enumeration_orderer",
            "approximation_valid": True,
            "requires_dft_validation": False,  # Already exhaustive
            "confidence_penalty": 0.0,
            "note": "One of multiple ordered configurations"
        }
    
    elif preprocessing == 'sqs_orderer':
        # SQS approximation always valid for high concentrations
        sqs_error = candidate.get('ordering_metadata', {}).get('sqs_error', 0)
        good_convergence = sqs_error < 0.01
        
        candidate['validation_flags'] = {
            "preprocessing_applied": "sqs_orderer",
            "approximation_valid": good_convergence,
            "requires_dft_validation": not good_convergence,  # Validate if poor convergence
            "confidence_penalty": 0.1 if good_convergence else 0.3,
            "sqs_error": sqs_error
        }
    
    else:
        # No preprocessing (already ordered)
        candidate['validation_flags'] = {
            "preprocessing_applied": "none",
            "approximation_valid": True,
            "requires_dft_validation": False,
            "confidence_penalty": 0.0
        }
    
    return candidate
```

---

## Using Validation Flags in Ranking

Integrate confidence penalties into multi-objective ranking:

```python
def calculate_confidence_score(candidate):
    """Calculate confidence-adjusted score for ranking."""
    base_score = candidate['scores']['total_score']
    confidence_penalty = candidate['validation_flags']['confidence_penalty']
    
    # Apply penalty
    adjusted_score = base_score * (1.0 - confidence_penalty)
    
    # Flag for DFT if high score but low confidence
    if base_score > 0.8 and confidence_penalty > 0.1:
        candidate['requires_dft_validation'] = True
        candidate['priority'] = 'high'
    
    return adjusted_score

# Apply to all candidates
for candidate in filtered_candidates:
    candidate['confidence_score'] = calculate_confidence_score(candidate)

# Re-rank by confidence score
ranked_candidates = sorted(filtered_candidates, 
                          key=lambda x: x['confidence_score'], 
                          reverse=True)
```

---

## Troubleshooting

### Error: "Structure has partial site occupancies"

**Cause:** Disordered structure passed to MatGL/matcalc without preprocessing

**Solution:** Add preprocessing layer before validation step

```python
if not structure.is_ordered:
    result = pymatgen_majority_orderer(input_structures=structure_cif, ...)
    structure = result['structures'][0]
```

### Issue: Enumeration orderer returns too many structures

**Cause:** `n_structures` set too high or small supercell with high entropy

**Solution:** Reduce `n_structures` to top 10-20, or increase `supercell_size`

```python
# Don't screen all 100 configurations
result = pymatgen_enumeration_orderer(
    ...,
    n_structures=10,  # Just top 10 by Ewald energy
    sort_by='ewald'
)
```

### Issue: SQS orderer high sqs_error (poor convergence)

**Cause:** Insufficient MC steps or too small supercell

**Solution:** Increase `n_mc_steps` and/or `supercell_size`

```python
# For high-entropy (4+ species), need more steps
result = pymatgen_sqs_orderer(
    ...,
    supercell_size=5,  # Larger supercell
    n_mc_steps=500000,  # More MC steps
)
```

### Issue: Unknown which strategy to use

**Cause:** No metadata from generator, unclear doping concentration

**Solution:** Calculate doping concentration manually

```python
def estimate_strategy(structure):
    """Heuristic to determine ordering strategy."""
    # Calculate minority species fraction
    min_occupancy = min(
        min(site.species.values()) 
        for site in structure if len(site.species) > 1
    )
    
    if min_occupancy < 0.10:
        return 'majority'
    elif min_occupancy > 0.20:
        return 'sqs'
    else:
        return 'majority'  # Use for screening, validate with SQS later
```

---

## Performance Impact

| Strategy | Input Size | Output Count | Time per Structure | Screening Impact |
|----------|------------|-------------|-------------------|------------------|
| Majority | Unit cell | 1 | ~0.1s | Negligible (~10s for 100 candidates) |
| Enumeration | 2×2×2 supercell | 10-50 | ~2-10s | Moderate (10× candidates × 2-10s = 200-1000s) |
| SQS | 4×4×4 supercell | 1-5 | ~10-60s | High (large supercells → slower ML calc)|

**Recommendation:** Use majority orderer for initial screening (minutes), then upgrade top candidates to SQS for validation (hours).
