# MatClaw

**Agent tools and skills for autonomous materials research**

MatClaw is a library of specialized tools and skills designed for AI agents working in computational materials discovery. It provides capabilities across the full materials research lifecycle—from candidate generation and simulation to active learning and experiment planning.

## Architecture

MatClaw follows a layered architecture:

```
┌─────────────────────────────────────────┐
│              AI Agents                  │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│              Skills                     │  ← High-level workflows
│     (orchestrate multiple tools)        │
└─────────────────┬───────────────────────┘
                  │ Direct tool call or matclaw_sdk
┌─────────────────▼───────────────────────┐
│             MCP Server                  │  ← Exposes tools via stdio or HTTP
│   ┌─────────────────────────────────┐   │
│   │           Tools                 │   │
│   └─────────────────────────────────┘   │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│    External Services & Libraries        │
└─────────────────────────────────────────┘
```

**Tools** are implemented within the MCP server and provide atomic operations. **Skills** are agent workflows that call multiple tools through the MCP protocol to accomplish complex research tasks.

## Available Skills

| Skill | Description |
|-------|-------------|
| **urdf-validator** | Validate and auto-fix URDF robot models for Isaac Sim / USD compatibility |
| **lula-description-generator** | Generate Lula robot descriptions with collision-sphere placement for NVIDIA Isaac Sim |
| **isaac-lab-scene-init** | Initialize robot scenes in NVIDIA Isaac Lab |
| **candidate-generator** | Generate candidate materials using pymatgen structure manipulation tools |
| **candidate-screener** | Screen candidate materials using ML prediction and stability analysis |
| **vasp-ase** | VASP DFT calculations using ASE interface |
| **orca_skills** | Quantum chemistry workflows: density/ESP cube generation, frontier orbital analysis, output summarization, directory triage |
| **synthesis-planner** | Intelligent synthesis route planning - always tries literature search (Materials Project) first, falls back to template-based routes only when no literature data exists |
| **active-learning** | Autonomous synthesis optimization using ARROWS with automated XRD characterization |
| **nsys-optimizer** | Profile and optimize CUDA/GPU code using NVIDIA Nsight Systems |

## Available Tools

| Category | Tools |
|----------|-------|
| **URDF** | Robot model validation and fixing for Isaac Sim/USD compatibility (`urdf_validate`, `urdf_fix`, `urdf_inspect`) |
| **Lula** | Generate Lula robot descriptions with automated collision-sphere placement for Isaac Sim motion planning (`lula_generate_robot_description`) |
| **ASE** | Database management (`connect_or_create_db`, `store_result`, `query`, `get_atoms`, `list_databases`) |
| **Materials Project** | Material search, property data, synthesis recipes, detailed property data (`search_materials`, `get_material_properties`, `get_detailed_property_data`, `search_recipe`) |
| **PubChem** | Chemical compound search, properties, and safety data (`search_compounds`, `get_compound_properties`, `get_safety_data`) |
| **Pymatgen** | Structure generation: substitution, enumeration, defects, SQS, ion exchange, perturbation, prototypes (7 tools) |
| **Analysis** | Structure validation, composition analysis, structure analysis, stability analysis, structure fingerprinting (5 tools) |
| **MatGL** | MatGL predictions for structure relaxation, band gap, and formation energy (`matgl_relax_structure`, `matgl_predict_bandgap`, `matgl_predict_eform`) |
| **ChemLLM** | Molecule binding and synthesizability prediction using fine-tuned LLMs (`predict_molecule_binding`, `predict_molecule_synthesizability`) |
| **Selection** | Multi-objective ranking (Pareto, weighted sum, constraint-based) (`multi_objective_ranker`) |
| **ORCA** | Quantum chemistry output analysis and cube file generation (`orca_analysis_tools`, `orca_cube_tools`) |
| **Synthesis Planning** | Recipe quantification and template-based route generation |
| **ElemwiseRetro** | Synthesis recipe prediction for inorganic solid state synthesis (`er_predict_precursors`, `er_predict_temperature`) |
| **ARROWS** | Campaign management for synthesis active learning through ARROWS (`arrows_initialize_campaign`, `arrows_suggest_experiment`, `arrows_record_result`) |
| **Bayesian Optimization** | Campaign management for synthesis active learning through Bayesian Optimization (`bo_initialize_campaign`, `bo_suggest_experiment`, `bo_record_result`) |
| **Characterization** | Automated phase identification from powder diffraction patterns using deep learning (`xrd_analyze_pattern`) |
| **Image Retrieval** | Scientific paper figure extraction, image segmentation, SEM classification (`paper_image_extract`, `image_segmentation`, `sem_image_classification`) |


## Setup

### 1. MCP Server (`mcp/`)

Some dependencies require pre-compiled binaries matching your PyTorch and CUDA versions.

**Option A — auto-install:**
```bash
cd mcp
./setup.sh
```

**Option B — manual:**
```bash
cd mcp
python -m venv venv
source venv/bin/activate
# If needed, update --find-links in requirements.txt for your Torch/CUDA version
pip install -r requirements.txt
```

### 2. Python SDK (`sdk/`)

```bash
cd sdk
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Running the Server

The MCP server supports two transport modes:

### Stdio (default)

VS Code launches the server automatically when configured in `.vscode/mcp.json`. Run manually:

```bash
cd mcp
source venv/bin/activate
python server.py                            # uses --transport stdio (default)
```

### HTTP (streamable-http)

Start the server as a long-lived HTTP process, then connect from anywhere:

```bash
cd mcp
source venv/bin/activate
python server.py --transport streamable-http --port 8500
```

Switch the SDK to HTTP mode by editing `sdk/config.yaml`:
```yaml
transport: http
http:
  url: http://localhost:8500
```

Or set environment variables instead:
```bash
export MATCLAW_TRANSPORT=http
export MATCLAW_HTTP_URL=http://localhost:8500
```

## Python SDK (`matclaw_sdk`)

The SDK provides a thin client over the MCP server, letting you call any tool with a direct Python import — no MCP boilerplate needed.

### Quick start

```python
from matclaw_sdk import mp_search_materials, matgl_predict_bandgap

result = mp_search_materials(formula="NaCl")
print(result)
```

All tools exposed by the server are available as top-level imports from `matclaw_sdk`.

### Configuration

The SDK auto-loads configuration from (highest priority first):

1. **Environment variables** — `MATCLAW_TRANSPORT`, `MATCLAW_HTTP_URL`, `MATCLAW_TIMEOUT`
2. **`~/.matclaw/config.yaml`** — user-wide config
3. **`./matclaw.config.yaml`** — project-specific config
4. **`sdk/config.yaml`** — bundled defaults

```yaml
# sdk/config.yaml
transport: stdio                       # "stdio" or "http"
stdio:
  command: "/path/to/venv/bin/python"
  args: ["/path/to/server.py", "--transport", "stdio"]
http:
  url: http://localhost:8500
  verify_ssl: false
timeout: 30
```

### Async usage

For batch scripts making many sequential tool calls, use `async_call_tool` directly:

```python
import asyncio
from matclaw_sdk import async_call_tool

async def main():
    result = await async_call_tool("mp_search_materials", formula="NaCl")
    print(result)

asyncio.run(main())
```

## Development Status

⚠️ **This project is under active development.** APIs and workflows may change.

## License

See [LICENSE](LICENSE) for details.
