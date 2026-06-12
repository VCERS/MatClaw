# MatClaw

**Give an AI agent a lab, not a login shell.**

MatClaw turns computational materials discovery into something an AI agent can run end-to-end — generating candidate structures, screening them with ML, validating the survivors with DFT, and planning their synthesis — while the heavy compute stays safely behind a controlled interface. No SSH keys handed out, no shell access, no surprises.

---

## Why three layers?

Materials discovery is compute-heavy and security-sensitive. ML property prediction needs GPUs, DFT validation needs an HPC cluster, and a single screening campaign can touch thousands of candidates. Three problems shaped MatClaw's architecture — and each layer solves one:

### 🛡️ MCP Server (Tools) — *the safe interface to your hardware*
You don't want to hand an agent SSH access to your GPU box or HPC cluster. So the **MCP server runs on the hardware** and exposes a fixed catalog of well-scoped **tools** over stdio or HTTP. An agent on a laptop calls `matgl_predict_bandgap` or `dft_submit_calculation` and gets structured results back — without a shell, a filesystem path, or any capability the tool doesn't explicitly grant. The tool boundary *is* the security and audit boundary: the agent can do exactly what the tools allow, and nothing else.

### 📚 Skills — *teaching the agent how to do science, not just call functions*
Knowing the tools isn't the same as knowing the workflow. A tool can write a VASP input file; it can't decide that your oxide needs spin polarization, or that ML screening should come *before* you spend HPC hours on DFT. **Skills are the playbooks** — they teach an agent how to chain tools into a complete, judgment-laden workflow: *generate → screen → validate → synthesize.*

### ⚡ Python SDK — *throughput for when 1 candidate becomes 1,000*
Driving tools one agent-call at a time is fine for a handful of structures, but it doesn't scale to a high-throughput campaign. **`matclaw_sdk` exposes every server tool as a plain Python import**, so an agent (or a human) can write a script that loops over a thousand candidates programmatically — same tools, same server, batch throughput — instead of issuing thousands of individual tool calls.

```
      AI agent  (laptop · IDE · remote · sandbox)
                          │
   ┌──────────────────────▼───────────────────────┐
   │         Skills — workflow playbooks          │────┐
   │    generate → screen → DFT → synthesize      │    │  matclaw_sdk
   └──────────────────────┬───────────────────────┘    │  Python client for 
                          │  direct tool-calls         │  large batch jobs
   ┌──────────────────────▼───────────────────────┐    │
   │  MCP Server — a fixed, safe catalog of Tools │◄───┘
   └──────────────────────┬───────────────────────┘
                          │  runs server-side, on the metal
   ┌──────────────────────▼───────────────────────┐
   │           GPUs · HPC/SLURM · databases       │
   └──────────────────────────────────────────────┘
```

---

## The discovery funnel

The materials skills compose into one autonomous high-throughput pipeline that runs all the way from a hypothesis to a synthesized, characterized material. Each stage narrows the field and hands provenance to the next:

```
      IN SILICO  ·  narrow the field computationally
      ┌──────────────────────────────────────────────┐
      │            Candidate generation              │
      │   generate & enumerate 1000s of candidates   │
      └───────────────────────┬──────────────────────┘
                              ▼
      ┌──────────────────────────────────────────────┐
      │             Candidate screening              │
      │    ML property + stability screen → 100s     │
      └───────────────────────┬──────────────────────┘
                              ▼
      ┌──────────────────────────────────────────────┐
      │                      DFT                     │
      │   DFT verification → 10s high-confidence     │
      └───────────────────────┬──────────────────────┘
                              ▼
      ┌──────────────────────────────────────────────┐
      │               Synthesis planning             │
      │       plan a synthesis route → recipe        │
      └───────────────────────┬──────────────────────┘
                              │  high-confidence candidates → the bench
                              ▼
      IN THE LAB  ·  make it, then learn from it
      ┌──────────────────────────────────────────────┐
      │         Experiment orchestration*            │◄──┐
      │   robotic synthesis + XRD characterization   │   │
      └───────────────────────┬──────────────────────┘   │ refine
                              ▼                          │ synthesis
      ┌──────────────────────────────────────────────┐   │
      │                Active learning               │   │
      │     suggest next conditions (ARROWS / BO)    │───┘
      └───────────────────────┬──────────────────────┘
                              ▼
      ┌──────────────────────────────────────────────┐
      │             validated material               │
      └──────────────────────────────────────────────┘

      * planned — robotic experiment orchestration (tools + skill), on the roadmap
```

ML screening is cheap, so it runs first and wide; DFT is expensive, so it runs last and narrow — only on the candidates that earned it. The `vasp` and `orca` skills ingest the ML predictions `candidate-screener` flagged for verification and return DFT-confirmed values with upgraded confidence.

The candidates that clear DFT with high confidence graduate from *in silico* to the bench. There, a planned **robotic experiment-orchestration** layer executes the synthesis recipe on automated hardware and characterizes the product (e.g. by XRD), while **active-learning** (ARROWS / Bayesian optimization) reads each result and proposes the next set of conditions — closing a self-driving loop that refines the synthesis procedure until the target phase is obtained. This is also where MatClaw's robotics tooling (Isaac / URDF / Lula) meets the discovery pipeline.

---

## Available Skills

### Materials discovery
| Skill | Description |
|-------|-------------|
| **candidate-generator** | Generate candidate materials using pymatgen structure-manipulation tools |
| **candidate-screener** | Screen candidates via hierarchical property retrieval (Materials Project → ASE cache → ML) and multi-objective ranking |
| **stability-analyzer** | Assess thermodynamic stability and convex-hull distance for candidate structures |
| **vasp** | End-to-end periodic plane-wave DFT with VASP: design, submit/monitor, triage, and interpret calculations via the `dft_*` tools |
| **orca** | End-to-end molecular quantum chemistry with ORCA: design, submit/monitor, triage, interpret, and generate visualization cubes via the `dft_*` and `orca_*` tools |
| **synthesis-planner** | Synthesis route planning — literature-first (Materials Project), falling back to template-based routes when no literature data exists |
| **active-learning** | Autonomous synthesis optimization with ARROWS and automated XRD characterization |

### Robotics & GPU (NVIDIA Isaac / CUDA)
| Skill | Description |
|-------|-------------|
| **urdf-validator** | Validate and auto-fix URDF robot models for Isaac Sim / USD compatibility |
| **lula-description-generator** | Generate Lula robot descriptions with collision-sphere placement for Isaac Sim |
| **isaac-lab-scene-init** | Initialize robot scenes in NVIDIA Isaac Lab |
| **nsys-optimizer** | Profile and optimize CUDA/GPU code using NVIDIA Nsight Systems |

## Available Tools

| Category | Tools |
|----------|-------|
| **DFT (VASP + ORCA)** | Async HPC job lifecycle for plane-wave (VASP) and molecular (ORCA) DFT — prepare, submit, poll, fetch, restart, cancel (`dft_prepare_calculation`, `dft_submit_calculation`, `dft_get_calculation_status`, `dft_fetch_results`, `dft_restart_calculation`, `dft_cancel_calculation`) |
| **ORCA** | ORCA output analysis and `orca_plot` cube generation (`orca_summarize_output`, `orca_batch_summarize_outputs`, `orca_validate_environment`, `orca_validate_calc_dir`, `orca_find_matching_gbw`, `orca_generate_homo_lumo_cubes`, `orca_generate_density_esp_cubes`, …) |
| **Materials Project** | Material search, property data, synthesis recipes (`mp_search_materials`, `mp_get_material_properties`, `mp_get_detailed_property_data`, `mp_search_recipe`) |
| **PubChem** | Chemical compound search, properties, and safety data (`pubchem_search_compounds`, `pubchem_get_compound_properties`, `pubchem_get_safety_data`) |
| **Pymatgen** | Structure generation: substitution, enumeration, defects, SQS, ion exchange, perturbation, prototypes, ordering (12 tools) |
| **Analysis** | Structure validation, composition analysis, structure analysis, fingerprinting (`structure_validator`, `composition_analyzer`, `structure_analyzer`, `structure_fingerprinter`) |
| **MatGL** | ML predictions for relaxation, band gap, and formation energy (`matgl_relax_structure`, `matgl_predict_bandgap`, `matgl_predict_eform`) |
| **matcalc** | ML-potential property calculations: elasticity, phonons, EOS, surfaces, MD, NEB, and more (11 tools) |
| **ASE** | Result database management (`ase_connect_or_create_db`, `ase_store_result`, `ase_query`, `ase_get_atoms`, `ase_list_databases`) |
| **ChemLLM** | Molecule binding and synthesizability prediction via fine-tuned LLMs (`predict_molecule_binding`, `predict_molecule_synthesizability`) |
| **Selection** | Multi-objective ranking — Pareto, weighted sum, constraint-based (`multi_objective_ranker`) |
| **Synthesis Planning** | Recipe quantification and template-based route generation (`synthesis_recipe_quantifier`) |
| **ElemwiseRetro** | Synthesis recipe prediction for inorganic solid-state synthesis (`er_predict_precursors`, `er_predict_temperature`) |
| **ARROWS** | Active-learning campaign management for synthesis (`arrows_initialize_campaign`, `arrows_suggest_experiment`, `arrows_record_result`) |
| **Bayesian Optimization** | Active-learning campaign management via Bayesian optimization (`bo_initialize_campaign`, `bo_suggest_experiment`, `bo_record_result`) |
| **Characterization** | Automated phase identification from powder XRD using deep learning (`xrd_analyze_pattern`) |
| **Image Retrieval** | Scientific-paper figure extraction, image segmentation, SEM classification |
| **URDF** | Robot model validation and fixing for Isaac Sim/USD (`urdf_validate`, `urdf_fix`, `urdf_inspect`) |
| **Lula** | Lula robot descriptions with automated collision-sphere placement (`lula_generate_robot_description`) |

---

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

For DFT job submission, point the server at your scheduler and engine paths via `config.yaml` (see `mcp/config.example.yaml`) or `MATCLAW_DFT_*` environment variables.

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

> **⚠️ Stdio timeout caveat:** The SDK's stdio transport processes requests sequentially — the server handles one tool call at a time. If a request times out, the server keeps running it, so subsequent requests queue behind it. A timeout that's too short can cause cascading delays where every new request hits the already-busy server. Set `timeout` generously (e.g., 600s for relaxations) or use the HTTP transport below.

### HTTP (streamable-http)

Start the server as a long-lived HTTP process, then connect from anywhere:

```bash
cd mcp
source venv/bin/activate

# Single worker — local only
python server.py --transport streamable-http --port 8500

# Multiple workers — concurrent requests (recommended)
python server.py --transport streamable-http --port 8500 --workers 4

# Remote access — bind to all interfaces
python server.py --transport streamable-http --host 0.0.0.0 --port 8500 --workers 4
```

> **✅ Why HTTP + workers is recommended:** Each worker is an independent process, so a slow tool call (e.g., a 10-minute relaxation) on worker 1 doesn't block worker 2 from handling the next request. HTTP is also stateless, eliminating the protocol desynchronization risk that stdio faces when timeouts occur. For batch screening scripts making many sequential tool calls, this dramatically improves throughput and reliability.

Switch the SDK to HTTP mode by editing `sdk/config.yaml`:
```yaml
transport: http
http:
  url: http://localhost:8500
timeout: 600  # generous per-request timeout; no desync risk with HTTP
```

Or set environment variables instead:
```bash
export MATCLAW_TRANSPORT=http
export MATCLAW_HTTP_URL=http://localhost:8500
export MATCLAW_TIMEOUT=600
```

## Python SDK (`matclaw_sdk`)

The SDK is a thin client over the MCP server: call any tool with a direct Python import — no MCP boilerplate, no per-call agent round-trip. This is the path for **large-batch jobs**, where you loop over many candidates programmatically.

### Quick start

```python
from matclaw_sdk import mp_search_materials, matgl_predict_bandgap

result = mp_search_materials(formula="NaCl")
print(result)
```

All tools exposed by the server are available as top-level imports from `matclaw_sdk`.

### Batch screening example

```python
from matclaw_sdk import matgl_predict_eform

# Screen a thousand candidate structures through the MCP server — no SSH,
# the GPU work happens server-side, you just collect the results.
survivors = []
for structure in candidate_structures:          # e.g. 1,000 candidates
    eform = matgl_predict_eform(input_structure=structure)
    if eform["success"] and eform["formation_energy_per_atom"] < -0.5:
        survivors.append(structure)
print(f"{len(survivors)} candidates passed ML screening")
```

### Listing available tools

```python
from matclaw_sdk import get_tools, show_tools

show_tools()                 # print a formatted list
tools = get_tools()          # or get the raw list for programmatic use
```

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
