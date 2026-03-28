# Verification-Augmented Reasoning (VAR)

A research system that improves LLM reasoning through **multi-layer verification**: each analytical claim is independently tested by a firewalled Monte Carlo simulation, with statistical hypothesis testing via e-values delivering anytime-valid rejection guarantees.

## Architecture

```
Problem ──► Reasoning Step ──► Action Code ──► Observation
                                                   │
                                          ┌────────▼────────┐
                                          │ Generate Inference│
                                          │ (premises, claim) │
                                          └────────┬────────┘
                                                   │
                              ┌─────────────────────┼─────────────────────┐
                              ▼                     ▼                     ▼
                         Layer 0               Layer 0.5              Layer 1
                     Static Sanity           Symbolic Check      Firewalled Simulation
                    ┌──────────────┐     ┌──────────────────┐   ┌──────────────────┐
                    │ Range check  │     │ Assert / SymPy / │   │ Separate LLM call│
                    │ Integrality  │     │ Z3 verification  │   │ (no chain ctx)   │
                    │ Provenance   │     │ Tautology gate   │   │ Monte Carlo code │
                    │ check (AST)  │     │ Pattern-vtype    │   │ Execute isolated │
                    └──────┬───────┘     │ enforcement      │   │ Parse output     │
                           │             └────────┬─────────┘   │ E-value test     │
                           │                      │             └────────┬─────────┘
                           └──────────────────────┼──────────────────────┘
                                                  ▼
                                          Layer 2: Claim Registry
                                       (global consistency check)
                                                  │
                                          ALL PASS ──► Accept step
                                          ANY FAIL ──► Retry / Backtrack
```

### Verification Layers

| Layer | Name | Purpose | Speed |
|-------|------|---------|-------|
| 0 | Static Sanity | Range plausibility, integrality, provenance (AST-based untraced constant detection) | < 1ms |
| 0.5 | Symbolic | Execute assert/SymPy/Z3 code in isolated namespace; tautology gate; pattern-vtype enforcement | < 5s |
| 1 | Simulation | **Firewalled** LLM call generates Monte Carlo code that independently estimates the claimed value. E-value hypothesis test compares analytical vs empirical. | ~15s |
| 2 | Claim Registry | Track all accepted claims; detect contradictions between steps | < 1ms |

### E-Value Statistical Engine

The simulation layer uses **e-values** (evidence measures) for hypothesis testing:

- **Proportion claims** (probabilities): Likelihood-ratio e-value with MLE alternative
- **Continuous claims** (expected values): Gaussian likelihood-ratio via CLT
- **Deterministic claims** (exact counts): Exact match or infinite evidence

Thresholds: `E > 1000` → reject (α ≈ 0.001), `E > 20` → suspect (α ≈ 0.05)

E-values are **anytime-valid** (stop sampling at any time, guarantee holds) and **composable** (multiply across steps — no Bonferroni correction needed).

### Information Firewall

The simulation call is deliberately isolated:
- Receives **only** the problem statement and the specific claim to test
- Does **not** see the reasoning chain, analytical derivation, or intermediate steps
- This prevents the simulator from inheriting the reasoner's misconceptions

## Setup

1. **Install dependencies:**
   ```bash
   pip install -e ".[dev]"
   ```

2. **Set your API key:**
   ```bash
   cp .env.example .env
   # Edit .env and add your Gemini API key
   ```

3. **Ensure Docker is running** (required for sandboxed code execution).

## Usage

### Run AIME evaluation
```bash
# Single problem (AIME 2025 I P13)
python run_aime_single.py

# Full AIME set
python run_aime.py

# AIME 2025 problems
python run_aime_2025.py
```

### Run an experiment
```bash
# Full VAR system on GSM8K
python -m var_reasoning run --condition C --benchmark gsm8k --num-problems 200

# Baseline (one-shot, no tools)
python -m var_reasoning run --condition A --benchmark gsm8k --num-problems 200

# Code execution only (no verification)
python -m var_reasoning run --condition B --benchmark math --num-problems 200

# Ceiling (Gemini 2.5 Pro, one-shot)
python -m var_reasoning run --condition D --benchmark gsm8k --num-problems 200
```

### Analyze results
```bash
python -m var_reasoning analyze --benchmark gsm8k
python -m var_reasoning plot --benchmark gsm8k
```

## Experimental Conditions

| Condition | Model | Code Execution | Verification | Backtracking |
|-----------|-------|---------------|--------------|--------------|
| A (baseline) | Gemini 2.5 Flash | No | No | No |
| B (code exec) | Gemini 2.5 Flash | Yes | No | No |
| C (full VAR) | Gemini 2.5 Flash | Yes | Yes (multi-layer) | Yes |
| D (ceiling) | Gemini 2.5 Pro | No | No | No |

## Project Structure

```
src/var_reasoning/
├── engine/           # Core reasoning loop + backtracking
├── models/           # LLM provider, Pydantic schemas, session state
├── prompts/          # System prompts (reasoning, inference, simulation)
├── sandbox/          # Docker-based code executor
├── verification/     # Multi-layer verification pipeline
│   ├── static_sanity.py        # Layer 0: AST-based sanity gates
│   ├── assert_verifier.py      # Layer 0.5: Python assertions
│   ├── sympy_verifier.py       # Layer 0.5: SymPy symbolic checks
│   ├── z3_verifier.py          # Layer 0.5: Z3 satisfiability
│   ├── tautology_check.py      # Layer 0.5: Reject literal-only assertions
│   ├── simulation_verifier.py  # Layer 1: Firewalled simulation orchestrator
│   ├── e_value.py              # Layer 1: E-value statistical engine
│   ├── claim_registry.py       # Layer 2: Global consistency tracking
│   └── verification_router.py  # Pipeline orchestrator (all layers)
└── benchmarks/       # Dataset loaders (GSM8K, MATH, FOLIO, HumanEval)
```

## Benchmarks

- **GSM8K**: Grade school math word problems
- **MATH**: Competition mathematics
- **FOLIO**: First-order logic inference
- **HumanEval**: Code generation

## Running Tests
```bash
pytest
```
