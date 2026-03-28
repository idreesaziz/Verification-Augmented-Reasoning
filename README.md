# Verification-Augmented Reasoning (VAR)

Adversarial empirical falsification of LLM reasoning through **premise provenance tracking** and **information-firewalled verification**. Every intermediate claim is classified by its origin — inherited, computed, or derived — and derived claims are stress-tested by an independent adversarial verifier that generates its own falsification code without seeing the reasoning chain.

## Core Insight

LLM reasoning agents fail not at code execution but at **wrong modeling assumptions** — probabilistic claims asserted without computation, logical leaps stated as fact. Standard verification (assertions, formal proofs, self-consistency) cannot catch these because:

- **Self-authored verification**: The reasoning model writes its own checks and will always find a tautology
- **Formal methods**: Can prove `C(25,2) = 300` but cannot detect that the model is computing the *wrong quantity*
- **Neural verifiers**: Score steps opaquely without producing falsifiable evidence

The solution: **premises are typed, and derived premises are independently falsified.**

## Architecture

### Premise Provenance

Every premise in the reasoning chain has exactly one type:

| Type | Source | Verification | Example |
|------|--------|-------------|---------|
| **INHERITED** | Verbatim from problem statement | None needed — axiom | "There are 25 segments" |
| **COMPUTED** | Printed by executed code in a prior step | None needed — empirical fact | "Intersection probability = 0.4722" |
| **DERIVED** | Logical deduction from prior premises | **Adversarial falsification** | "Every pair of chords intersects" |

The model does not classify its own premises. The system classifies them by comparing each premise against (a) the problem text and (b) prior step observations. **Everything that isn't INHERITED or COMPUTED is DERIVED, and gets stress-tested.**

### Adversarial Verification Pipeline

DERIVED premises are sent to an independent adversarial verifier behind an **information firewall**:

```
  Reasoning Agent                          Adversarial Verifier
  ─────────────                            ────────────────────

  Step N produces:                         Receives ONLY:
  ┌─────────────────────┐                  ┌─────────────────────┐
  │ [INHERITED] 25 segs │──── skip ────    │ Problem statement   │
  │ [COMPUTED]  0.4722  │──── skip ────    │                     │
  │ [DERIVED]   "all    │────────────────► │ Bare claim: "all    │
  │   pairs intersect"  │  FIREWALL        │   pairs intersect"  │
  └─────────────────────┘   (no reasoning  └──────────┬──────────┘
                             chain, no code,           │
                             no prior steps)           ▼
                                           ┌──────────────────────┐
                                           │ Choose attack tool:  │
                                           │                      │
                                           │ • Z3  → find a       │
                                           │   counterexample     │
                                           │                      │
                                           │ • Monte Carlo → sim  │
                                           │   the actual process  │
                                           │   and measure         │
                                           │                      │
                                           │ • Brute-force →      │
                                           │   enumerate all cases │
                                           └──────────┬──────────┘
                                                      │
                                                      ▼
                                           ┌──────────────────────┐
                                           │ Sandbox execution    │
                                           │ → produces a number  │
                                           └──────────┬──────────┘
                                                      │
                                                      ▼
                                           ┌──────────────────────┐
                                           │ Statistical test     │
                                           │ (e-value, exact      │
                                           │  match, SAT/UNSAT)   │
                                           │                      │
                                           │ → REJECT / SURVIVE   │
                                           └──────────────────────┘
```

### Key Properties

**Information firewall**: The verifier sees the problem + bare claim only. It cannot inherit the reasoner's blind spots because it never sees the reasoning.

**Adversarial framing**: The verifier is prompted to *assume the claim is wrong* and find the hidden assumption that makes it false. It generates falsification code, not confirmation code.

**Dual falsification tools**:
- **Z3** — for universal/logical claims ("all X satisfy P" → find counterexample)
- **Monte Carlo** — for stochastic/probabilistic claims ("E[X] = v" → simulate and measure)
- The verifier chooses whichever tool is most likely to break the claim.

**Model cannot influence verification**: The adversary generates the code. A fixed statistical test makes the accept/reject decision. The reasoning model never touches either.

**Structured outputs everywhere**: All LLM calls use structured output schemas (Pydantic models). Premise types, claim formats, tool choices — all enforced by the schema, not by prompt compliance.

### E-Value Statistical Engine

For Monte Carlo falsification, the system uses **e-values** for hypothesis testing:

- **Proportion claims**: Likelihood-ratio e-value with MLE alternative
- **Continuous claims**: Gaussian likelihood-ratio via CLT
- **Deterministic claims**: Exact match or infinite evidence

Thresholds: `E > 1000` → REJECT, `E > 20` → SUSPECT

E-values are **anytime-valid** and **composable** (multiply across steps — no Bonferroni needed).

### Failure Modes and Safeguards

| Failure | Mitigation |
|---------|------------|
| Simulation timeout | Progressive fallback (50K → 5K → 500 trials). Timeout → BLOCK, never PASS. |
| False rejection (buggy sim) | High e-value threshold. Convergence check at two sample sizes. Worst case = one retry. |
| Non-simulatable claim | Reported as INCONCLUSIVE. Z3 may still handle it. Some claims genuinely can't be tested. |
| Model mislabels premises | Model doesn't classify — the system does, by matching against problem text and prior observations. |

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

## Project Structure (Redesign — WIP)

```
src/var_reasoning/
├── engine/             # Core reasoning loop
│   ├── step_engine.py          # Step execution + code repair
│   └── backtracking.py         # Retry / backtrack on rejection
├── models/             # LLM provider, Pydantic schemas, session state
│   ├── gemini_provider.py      # Structured output LLM calls
│   ├── schemas.py              # Enforced schemas for all LLM outputs
│   └── state.py                # Session, CompletedStep, premise chain
├── prompts/            # System prompts
│   ├── reasoning_prompt.py     # Reasoner: compute, don't assume
│   └── adversary_prompt.py     # Adversarial verifier: assume wrong, falsify
├── provenance/         # Premise provenance system
│   ├── classifier.py           # Classify premises: INHERITED / COMPUTED / DERIVED
│   └── chain.py                # Provenance graph across steps
├── verification/       # Adversarial falsification engine
│   ├── adversary.py            # Firewalled adversarial verifier orchestrator
│   ├── e_value.py              # E-value statistical engine
│   └── sandbox.py              # Isolated code execution (subprocess)
└── benchmarks/         # Dataset loaders (GSM8K, MATH, FOLIO, HumanEval)
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
