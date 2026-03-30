# Verification-Augmented Reasoning (VAR)

Adversarial empirical falsification of LLM reasoning through **fact provenance tracking** and **information-firewalled verification**. Every intermediate claim is typed by its origin — given, computed, or derived — and derived claims are stress-tested by an independent adversarial verifier that generates its own falsification attempt without ever seeing the reasoning chain.

## Core Insight

LLM reasoning fails not at arithmetic but at **unverified modeling assumptions** — probabilistic claims stated without computation, logical leaps dressed as facts. Standard verification cannot catch these because:

- **Self-authored checks**: The reasoner writes its own assertions and will never assert its own blind spots
- **Formal methods**: Can prove `C(25,2) = 300` but not that the model is computing the *right quantity*
- **Self-consistency**: Agreement between samples of the same model does not rule out shared systematic error

The fix: **every fact is typed, and derived facts are independently falsified.**

## Architecture

### Fact Provenance (FactPool)

Every fact in the reasoning chain has exactly one type:

| Type | Origin | Verification | Example |
|------|--------|-------------|---------|
| **GIVEN** | Problem statement verbatim | Axiom — none needed | "There are 25 chords" |
| **COMPUTED** | Printed by executed code | Empirical — trust the stdout | `P(intersect) = 0.4722` |
| **DERIVED** | Logical deduction from prior facts | **Adversarial falsification required** | "Any two cross-quadrant chords intersect" |

The model does not classify its own outputs. The `FactPool` classifies them structurally: code output → COMPUTED, explicit `Derivation` schema objects → DERIVED, problem text → GIVEN. **The model cannot promote its own assumptions to trusted facts.**

### Adversarial Verification Pipeline

DERIVED facts are forwarded to a firewalled adversarial verifier:

```
  Reasoner                                 Adversarial Verifier
  ────────                                 ────────────────────

  Step N outputs:                          Receives ONLY:
  ┌────────────────────┐                   ┌────────────────────┐
  │ [GIVEN]  25 chords │──── skip ───      │ Problem statement  │
  │ [COMPUTED] 0.4722  │──── skip ───      │                    │
  │ [DERIVED] "chords  │───────────────►   │ Bare claim: "chord │
  │  always intersect" │   FIREWALL        │  pairs intersect"  │
  └────────────────────┘  (no reasoning,   └─────────┬──────────┘
                           no code, no               │
                           prior context)            ▼
                                           ┌─────────────────────┐
                                           │ Choose attack mode: │
                                           │                     │
                                           │ Z3          →       │
                                           │  find counterexample│
                                           │                     │
                                           │ Monte Carlo →       │
                                           │  simulate & measure │
                                           │                     │
                                           │ Brute Force →       │
                                           │  enumerate all cases│
                                           └─────────┬───────────┘
                                                     │
                                                     ▼
                                           ┌──────────────────────┐
                                           │ Isolated subprocess  │
                                           │ execution → number   │
                                           └─────────┬────────────┘
                                                     │
                                                     ▼
                                           ┌──────────────────────┐
                                           │ Statistical decision  │
                                           │ e-value / SAT / exact │
                                           │                       │
                                           │ → REJECT / SURVIVE   │
                                           └──────────────────────┘
```

If REJECT → the step is discarded and the reasoner backtracks with the adversary's counterevidence.  
If SURVIVE → the DERIVED fact is committed to the FactPool at confidence `1 − 1/e`.

### Key Design Properties

**Information firewall**: The verifier sees only the problem statement and the bare isolated claim. It cannot inherit the reasoner's blind spots.

**Adversarial framing**: The verifier is prompted to *assume the claim is wrong* and find the hidden condition that makes it fail. It generates falsification code, not confirmation code.

**Three attack modes**:
- **Z3** — for universal/logical claims: find a counterexample via constraint solving
- **Monte Carlo** — for stochastic/probabilistic claims: simulate the process and compare to the claimed value
- **Brute Force** — for finite enumerable claims: check all cases

**Structured output throughout**: Every LLM call returns a Pydantic-validated schema (`ReasoningStep`, `FalsificationAttempt`, `FinalAnswer`). No prompt-compliance parsing.

**Model cannot influence its own verdict**: The adversary generates the code; a fixed statistical test makes the accept/reject call. The reasoner never touches either.

### E-Value Statistical Engine

Monte Carlo falsification uses **e-values** rather than p-values:

- **Proportion claims**: Likelihood-ratio e-value with MLE alternative
- **Continuous/mean claims**: Gaussian likelihood-ratio via CLT
- **Deterministic claims**: Exact match (e = ∞ or 0)

Thresholds: `e > 1000` → REJECT, `e > 20` → SUSPECT.  
E-values are **anytime-valid** and **composable** — no multiple-testing correction needed across steps.

### Limitations and Known Failure Modes

| Failure | Mitigation |
|---------|------------|
| Adversary simulation timeout | 45s hard cutoff; timeout → INCONCLUSIVE, never forced SURVIVE |
| False rejection from buggy simulation | High e-value threshold (1000×); worst case = one backtrack retry |
| Unsimulatable claim | Reported INCONCLUSIVE; Z3 may still handle it |
| Same-model epistemic correlation | Both reasoner and adversary share the same model's systematic blind spots; firewalling is causal, not cognitive — a known open problem |

The last point is the central limitation of single-model adversarial verification: **prompt-level independence ≠ epistemic independence**. An adversary drawn from the same distribution will tend to reproduce the same modeling errors as the reasoner, even when it cannot see the reasoning.

## Setup

```bash
pip install -e ".[dev]"
```

Set your Gemini API key:
```bash
export GEMINI_API_KEY=your_key_here   # Linux/macOS
$env:GEMINI_API_KEY="your_key_here"  # Windows PowerShell
```

Code execution runs in an isolated subprocess (no Docker required for basic use).

## Usage

### Single problem
```bash
python run_aime_single.py
```

### Experiment suite
```bash
# Full VAR system
python -m var_reasoning run --condition C --benchmark gsm8k --num-problems 200

# Baseline (no tools)
python -m var_reasoning run --condition A --benchmark gsm8k --num-problems 200

# Code execution only, no adversarial verification
python -m var_reasoning run --condition B --benchmark math --num-problems 200
```

### Analysis
```bash
python -m var_reasoning analyze --benchmark gsm8k
python -m var_reasoning plot --benchmark gsm8k
```

## Experimental Conditions

| Condition | Code Execution | Adversarial Verification | Backtracking |
|-----------|---------------|--------------------------|--------------|
| A — baseline | No | No | No |
| B — code only | Yes | No | No |
| C — full VAR | Yes | Yes | Yes |
| D — ceiling (Pro) | No | No | No |

All conditions use Gemini 2.5 Flash except D (Gemini 2.5 Pro).

## Project Structure

```
src/var_reasoning/
├── engine/
│   ├── step_engine.py          # Main loop: reason → compute → falsify → commit
│   ├── backtracking.py         # Backtrack on adversarial rejection
│   └── feedback.py             # Build conversation context from FactPool
├── models/
│   ├── gemini_provider.py      # Gemini API wrapper with structured output
│   ├── schemas.py              # Pydantic schemas for all LLM I/O
│   └── state.py                # FactPool, Session, Fact, FalsificationResult
├── prompts/
│   ├── reasoning_prompt.py     # Reasoner system prompt
│   └── adversary_prompt.py     # Adversary system prompt
├── verification/
│   ├── adversary.py            # Firewalled adversarial falsification orchestrator
│   ├── e_value.py              # E-value statistical engine
│   ├── z3_verifier.py          # Z3 constraint-based verifier
│   └── sympy_verifier.py       # Symbolic math verifier
├── sandbox/
│   └── executor.py             # Isolated code execution
├── benchmarks/                 # Dataset loaders: GSM8K, MATH, FOLIO, HumanEval
└── experiment/                 # Condition runner, metrics, cost tracking
```

## Benchmarks

- **GSM8K** — grade school math word problems
- **MATH** — competition mathematics  
- **FOLIO** — first-order logic natural language inference
- **HumanEval** — code generation

## Running Tests
```bash
pytest
```
