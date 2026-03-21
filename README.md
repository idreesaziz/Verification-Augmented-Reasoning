# Verification-Augmented Reasoning (VAR)

A research system that improves LLM reasoning by having models write Python code to investigate problems, generate formal verification targets for their own inferences, and backtrack when verification fails.

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
| C (full VAR) | Gemini 2.5 Flash | Yes | Yes | Yes |
| D (ceiling) | Gemini 2.5 Pro | No | No | No |

## Benchmarks

- **GSM8K**: Grade school math word problems
- **MATH**: Competition mathematics
- **FOLIO**: First-order logic inference
- **HumanEval**: Code generation

## Running Tests
```bash
pytest
```
