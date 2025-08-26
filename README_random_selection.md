# Random Selection in Best-of-N Rollout

This document explains how to use the new random selection feature in the `vLLMBestOfN` rollout, which allows you to select plans randomly instead of using the oracle/autorater service.

## Overview

The `vLLMBestOfN` rollout now supports two modes of plan selection:

1. **Oracle Selection (Default)**: Uses the autorater service to evaluate and select the best plan
2. **Random Selection**: Randomly selects a plan from the generated candidates

## Configuration

### 1. Command Line Arguments

You can enable random selection using the following command line arguments:

```bash
python math_eval.py \
    --rollout_name vllm_best_of_n \
    --n_candidates 5 \
    --use_random_selection \
    --random_seed 42
```

### 2. Configuration File

You can also set these parameters in your configuration file:

```python
from evaluation_base import EvaluationConfig

config = EvaluationConfig(
    rollout_name="vllm_best_of_n",
    n_candidates=5,
    use_random_selection=True,
    random_seed=42,
    # ... other parameters
)
```

### 3. Rollout Config JSON

The `rollout_config.json` file now includes these parameters:

```json
{
    "rollout": {
        "name": "vllm_best_of_n",
        "n_candidates": 10,
        "use_random_selection": false,
        "random_seed": 42,
        // ... other parameters
    }
}
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_random_selection` | bool | `False` | Enable random selection instead of oracle |
| `random_seed` | int | `42` | Seed for reproducible random selection |
| `n_candidates` | int | `10` | Number of candidates to generate per prompt |

## Use Cases

### 1. **Baseline Comparison**
Use random selection to establish a baseline performance without the oracle:
```bash
python math_eval.py --rollout_name vllm_best_of_n --use_random_selection
```

### 2. **Reproducible Experiments**
Use a fixed seed for reproducible results:
```bash
python math_eval.py --rollout_name vllm_best_of_n --use_random_selection --random_seed 123
```

### 3. **Oracle vs Random Analysis**
Compare performance between oracle and random selection:
```bash
# Oracle selection
python math_eval.py --rollout_name vllm_best_of_n --n_candidates 5

# Random selection
python math_eval.py --rollout_name vllm_best_of_n --n_candidates 5 --use_random_selection
```

## Example Script

See `example_random_selection.py` for a complete example:

```python
#!/usr/bin/env python3
from evaluation_base import EvaluationConfig
from math_eval import MathEvaluator

# Create configuration with random selection enabled
config = EvaluationConfig(
    rollout_name="vllm_best_of_n",
    n_candidates=5,
    use_random_selection=True,
    random_seed=123,
    dataset="math500",
    template_type="plan_first",
)

# Run evaluation
evaluator = MathEvaluator(config)
evaluator.run_evaluation()
```

## How It Works

1. **Generation**: Generate `n_candidates` responses for each prompt
2. **Extraction**: Extract plan candidates from `<answer></answer>` tags
3. **Selection**: 
   - If `use_random_selection=True`: Randomly select one candidate
   - If `use_random_selection=False`: Use autorater service to select best candidate
4. **Continuation**: Continue generation with the selected candidate

## Benefits

- **No External Dependencies**: Works without autorater service
- **Reproducible**: Fixed seed ensures consistent results
- **Fast**: No network calls for plan evaluation
- **Baseline**: Establishes performance lower bound
- **Research**: Useful for ablation studies

## Limitations

- **No Quality Control**: Random selection may choose poor plans
- **Lower Performance**: Expected to perform worse than oracle selection
- **No Learning**: Doesn't improve from feedback

## Output

When random selection is enabled, you'll see output like:

```
Initialized vLLMBestOfN with n_candidates=5
Plan selection: Random
Random selection seed: 42
...
Prompt 0: Using random selection from 5 candidates
Random selection: selected candidate 3 from 5 candidates
```

## Troubleshooting

### Random Selection Always Picks Same Candidate
- Ensure `random_seed` is different for each experiment
- Check that `prompt_idx` is being used in seed calculation

### No Random Selection
- Verify `use_random_selection=True` is set
- Check that `rollout_name` is `vllm_best_of_n`
- Ensure configuration is properly loaded

### Inconsistent Results
- Use different `random_seed` values
- Check that the seed is being properly applied
- Verify no other randomization is affecting results 