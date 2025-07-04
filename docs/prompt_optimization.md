# Prompt Optimization

The project includes a small helper in `core/dspy_utils.py` for refining DSPy programs.

## `apply_dspy_optimizer`

`apply_dspy_optimizer(program, metric, trainset)` runs DSPy's `BootstrapFewShot` and
`MIPROv2` optimizers using batch sizes defined in `config.py`. It returns the
optimized DSPy program.

Population agents or the `GodAgent` can optionally call this function to improve
their prompts. A minimal example looks like this:

```python
from core.dspy_utils import apply_dspy_optimizer
from agents.population_agent import PopulationAgent

def metric(example, pred):
    return 1.0  # implement your scoring logic

agent = PopulationAgent(agent_id="A1", system_instruction="...", spec={})
dspy_program = agent.system_instruction  # or a DSPy Module producing it
optimized = apply_dspy_optimizer(dspy_program, metric, trainset=["demo"])
```

Replace `metric` and `trainset` with your own evaluation function and examples.
The returned program can be used by the agent in subsequent runs.
