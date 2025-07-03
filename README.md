# AIgentChat-Dspy-light

This project provides a simplified backend demonstrating a multi-agent research system. It uses OpenAI through LangChain and DSPy for prompt optimization.

- [Installation](docs/installation.md)
- [Usage](docs/usage.md)
- [Troubleshooting](docs/troubleshooting.md)

## Architecture Overview

The core of the system lives in `integrated_system.py`. A `PopulationGenerator` first builds agent specifications which are used by the `GodAgent` to spawn `PopulationAgent` instances. A single `WizardAgent` converses with each population agent and submits logs to an `EnhancedJudgeAgent` for scoring. The `TokenTracker` and `StructuredLogger` record usage statistics and events.

```
PopulationGenerator -> GodAgent -> PopulationAgent
                               \-> WizardAgent -> EnhancedJudgeAgent
```

`main.py` wires these components together and exposes a command line interface.

## Quick Start

Install dependencies and templates then run the analysis simulation:
```bash
pip install -r requirements.txt
python create_templates.py  # if templates are missing
export OPENAI_API_KEY=sk-...
python main.py
```

To launch the (currently experimental) dashboard run:
```bash
python main.py --dashboard
```
