# AIgentChat-Dspy-light

This project provides a simplified backend demonstrating a multi-agent research system. It uses OpenAI through LangChain and DSPy for prompt optimization.

- [Installation](docs/installation.md)
- [Usage](docs/usage.md)
- [Configuration](docs/configuration.md)
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
python scripts/create_templates.py  # if templates are missing
export OPENAI_API_KEY=sk-...
python main.py
```

Logs are written to the `logs/` directory, which is created automatically if it
does not already exist.

To launch the (currently experimental) dashboard run:
```bash
python main.py --dashboard
```
This will start a simple Flask app that displays a placeholder page.

This project provides a simplified backend demonstrating a multi-agent research system. It uses OpenAI via LangChain and DSPy for prompt optimization.


## Usage

1. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```
2. Create default templates if they do not exist
   ```bash
   python scripts/create_templates.py
   ```
3. Copy `.env.example` to `.env` and update the values
   ```bash
   cp .env.example .env
   # edit `.env` and set OPENAI_API_KEY and other options
   ```
   Additional environment variables are described in
   [docs/configuration.md](docs/configuration.md).
4. Run the simulation
   ```bash
   python main.py
   ```

## Docker

1. Build the image and run the default service
   ```bash
   docker compose build
   docker compose run app
   ```
2. To launch the (experimental) dashboard interface
   ```bash
   docker compose run dashboard
   ```


## Logs

All runtime logs are stored in the `logs/` directory. The folder is kept under version control using a `.gitkeep` file, while other log files are ignored via `.gitignore`.
The main application writes to `logs/system.log` and `logs/token_usage.json`.
`StructuredLogger` and the token tracker automatically create the `logs/` directory if it is missing.

Use the helper script `scripts/clean_logs.py` to archive or clear old log files.
Use `scripts/analyze_results.py` to summarize log statistics.

## Running tests

The test suite uses mocked API calls and does not require an OpenAI key. After
installing the dependencies simply run:

```bash
pytest
```


