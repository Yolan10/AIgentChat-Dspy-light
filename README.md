# AIgentChat-Dspy-light

This project provides a simplified backend demonstrating a multi-agent research system. It uses OpenAI via LangChain and DSPy for prompt optimization.

## Usage

1. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```
2. Create default templates if they do not exist
   ```bash
   python create_templates.py
   ```
3. Export your `OPENAI_API_KEY` and run the simulation
   ```bash
   export OPENAI_API_KEY=sk-...
   python main.py
   ```

## Logs

All runtime logs are stored in the `logs/` directory. The folder is kept under version control using a `.gitkeep` file, while other log files are ignored via `.gitignore`.
The main application writes to `logs/system.log` and `logs/token_usage.json`.

Use the helper script `scripts/clean_logs.py` to archive or clear old log files.
