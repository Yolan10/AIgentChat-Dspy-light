# Usage

This project demonstrates a simplified multi-agent research system. Once the dependencies are installed and the `OPENAI_API_KEY` is exported, you can run the simulation or launch the dashboard.

## Running the analysis simulation

```bash
python main.py
```

The system will generate a population of agents and simulate conversations with them using the integrated agent framework.
After the run, execute `scripts/analyze_results.py` to summarize log data.

## Starting the (experimental) dashboard

```bash
python main.py --dashboard
```

The dashboard is currently a placeholder and will simply print a message that it is not yet implemented.

