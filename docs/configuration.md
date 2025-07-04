# Configuration

The application can be customized through environment variables.
These variables can be set in a `.env` file or exported in your shell.

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | API key for OpenAI | - |
| `LLM_MODEL` | LLM model name | `gpt-4.1-nano` |
| `POPULATION_SIZE` | Number of population agents to spawn | `36` |
| `SELF_IMPROVE_AFTER` | Comma separated conversation numbers triggering self improvement | `1,10,35` |
| `PARALLEL_CONVERSATIONS` | Run conversations concurrently when `true` | `true` |
| `DSPY_MIPRO_MINIBATCH_SIZE` | Mini-batch size for DSPy Mipro optimizer | `4` |
| `DSPY_BOOTSTRAP_MINIBATCH_SIZE` | Mini-batch size for DSPy bootstrap optimizer | `2` |
| `MAX_TURNS` | Turns per conversation | `6` |
| `HISTORY_BUFFER_LIMIT` | Stored conversations retained by the wizard | `50` |
| `LOG_LEVEL` | Verbosity of runtime logs | `INFO` |

