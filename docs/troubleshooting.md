# Troubleshooting

Common issues and how to resolve them.

## Missing OpenAI key
If you encounter `Missing OPENAI_API_KEY` errors, ensure that the environment variable is set:
```bash
export OPENAI_API_KEY=sk-...
```

## Templates not found
If the system reports missing template files, generate them with:
```bash
python scripts/create_templates.py
```

## API connection errors
Network or authentication problems may cause OpenAI API requests to fail. Verify your API key and internet connection.

