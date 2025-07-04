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

## Debugging CORS issues

Earlier versions of the project included a helper script `scripts/debugCORS.py` for testing cross-origin behavior. The file has been removed, but its functionality can be recreated with this minimal Flask snippet:

```python
from flask import Flask
app = Flask(__name__)

@app.route('/')
def index():
    return 'OK'

if __name__ == '__main__':
    app.run(debug=True)
```

Run it locally whenever you need a quick server to troubleshoot CORS problems.

