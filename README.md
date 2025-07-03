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
3. Copy `.env.example` to `.env` and update the values
   ```bash
   cp .env.example .env
   # edit .env and set OPENAI_API_KEY
   ```
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
