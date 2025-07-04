import argparse
from pathlib import Path
import config
from core.utils import ensure_logs_dir
from core.integrated_system import IntegratedSystem
from web import create_app


def validate_environment():
    if not config.OPENAI_API_KEY:
        raise SystemExit("OPENAI_API_KEY not set")
    config.validate_configuration()
    ensure_logs_dir()
    required_templates = [
        "templates/population_instruction.txt",
        "templates/research_wizard_prompt.txt",
        "templates/judge_prompt.txt",
        "templates/self_improve_prompt.txt",
    ]
    for path in required_templates:
        if not Path(path).exists():
            print(f"Missing template {path}. Run scripts/create_templates.py")
            raise SystemExit(1)


def run_simulation():
    validate_environment()
    system = IntegratedSystem()
    system.run()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dashboard", action="store_true")
    args = parser.parse_args()
    if args.dashboard:
        app = create_app()
        app.run(debug=True)
    else:
        run_simulation()


if __name__ == "__main__":
    main()
