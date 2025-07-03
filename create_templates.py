from pathlib import Path

TEMPLATES = {
    "population_instruction.txt": "Generate {n} personas experiencing hearing loss as JSON array.",
    "research_wizard_prompt.txt": "You are a research interviewer studying hearing loss.",
    "judge_prompt.txt": "Evaluate the conversation for quality and success.",
    "self_improve_prompt.txt": "Given feedback, improve the wizard prompt." ,
}

for name, content in TEMPLATES.items():
    path = Path("templates") / name
    if not path.exists():
        path.write_text(content)
        print(f"Created {path}")
