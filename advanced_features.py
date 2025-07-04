from __future__ import annotations
import json
from pathlib import Path
from typing import List
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from core.structured_logger import StructuredLogger
from core.console_logger import ConsoleLogger
from core.utils import extract_json_array

TEMPLATE_PATH = Path("templates/population_instruction.txt")

class PopulationGenerator:
    def __init__(self):
        self.logger = StructuredLogger()
        self.console = ConsoleLogger()

    def generate(self, instruction: str, n: int) -> List[dict]:
        self.logger.log("population_generation_start", instruction=instruction, count=n)
        self.console.log(f"Generating {n} personas")
        if not TEMPLATE_PATH.exists():
            self.logger.log("population_template_missing")
            self.console.log("population template missing", level="error")
            return self._fallback_personas(n)
        template = TEMPLATE_PATH.read_text()
        prompt = template.format(instruction=instruction, n=n)
        llm = ChatOpenAI()
        try:
            resp = llm.invoke([HumanMessage(content=prompt)])
            candidates = extract_json_array(resp.content)
            if not candidates:
                raise ValueError("parse failed")
            self.logger.log("population_generated", generated=len(candidates))
            self.console.log(f"Generated {len(candidates)} personas")
            return candidates
        except Exception as e:
            self.logger.log("population_generate_error", level="error", error=str(e))
            self.console.log(f"Generation error: {e}", level="error")
            return self._fallback_personas(n)

    def _fallback_personas(self, n: int) -> List[dict]:
        self.logger.log("population_fallback", count=n)
        self.console.log(f"Fallback to {n} simple personas")
        base = {"name": "Person", "age": 30, "experience": "general"}
        return [dict(base, name=f"Person{i}") for i in range(1, n + 1)]


