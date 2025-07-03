from __future__ import annotations
import json
from pathlib import Path
from typing import List
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from structured_logger import StructuredLogger
from utils import extract_json_array

TEMPLATE_PATH = Path("templates/population_instruction.txt")

class PopulationGenerator:
    def __init__(self):
        self.logger = StructuredLogger()

    def generate(self, instruction: str, n: int) -> List[dict]:
        if not TEMPLATE_PATH.exists():
            return self._fallback_personas(n)
        template = TEMPLATE_PATH.read_text()
        prompt = template.format(instruction=instruction, n=n)
        llm = ChatOpenAI()
        try:
            resp = llm.invoke([HumanMessage(content=prompt)])
            candidates = extract_json_array(resp.content)
            if not candidates:
                raise ValueError("parse failed")
            return candidates
        except Exception as e:
            self.logger.log("population_generate_error", error=str(e))
            return self._fallback_personas(n)

    def _fallback_personas(self, n: int) -> List[dict]:
        base = {"name": "Person", "age": 30, "experience": "general"}
        return [dict(base, name=f"Person{i}") for i in range(1, n + 1)]

