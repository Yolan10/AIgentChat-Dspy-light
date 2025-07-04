from datetime import datetime
from typing import Dict
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

from core.structured_logger import StructuredLogger

class GodAgent:
    def __init__(self):
        self.llm = ChatOpenAI()
        self.logger = StructuredLogger()

    def spawn_population_from_spec(self, spec: Dict, run_no: int, idx: int):
        from agents.population_agent import PopulationAgent

        timestamp = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
        agent_id = f"{run_no}.{idx}_{timestamp}"
        persona = spec
        instruction = f"You are {persona.get('name')} with hearing loss experience: {persona.get('experience', '')}."
        agent = PopulationAgent(agent_id=agent_id, system_instruction=instruction, spec=spec)
        self.logger.log(
            "spawned_agent",
            agent_id=agent_id,
            persona=persona.get("name"),
        )
        return agent
