from datetime import datetime
from typing import Dict
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

from core.structured_logger import StructuredLogger
from core.console_logger import ConsoleLogger

class GodAgent:
    def __init__(self):
        self.llm = ChatOpenAI()
        self.logger = StructuredLogger()
        self.console = ConsoleLogger()

    def spawn_population_from_spec(self, spec: Dict, run_no: int, idx: int):
        """Instantiate a PopulationAgent from a persona specification."""
        from agents.population_agent import PopulationAgent

        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        agent_id = f"{run_no}.{idx}_{timestamp}"

        persona = spec
        name = persona.get("name")
        if not name:
            # Ensure every persona has a name for clearer logs
            self.logger.log(
                "persona_missing_name",
                level="warning",
                agent_id=agent_id,
                spec=persona,
            )
            name = "Unknown"
            persona["name"] = name

        instruction = (
            f"You are {name} with hearing loss experience: "
            f"{persona.get('experience', '')}."
        )

        agent = PopulationAgent(
            agent_id=agent_id, system_instruction=instruction, spec=persona
        )

        # Log the full persona specification for traceability
        self.logger.log("spawned_agent", agent_id=agent_id, persona=persona)
        self.console.log(f"Spawned agent {agent_id}")
        return agent

