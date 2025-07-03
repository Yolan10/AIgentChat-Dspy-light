from dataclasses import dataclass
from typing import Dict
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

from core.token_tracker import tracker

@dataclass
class PopulationAgent:
    agent_id: str
    system_instruction: str
    spec: Dict

    def __post_init__(self):
        self.llm = ChatOpenAI()

    def introduce(self) -> str:
        prompt = [SystemMessage(content=self.system_instruction),
                  HumanMessage(content="Introduce yourself briefly.")]
        resp = self.llm.invoke(prompt)
        tracker.add_usage(resp.usage.prompt_tokens, resp.usage.completion_tokens)
        return resp.content

    def respond_to(self, message: str) -> str:
        resp = self.llm.invoke([SystemMessage(content=self.system_instruction),
                               HumanMessage(content=message)])
        tracker.add_usage(resp.usage.prompt_tokens, resp.usage.completion_tokens)
        return resp.content
