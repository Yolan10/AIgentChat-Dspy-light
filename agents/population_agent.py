from dataclasses import dataclass
from typing import Dict
from langchain_openai import ChatOpenAI
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
        if getattr(resp, "usage_metadata", None):
            tracker.add_usage(resp.usage_metadata.input_tokens,
                             resp.usage_metadata.output_tokens)
        return resp.content

    def respond_to(self, message: str) -> str:
        resp = self.llm.invoke([
            SystemMessage(content=self.system_instruction),
            HumanMessage(content=message),
        ])
        if getattr(resp, "usage_metadata", None):
            tracker.add_usage(resp.usage_metadata.input_tokens,
                             resp.usage_metadata.output_tokens)
        return resp.content
