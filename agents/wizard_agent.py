from pathlib import Path
from collections import deque
from typing import List, Dict
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

import config
from core.token_tracker import tracker
from core.structured_logger import StructuredLogger

class WizardAgent:
    def __init__(self, wizard_id: str):
        self.wizard_id = wizard_id
        self.llm = ChatOpenAI()
        self.logger = StructuredLogger()
        self.history_buffer = deque(maxlen=config.HISTORY_BUFFER_LIMIT)
        self.current_prompt = Path("templates/research_wizard_prompt.txt").read_text() if Path("templates/research_wizard_prompt.txt").exists() else "You are a helpful researcher."
        self.conversation_count = 0

    def set_run(self, run_no: int):
        self.run_no = run_no

    def converse_with(self, pop_agent) -> Dict:
        log = {"wizard_id": self.wizard_id, "pop_agent_id": pop_agent.agent_id, "turns": []}
        intro = pop_agent.introduce()
        log["turns"].append({"speaker": "pop", "text": intro})

        message_history = [SystemMessage(content=self.current_prompt), HumanMessage(content=intro)]
        for _ in range(config.MAX_TURNS):
            resp = self.llm.invoke(message_history)
            tracker.add_usage(resp.usage.prompt_tokens, resp.usage.completion_tokens)
            wizard_reply = resp.content
            log["turns"].append({"speaker": "wizard", "text": wizard_reply})
            pop_resp = pop_agent.respond_to(wizard_reply)
            log["turns"].append({"speaker": "pop", "text": pop_resp})
            message_history.append(HumanMessage(content=pop_resp))
        self.conversation_count += 1
        self.history_buffer.append(log)
        return log

    def add_judge_feedback(self, result: Dict):
        self.logger.log("judge_feedback", **result)

    def _should_self_improve(self) -> bool:
        return self.conversation_count in config.SELF_IMPROVE_AFTER

    def self_improve(self):
        self.logger.log("self_improve", step=self.conversation_count)
