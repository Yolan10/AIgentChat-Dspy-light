from pathlib import Path
from collections import deque
from typing import List, Dict, Iterable
import dspy
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

import config
from core.token_tracker import tracker
from core.utils import get_usage_tokens
from core.dspy_utils import apply_dspy_optimizer, build_dataset as _build_dataset, get_miprov2
from core.structured_logger import StructuredLogger
from core.console_logger import ConsoleLogger
import json

IMPROVED_PROMPTS_LOG = Path("logs/improved_prompts.log")


def build_dataset(logs: Iterable[Dict]) -> List[dspy.Example]:
    """Return a list of :class:`dspy.Example` for test convenience."""
    dataset: List[dspy.Example] = []
    for log in logs:
        score = log.get("score") or log.get("overall") or 0.0
        conversation = " ".join(turn.get("text", "") for turn in log.get("turns", []))
        dataset.append(dspy.Example(conversation=conversation, score=score))
    return dataset

class WizardAgent:
    def __init__(self, wizard_id: str):
        self.wizard_id = wizard_id
        self.llm = ChatOpenAI()
        self.logger = StructuredLogger()
        self.console = ConsoleLogger()
        self.history_buffer = deque(maxlen=config.HISTORY_BUFFER_LIMIT)
        self.current_prompt = Path("templates/research_wizard_prompt.txt").read_text() if Path("templates/research_wizard_prompt.txt").exists() else "You are a helpful researcher."
        self.conversation_count = 0

    def set_run(self, run_no: int):
        self.run_no = run_no

    def converse_with(self, pop_agent) -> Dict:
        log = {"wizard_id": self.wizard_id, "pop_agent_id": pop_agent.agent_id, "turns": []}
        self.logger.log("conversation_start", agent_id=pop_agent.agent_id)
        self.console.log(f"Conversation with {pop_agent.agent_id} started")
        intro = pop_agent.introduce()
        log["turns"].append({"speaker": "pop", "text": intro})

        message_history = [SystemMessage(content=self.current_prompt), HumanMessage(content=intro)]
        for _ in range(config.MAX_TURNS):
            resp = self.llm.invoke(message_history)
            if getattr(resp, "usage_metadata", None):
                prompt_tokens, completion_tokens = get_usage_tokens(resp.usage_metadata)
                tracker.add_usage(prompt_tokens, completion_tokens)
            wizard_reply = resp.content
            log["turns"].append({"speaker": "wizard", "text": wizard_reply})
            pop_resp = pop_agent.respond_to(wizard_reply)
            log["turns"].append({"speaker": "pop", "text": pop_resp})
            self.logger.log(
                "conversation_turn",
                wizard_reply=wizard_reply,
                pop_reply=pop_resp,
            )
            self.console.log(f"W: {wizard_reply[:50]} | A: {pop_resp[:50]}")
            message_history.append(HumanMessage(content=pop_resp))
        self.conversation_count += 1
        self.history_buffer.append(log)
        self.logger.log(
            "conversation_end",
            agent_id=pop_agent.agent_id,
            turns=len(log["turns"]),
        )
        self.console.log(f"Conversation with {pop_agent.agent_id} ended")
        return log

    def add_judge_feedback(self, result: Dict):
        self.logger.log("judge_feedback", **result)
        self.console.log(f"Judge scored {result}")
        if self.history_buffer:
            conv = self.history_buffer[-1]
            conv["score"] = result.get("overall", result.get("score"))
            conv["judge_feedback"] = result

    def _should_self_improve(self) -> bool:
        return self.conversation_count in config.SELF_IMPROVE_AFTER

    def self_improve(self):
        self.logger.log("self_improve", step=self.conversation_count)
        self.console.log(f"Self improve step {self.conversation_count}")
        self.current_prompt = apply_dspy_optimizer(
            self.current_prompt,
            self.history_buffer,
            get_opt=get_miprov2,
        )
        improver_type = type(get_miprov2(lambda *_: 0)).__name__
        avg_score = 0.0
        scores = [c.get("score") or 0 for c in self.history_buffer]
        if scores:
            avg_score = sum(scores) / len(scores)
        entry = {
            "id": f"improved_prompts_wizzard_{self.run_no}.{self.conversation_count}",
            "prompt": self.current_prompt,
            "avg_score": avg_score,
            "improver": improver_type,
        }
        IMPROVED_PROMPTS_LOG.parent.mkdir(parents=True, exist_ok=True)
        with IMPROVED_PROMPTS_LOG.open("a") as f:
            f.write(json.dumps(entry) + "\n")
        self.logger.log("prompt_improved", **entry)

