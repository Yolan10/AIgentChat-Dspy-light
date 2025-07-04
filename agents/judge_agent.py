from pathlib import Path
import re
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from core.structured_logger import StructuredLogger
from core.console_logger import ConsoleLogger
from core.token_tracker import tracker
from core.utils import get_usage_tokens

class EnhancedJudgeAgent:
    def __init__(self, judge_id: str, improvement_interval: int = 20):
        self.judge_id = judge_id
        self.llm = ChatOpenAI(temperature=0.2)
        self.logger = StructuredLogger()
        self.console = ConsoleLogger()
        self.improvement_interval = improvement_interval

    def judge(self, conversation_log: dict) -> dict:
        """Score a conversation on a 0-1 scale using an LLM prompt."""
        self.logger.log("judge_start", conversation=conversation_log.get("pop_agent_id"))
        self.console.log(f"Judging {conversation_log.get('pop_agent_id')}")

        template_path = Path("templates/judge_prompt.txt")
        if template_path.exists():
            template = template_path.read_text()
        else:
            template = (
                "Evaluate the conversation and return a score between 0 and 1"
            )

        turns = conversation_log.get("turns", [])
        conv_text = "\n".join(f"{t.get('speaker')}: {t.get('text')}" for t in turns)

        messages = [SystemMessage(content=template), HumanMessage(content=conv_text)]
        resp = self.llm.invoke(messages)

        if getattr(resp, "usage_metadata", None):
            prompt_tokens, completion_tokens = get_usage_tokens(resp.usage_metadata)
            tracker.add_usage(prompt_tokens, completion_tokens)

        match = re.search(r"([0-9]*\.?[0-9]+)", resp.content)
        try:
            score = float(match.group(1)) if match else 0.0
        except Exception:
            score = 0.0
        score = max(0.0, min(score, 1.0))

        result = {"overall": score, "success": score >= 0.5}
        self.logger.log("judged", result=result)
        self.console.log(f"Judge result {result}")
        return result

