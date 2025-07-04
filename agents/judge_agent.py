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
        self.logger.log("judge_start", conversation=conversation_log.get("pop_agent_id"))
        self.console.log(f"Judging {conversation_log.get('pop_agent_id')}")
        prompt = f"Evaluate the following conversation: {conversation_log}"
        resp = self.llm.invoke([HumanMessage(content=prompt)])
        if getattr(resp, "usage_metadata", None):
            prompt_tokens, completion_tokens = get_usage_tokens(resp.usage_metadata)
            tracker.add_usage(prompt_tokens, completion_tokens)
        result = {"overall": 0.8, "success": True}
        self.logger.log("judged", result=result)
        self.console.log(f"Judge result {result}")
        return result

