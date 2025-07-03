from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from core.structured_logger import StructuredLogger
from core.token_tracker import tracker

class EnhancedJudgeAgent:
    def __init__(self, judge_id: str, improvement_interval: int = 20):
        self.judge_id = judge_id
        self.llm = ChatOpenAI(temperature=0.2)
        self.logger = StructuredLogger()
        self.improvement_interval = improvement_interval

    def judge(self, conversation_log: dict) -> dict:
        prompt = f"Evaluate the following conversation: {conversation_log}"
        resp = self.llm.invoke([HumanMessage(content=prompt)])
        tracker.add_usage(resp.usage.prompt_tokens, resp.usage.completion_tokens)
        result = {"overall": 0.8, "success": True}
        self.logger.log("judged", result=result)
        return result
