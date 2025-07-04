import threading
from concurrent.futures import ThreadPoolExecutor
from typing import List

import config
from core.structured_logger import StructuredLogger
from core.population_generator import PopulationGenerator
from agents.god_agent import GodAgent
from agents.wizard_agent import WizardAgent
from agents.judge_agent import EnhancedJudgeAgent
from core.token_tracker import tracker
from core.utils import ensure_logs_dir, increment_run_number

class IntegratedSystem:
    def __init__(self):
        self.logger = StructuredLogger()
        self.generator = PopulationGenerator()
        self.god = GodAgent()
        self.wizard = WizardAgent(wizard_id="Wizard_001")
        self.primary_judge = EnhancedJudgeAgent(judge_id="Judge_001")
        self.judge_executor = ThreadPoolExecutor(max_workers=3)
        self.judgment_queue = []
        self.completed_judgments = []
        self.lock = threading.Lock()
        self.judgment_processor = threading.Thread(target=self._process_judgments, daemon=True)
        self.judgment_processor.start()

    def _process_judgments(self):
        while True:
            if self.judgment_queue:
                conv_log = self.judgment_queue.pop(0)
                result = self.primary_judge.judge(conv_log)
                with self.lock:
                    self.completed_judgments.append((conv_log, result))
                self.wizard.add_judge_feedback(result)
            else:
                threading.Event().wait(0.1)

    def _submit_for_judgment(self, log):
        self.judgment_queue.append(log)

    def _wait_for_pending_judgments(self):
        while self.judgment_queue:
            threading.Event().wait(0.1)

    def run(self):
        run_no = increment_run_number()
        self.wizard.set_run(run_no)
        tracker.set_run(run_no)
        personas = self.generator.generate("hearing loss personas", config.POPULATION_SIZE)
        agents = [self.god.spawn_population_from_spec(spec, run_no, i+1) for i, spec in enumerate(personas)]
        for idx, agent in enumerate(agents, start=1):
            log = self.wizard.converse_with(agent)
            self._submit_for_judgment(log)
            if self.wizard._should_self_improve():
                self._wait_for_pending_judgments()
                self.wizard.self_improve()
        self._wait_for_pending_judgments()
