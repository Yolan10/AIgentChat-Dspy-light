from core.dspy_utils import apply_dspy_optimizer

class DummyProgram:
    pass

class _MockBS:
    def __init__(self, metric, max_labeled_demos):
        self.init_args = (metric, max_labeled_demos)
    def compile(self, student, trainset, teacher=None, valset=None):
        student.bs_called = True
        return student

class _MockMipro:
    def __init__(self, metric, max_bootstrapped_demos):
        self.init_args = (metric, max_bootstrapped_demos)
    def compile(self, student, trainset, teacher=None, valset=None, **kw):
        student.mipro_called = True
        return student


def test_apply_dspy_optimizer(monkeypatch):
    monkeypatch.setattr('core.dspy_utils.BootstrapFewShot', _MockBS)
    monkeypatch.setattr('core.dspy_utils.MIPROv2', _MockMipro)
    prog = DummyProgram()
    optimized = apply_dspy_optimizer(prog, metric=lambda x,y:0, trainset=[1])
    assert optimized is prog
    assert getattr(optimized, 'bs_called', False)
    assert getattr(optimized, 'mipro_called', False)
