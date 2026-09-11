import numpy as np

from qwen_vla_rlt.rlt.online import EncodedObservation, ExecutionFeedback, OnlineRLTLoop
from qwen_vla_rlt.rlt.replay import ReplayBuffer, ReplaySource


def test_online_loop_records_executed_intervention_and_next_reference():
    class Environment:
        def execute(self, proposed_action):
            assert np.all(proposed_action == 1)
            return ExecutionFeedback(2, np.full((2, 2), 0.5), np.array([1, 0]), False, 1, True)

    class Trainer:
        def train_step(self, batch):
            assert batch[0].source is ReplaySource.HUMAN_INTERVENTION
            assert np.all(batch[0].action == 0.5)
            assert np.all(batch[0].reference == 1)
            assert np.all(batch[0].next_reference == 2)
            assert batch[0].executed_steps == 1
            return "updated"

    def encode(obs):
        return EncodedObservation(np.zeros(2), np.zeros(1), np.full((2, 2), obs))

    loop = OnlineRLTLoop(
        Environment(),
        encode,
        lambda _: None,
        ReplayBuffer(4),
        Trainer(),
        batch_size=1,
        warmup_chunks=1,
    )
    _, result = loop.step(1)
    assert result == "updated"
