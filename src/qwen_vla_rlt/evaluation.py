from __future__ import annotations

from dataclasses import dataclass

from .schemas import BrainDecision, CompletionStatus, Observation
from .task_graph import TaskGraph


@dataclass(frozen=True, slots=True)
class DecisionMetrics:
    samples: int
    stage_accuracy: float
    completion_accuracy: float
    legal_task_selection_rate: float


@dataclass(frozen=True, slots=True)
class EvaluationContext:
    observation: Observation
    current_key: str
    current_complete: bool = False


def evaluate_decisions(
    predictions: list[BrainDecision],
    targets: list[BrainDecision],
    graph: TaskGraph,
    contexts: list[EvaluationContext],
) -> DecisionMetrics:
    """Legality includes observed preconditions and current task state, not just vocabulary."""
    if not predictions or len(predictions) != len(targets) or len(contexts) != len(targets):
        raise ValueError("predictions, targets and contexts must share a non-zero length")
    stage_correct = completion_correct = legal = 0
    for prediction, target, context in zip(predictions, targets, contexts, strict=True):
        identity = (prediction.session_id, prediction.observation_id)
        if identity != (target.session_id, target.observation_id):
            raise ValueError("prediction and target identities differ")
        stage_correct += (prediction.task_id, prediction.stage_id) == (
            target.task_id,
            target.stage_id,
        )
        completion_correct += prediction.completion_status == target.completion_status
        try:
            node = graph.node_for(prediction)
            if identity != (context.observation.session_id, context.observation.observation_id):
                raise ValueError("evaluation observation mismatch")
            graph.validate_transition(context.current_key, node.key)
            if node.key != context.current_key and not context.current_complete:
                raise ValueError("current stage is incomplete")
            graph.validate_facts(node, context.observation)
            if prediction.next_task_id is not None:
                if prediction.completion_status is not CompletionStatus.COMPLETE:
                    raise ValueError("next task cannot start before completion")
                candidates = [
                    graph.nodes[key]
                    for key in node.allowed_next
                    if graph.nodes[key].task_id == prediction.next_task_id
                ]
                executable = False
                for candidate in candidates:
                    try:
                        graph.validate_facts(candidate, context.observation)
                        executable = True
                    except ValueError:
                        pass
                if not executable:
                    raise ValueError("next task preconditions are unmet")
        except ValueError:
            pass
        else:
            legal += 1
    count = len(predictions)
    return DecisionMetrics(count, stage_correct / count, completion_correct / count, legal / count)
