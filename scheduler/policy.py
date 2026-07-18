"""
policy.py

Scheduling policies for LLM inference request admission.

Every policy implements decide(request, sim_state) -> Action, so the
simulator (and later, the real vLLM integration) can swap policies
without changing anything else.
"""

from dataclasses import dataclass
from enum import Enum
from memory_model import MemoryModel


class Action(Enum):
    ACCEPT = "accept"
    DELAY = "delay"
    REJECT = "reject"


@dataclass
class SimState:
    """Snapshot of scheduler-relevant state at decision time."""
    current_used_bytes: float
    capacity_bytes: float
    queue_depth: int
    current_time: int


class SchedulingPolicy:
    """Base interface. Subclass this for any new policy."""

    name = "base"

    def decide(self, prompt_len: int, estimated_output_len: int, sim_state: SimState,
               wait_steps: int = 0) -> Action:
        raise NotImplementedError


class NoOpBaselinePolicy(SchedulingPolicy):
    """
    Mirrors default vLLM continuous batching behavior: accept every request
    immediately, with no memory-awareness. This is what we're benchmarking against.
    """

    name = "baseline_no_admission_control"

    def __init__(self, memory_model: MemoryModel):
        self.memory_model = memory_model

    def decide(self, prompt_len: int, estimated_output_len: int, sim_state: SimState,
               wait_steps: int = 0) -> Action:
        return Action.ACCEPT


class MemoryAwareAdmissionPolicy(SchedulingPolicy):
    """
    Our contribution: before admitting a request, estimate its worst-case
    memory footprint and check whether accepting it would push total usage
    past a safe threshold. If so, delay (retry later) rather than accept
    and risk an OOM failure. Reject outright if a request has already been
    waiting too long (avoids unbounded queue growth / starvation).
    """

    name = "memory_aware_admission_control"

    def __init__(
        self,
        memory_model: MemoryModel,
        safe_threshold: float = 0.85,   # don't admit past 85% of KV-cache pool
        max_wait_steps: int = 50,       # reject if delayed longer than this
    ):
        self.memory_model = memory_model
        self.safe_threshold = safe_threshold
        self.max_wait_steps = max_wait_steps

    def decide(self, prompt_len: int, estimated_output_len: int, sim_state: SimState,
               wait_steps: int = 0) -> Action:
        projected_request_bytes = self.memory_model.estimate_request_footprint(
            prompt_len, estimated_output_len
        )
        projected_total = sim_state.current_used_bytes + projected_request_bytes
        projected_utilization = projected_total / sim_state.capacity_bytes if sim_state.capacity_bytes else 1.0

        if projected_utilization <= self.safe_threshold:
            return Action.ACCEPT

        if wait_steps >= self.max_wait_steps:
            return Action.REJECT

        return Action.DELAY
