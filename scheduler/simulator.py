"""
simulator.py

Time-stepped simulation of GPU KV-cache memory usage under a given
scheduling policy. Lets you test policy logic entirely on CPU/Mac
before wiring it into real vLLM on the GPU machine.

Simplification: each active request generates one token per time step
(a reasonable abstraction for comparing scheduling POLICIES -- the actual
per-token wall-clock time is a vLLM/hardware concern, not a policy concern).
"""

from dataclasses import dataclass, field
from policy import SchedulingPolicy, Action, SimState


@dataclass
class RequestState:
    request_id: int
    prompt_len: int
    output_len: int
    arrival_time: int
    tokens_generated: int = 0
    admitted_time: int = None
    completed_time: int = None
    status: str = "pending"   # pending -> active -> completed / rejected / oom_dropped
    wait_steps: int = 0


@dataclass
class SimulationResult:
    completed: list = field(default_factory=list)
    rejected: list = field(default_factory=list)
    oom_dropped: list = field(default_factory=list)
    memory_trace: list = field(default_factory=list)   # (time, used_bytes) per step
    peak_utilization: float = 0.0

    def summary(self, capacity_bytes: float):
        total = len(self.completed) + len(self.rejected) + len(self.oom_dropped)
        latencies = [r.completed_time - r.arrival_time for r in self.completed]
        avg_latency = sum(latencies) / len(latencies) if latencies else float("nan")
        p95_latency = sorted(latencies)[int(0.95 * len(latencies))] if latencies else float("nan")
        return {
            "total_requests": total,
            "completed": len(self.completed),
            "rejected": len(self.rejected),
            "oom_dropped": len(self.oom_dropped),
            "oom_rate_pct": 100 * len(self.oom_dropped) / total if total else 0,
            "avg_latency_steps": round(avg_latency, 2),
            "p95_latency_steps": round(p95_latency, 2) if latencies else float("nan"),
            "peak_utilization_pct": round(100 * self.peak_utilization, 1),
            "throughput_tokens_completed": sum(r.output_len for r in self.completed),
        }


def run_simulation(requests, policy: SchedulingPolicy, memory_model, max_steps: int = None,
                    hard_oom_at_capacity: bool = True):
    """
    Runs the trace through the given policy.

    hard_oom_at_capacity: if True, if usage ever exceeds 100% capacity
    (shouldn't happen under a correct policy, but WILL happen for the
    no-admission-control baseline), the oldest active requests are force-dropped
    to simulate a real OOM crash recovery, and counted as oom_dropped.
    """
    requests_by_arrival = sorted(requests, key=lambda r: r["arrival_time"])
    pending = [RequestState(**r) for r in requests_by_arrival]
    active = []
    result = SimulationResult()

    t = 0
    max_steps = max_steps or (max(r["arrival_time"] for r in requests_by_arrival) + 3000)

    while (pending or active) and t <= max_steps:
        # 1. Admit new arrivals / retry delayed requests
        still_pending = []
        for req in pending:
            if req.arrival_time > t:
                still_pending.append(req)
                continue

            used_bytes = sum(
                memory_model.current_footprint(r.prompt_len, r.tokens_generated) for r in active
            )
            sim_state = SimState(
                current_used_bytes=used_bytes,
                capacity_bytes=memory_model.total_capacity_bytes,
                queue_depth=len(pending),
                current_time=t,
            )
            action = policy.decide(req.prompt_len, req.output_len, sim_state, wait_steps=req.wait_steps)

            if action == Action.ACCEPT:
                req.admitted_time = t
                req.status = "active"
                active.append(req)
            elif action == Action.DELAY:
                req.wait_steps += 1
                still_pending.append(req)
            else:  # REJECT
                req.status = "rejected"
                result.rejected.append(req)
        pending = still_pending

        # 2. Advance active requests by one token each
        still_active = []
        for req in active:
            req.tokens_generated += 1
            if req.tokens_generated >= req.output_len:
                req.status = "completed"
                req.completed_time = t
                result.completed.append(req)
            else:
                still_active.append(req)
        active = still_active

        # 3. Compute memory usage after this step, check for OOM
        used_bytes = sum(memory_model.current_footprint(r.prompt_len, r.tokens_generated) for r in active)
        utilization = memory_model.utilization(used_bytes)
        result.peak_utilization = max(result.peak_utilization, utilization)

        if hard_oom_at_capacity and utilization > 1.0:
            # Simulate an OOM crash: force-drop the most recently admitted requests
            # until we're back under capacity (crude but standard approximation)
            active.sort(key=lambda r: r.admitted_time)  # oldest first, drop newest last-in
            while active and memory_model.utilization(
                sum(memory_model.current_footprint(r.prompt_len, r.tokens_generated) for r in active)
            ) > 1.0:
                dropped = active.pop()  # drop the newest (most recently admitted)
                dropped.status = "oom_dropped"
                result.oom_dropped.append(dropped)
            used_bytes = sum(memory_model.current_footprint(r.prompt_len, r.tokens_generated) for r in active)

        result.memory_trace.append((t, used_bytes))
        t += 1

    return result
