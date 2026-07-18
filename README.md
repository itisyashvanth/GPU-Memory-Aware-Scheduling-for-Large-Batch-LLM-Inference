# GPU Memory-Aware Scheduling for Large-Batch LLM Inference

Cloud Computing / HPC course project — memory-aware admission control for LLM inference serving.

## Team
- [Your name] — policy design, simulation, K8s scheduler logic, report
- [Friend's name] — vLLM integration, GPU experiments, benchmarking

## Timeline
Mid-July – Second week of October (14 weeks)

## Project Structure
```
scheduler/    -> the admission-control / memory-aware policy code
benchmarks/   -> baseline benchmarking scripts (default vLLM, naive round-robin)
traces/       -> request trace generators + any public traces used
results/      -> raw output data, logs, plots from experiments
report/       -> the written report / paper draft
docs/         -> literature survey tracker, problem statement, meeting notes
```

## Base Paper
Zhang, Y., Yang, N., Pan, C., & Yuan, D. (2026). "Joint Optimization of Resource
Allocation and Request Batching for Multi-Tenant Inference Serving on GPU."
IEEE Transactions on Parallel and Distributed Systems.
https://ieeexplore.ieee.org/document/11223102/

## Setup
See docs/setup.md for environment setup instructions (vLLM install, GPU verification).

## Status
- [x] Week 1: Foundation — repo set up, base papers read, scheduler simulation built
- [ ] Weeks 2-4: Literature survey (~40 papers) + angle locked in
- [ ] Weeks 4-5: Problem formalization
- [ ] Weeks 5-7: Baseline benchmarking (real vLLM, on the 5080)
- [ ] Weeks 7-9: Method design refinement
- [ ] Weeks 9-11: Real implementation (wire policy.py into vLLM)
- [ ] Weeks 11-12: Evaluation on real hardware
- [ ] Weeks 12-14: Write-up + polish

## Scheduler Simulation (working prototype)

`scheduler/` and `benchmarks/run_simulation.py` implement a CPU-only simulation
of the memory-aware admission control policy, so the core logic can be tested
and iterated on without GPU access. Run it with:

```bash
cd benchmarks
python3 run_simulation.py
```

This compares our `MemoryAwareAdmissionPolicy` against a `NoOpBaselinePolicy`
(mirroring default vLLM behavior) on a synthetic bursty trace, and saves
comparison plots to `results/`.

**Sample result from a heavy-load trace (800 requests, 2GB simulated KV-cache pool):**

| Metric | Baseline (no control) | Memory-Aware Policy |
|---|---|---|
| OOM rate | 28.25% | 0.0% |
| Completed requests | 574 | 577 |
| Peak utilization | 103.1% (crashed) | 88.4% (held ceiling) |

This is a simulation result, not yet validated on real vLLM/hardware — that's
the Weeks 5-11 work. But it confirms the policy logic itself is sound before
spending GPU time on it.

### Next step: wiring into real vLLM

`scheduler/policy.py`'s `decide()` interface is designed to be a thin wrapper:
in real vLLM, `sim_state.current_used_bytes` should come from vLLM's actual
KV-cache usage metrics (exposed via its Prometheus metrics endpoint or internal
scheduler state), and `Action.ACCEPT/DELAY/REJECT` should gate whether a
request is submitted to vLLM's request queue at all. This is the friend's
Weeks 9-11 integration task.
