"""
generator.py

Synthesizes a realistic-ish LLM inference request trace:
- arrivals follow a Poisson process (standard assumption in serving literature)
- prompt lengths and output lengths follow lognormal distributions
  (heavy-tailed -- most requests short, some much longer, matching real
  chat/assistant traffic patterns)

Swap this out later for a real trace (e.g. Azure LLM Inference Trace) once
you want to validate against production-realistic patterns.
"""

import csv
import numpy as np


def generate_trace(
    num_requests: int = 500,
    arrival_rate_per_step: float = 2.0,   # avg requests arriving per time step
    prompt_len_mean: float = 4.5,         # lognormal params (in log-space)
    prompt_len_sigma: float = 0.6,
    output_len_mean: float = 4.8,
    output_len_sigma: float = 0.7,
    seed: int = 42,
):
    rng = np.random.default_rng(seed)

    # Poisson inter-arrival times -> cumulative arrival times
    inter_arrival_times = rng.exponential(scale=1.0 / arrival_rate_per_step, size=num_requests)
    arrival_times = np.cumsum(inter_arrival_times).astype(int)

    prompt_lens = rng.lognormal(mean=prompt_len_mean, sigma=prompt_len_sigma, size=num_requests).astype(int)
    output_lens = rng.lognormal(mean=output_len_mean, sigma=output_len_sigma, size=num_requests).astype(int)

    # clip to sane bounds
    prompt_lens = np.clip(prompt_lens, 8, 4000)
    output_lens = np.clip(output_lens, 4, 2000)

    requests = []
    for i in range(num_requests):
        requests.append({
            "request_id": i,
            "arrival_time": int(arrival_times[i]),
            "prompt_len": int(prompt_lens[i]),
            "output_len": int(output_lens[i]),
        })
    return requests


def save_trace_csv(requests, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["request_id", "arrival_time", "prompt_len", "output_len"])
        writer.writeheader()
        writer.writerows(requests)


def load_trace_csv(path):
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        return [
            {
                "request_id": int(row["request_id"]),
                "arrival_time": int(row["arrival_time"]),
                "prompt_len": int(row["prompt_len"]),
                "output_len": int(row["output_len"]),
            }
            for row in reader
        ]


if __name__ == "__main__":
    trace = generate_trace()
    save_trace_csv(trace, "sample_trace.csv")
    print(f"Generated {len(trace)} requests, saved to sample_trace.csv")
    print(f"Arrival span: {trace[0]['arrival_time']} to {trace[-1]['arrival_time']} time steps")
    print(f"Avg prompt_len: {sum(r['prompt_len'] for r in trace)/len(trace):.1f}")
    print(f"Avg output_len: {sum(r['output_len'] for r in trace)/len(trace):.1f}")
