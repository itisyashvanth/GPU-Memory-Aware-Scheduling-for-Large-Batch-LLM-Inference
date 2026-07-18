# Environment Setup

## GPU machine (5080) — vLLM setup

```bash
python3 -m venv vllm-env
source vllm-env/bin/activate
pip install vllm
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Verify serving works:
```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct --max-model-len 4096
```

Test with:
```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen2.5-1.5B-Instruct", "prompt": "Explain photosynthesis in one sentence.", "max_tokens": 50}'
```

If vLLM fails to install or the GPU isn't detected, capture the exact error message —
the 5080 is very new hardware (Blackwell architecture) and may need a specific
PyTorch nightly build or CUDA version pin. Don't debug this alone for more than an
hour before flagging it — this is the one dependency that blocks everything else.

## Non-GPU machine (Mac) — dev environment

```bash
python3 -m venv sim-env
source sim-env/bin/activate
pip install numpy pandas matplotlib scipy
```

This is enough for: trace generation, policy simulation logic (before it's wired
into real vLLM), and results analysis/plotting. No GPU/CUDA needed for this half
of the work.
