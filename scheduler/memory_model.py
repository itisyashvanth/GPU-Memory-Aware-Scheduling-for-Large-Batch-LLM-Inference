"""
memory_model.py

Estimates GPU KV-cache memory footprint for LLM inference requests.
This is deliberately model-config-driven so it reflects real numbers,
not arbitrary constants -- swap in your actual model's config when you
move from simulation to real vLLM integration.
"""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Config for a transformer decoder model, used to compute KV-cache size per token.

    Defaults below approximate a small model (~1.5B, Qwen2.5-1.5B-Instruct-class).
    Update these to match whatever model you actually load in vLLM.
    """
    num_layers: int = 28
    num_kv_heads: int = 2          # GQA: often fewer KV heads than attention heads
    head_dim: int = 128
    dtype_bytes: int = 2           # fp16/bf16 = 2 bytes; fp8 = 1 byte

    @property
    def bytes_per_token(self) -> float:
        """
        KV cache stores one Key vector and one Value vector per layer, per token.
        Size per token = 2 (K and V) * num_layers * num_kv_heads * head_dim * dtype_bytes
        """
        return 2 * self.num_layers * self.num_kv_heads * self.head_dim * self.dtype_bytes


class MemoryModel:
    """Estimates memory footprint of requests given a model config and GPU KV-cache pool size."""

    def __init__(self, model_config: ModelConfig, gpu_kv_cache_pool_bytes: int):
        self.model_config = model_config
        self.total_capacity_bytes = gpu_kv_cache_pool_bytes

    def tokens_to_bytes(self, num_tokens: int) -> float:
        return num_tokens * self.model_config.bytes_per_token

    def estimate_request_footprint(self, prompt_len: int, estimated_output_len: int) -> float:
        """
        Estimate the WORST-CASE memory footprint a request could reach:
        prompt tokens (all present from the start) + full estimated output length.
        This is what an admission-control policy should reason about --
        the request's memory need at its peak, not just right now.
        """
        return self.tokens_to_bytes(prompt_len + estimated_output_len)

    def current_footprint(self, prompt_len: int, tokens_generated_so_far: int) -> float:
        """Memory a request is actually using right now, given how far generation has progressed."""
        return self.tokens_to_bytes(prompt_len + tokens_generated_so_far)

    def utilization(self, used_bytes: float) -> float:
        return used_bytes / self.total_capacity_bytes if self.total_capacity_bytes else 0.0
