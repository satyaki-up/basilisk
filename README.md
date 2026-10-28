# Basilisk

Industrial grade LLM training.  
Based on other libraries out there, in both the pytorch and jax ecosystems (torchtitan, megatron-lm, olmo-core, deepspeed, marin, flax, equinox).


### Pending

- Configs
- Kernels
- DP, FSDP/Zero-3, TP/SP, CP, PP
- MoE (router, dispatch/combine), EP
- Checkpointing
- Activation checkpointing, gradient accumulation
- Mixed precision training
- Muon
- Long context (YaRN etc)
- Inference-specific optimizations (KV cache)


### Tests

```bash
uv run pytest tests/test_attention.py
```

