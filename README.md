# Basilisk

Industrial grade LLM training.  
Based on other libraries out there, in both the pytorch and jax ecosystems (torchtitan, megatron-lm, olmo-core, deepspeed, marin, flax, equinox).


### Pending

- Dataloader
- Optimizers (Muon etc.)
- CLI/config-based launch
- GPU support
- DP, FSDP/Zero-3, TP/SP, CP, PP
- MoE (router, dispatch/combine), EP
- Checkpointing
- Activation checkpointing, gradient accumulation
- Mixed precision training
- Long context (YaRN etc)
- Inference-specific optimizations (KV cache)


### Tests

```bash
uv run pytest tests/test_attention.py
```

