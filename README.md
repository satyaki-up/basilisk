# Basilisk

Industrial grade LLM training.


### Pending

- Trainer
- Optimizers (Muon etc.)
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

