"""Loading helpers for newer center models such as Qwen3.5.

Some newer model releases require a transformers build that references FP8 dtypes not
present in older torch builds. This module keeps that import path safe and provides
one central loader for optional device_map/max_memory handling.

Usage (call BEFORE importing transformers anywhere if you need the FP8 shim):
    from feedback_state.newarch_loader import apply_torch_fp8_shim
    apply_torch_fp8_shim()
"""
from __future__ import annotations

import torch


def dtype_from_name(name: str) -> torch.dtype:
    """Resolve the configured model dtype with bfloat16 as the safe default."""
    return {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }.get(str(name).lower(), torch.bfloat16)


def apply_torch_fp8_shim() -> None:
    """transformers>=5.x references torch.float8_e8m0fnu (added in torch 2.7) at import
    time. On torch 2.6 this raises AttributeError even though we never use FP8. Alias it
    to an existing fp8 dtype so the import succeeds; harmless because no FP8 path runs."""
    if not hasattr(torch, "float8_e8m0fnu"):
        torch.float8_e8m0fnu = torch.float8_e4m3fn  # type: ignore[attr-defined]


def load_central_model(model_name: str, dtype, local_files_only: bool, device_map=None, max_memory=None):
    """Return a CausalLM for `model_name`.

    device_map: passed to from_pretrained for multi-GPU sharding (e.g. "auto" splits a big
    model across visible GPUs). Default None = single-device (caller does .to(device)).
    max_memory: optional per-device memory cap dict, passed to from_pretrained."""
    apply_torch_fp8_shim()
    from transformers import AutoModelForCausalLM

    # Qwen3.5 registers a standard causal-LM class; all supported centers use the
    # same loading path here.
    kw = {"dtype": dtype, "local_files_only": local_files_only}
    if device_map is not None:
        kw["device_map"] = device_map
    if max_memory is not None:
        kw["max_memory"] = max_memory
    return AutoModelForCausalLM.from_pretrained(model_name, **kw)
