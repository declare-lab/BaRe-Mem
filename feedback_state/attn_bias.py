"""The memory's tilt on the central model's attention (Method 2 of the steering study), shared by every engine.

The record gives each peer i an estimate p_i.  The tilt adds a constant b_i to the attention score of every query
onto every token of peer i's block, in every layer and head:

    softmax(s + b)  ==  Norm(A * c),   c_i = exp(b_i)

so the un-normalised attention on peer i is multiplied by c_i and nothing else changes.  Two forms of b_i:

    logratio   b_i = gamma * log(p_i / max_j p_j)            (the favourite is untouched)
    logodds    b_i = gamma * (z_i - max_j z_j), z = logit p  (the Kalman state's own readout, same ordering)

A flat record (spread <= FLAT_SPREAD) gives no tilt.  The prompt carries no reliability text.

Helpers here map the tilt onto tokens (character spans of the peer blocks -> token positions) and apply it in HF
transformers through a forward pre-hook on the attention modules (eager / sdpa: the additive mask).  vLLM gets it
through feedback_state.vllm_attn_bias (patched Triton kernels, same numbers).
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import torch

FLAT_SPREAD = 0.1
CHAR_LIMIT = 3000
FORMS = ("logratio", "logodds")


def bias_values(probs: Sequence[float], gamma: float, form: str = "logratio") -> list[float]:
    """Per-peer additive score, 0 for the favourite; all zeros for a flat record or gamma 0."""
    probs = [float(p) for p in probs]
    if gamma == 0 or not probs or max(probs) - min(probs) <= FLAT_SPREAD:
        return [0.0] * len(probs)
    if form == "logratio":
        top = max(max(probs), 1e-3)
        return [gamma * math.log(max(p, 1e-3) / top) for p in probs]
    if form == "logodds":
        z = [math.log(min(max(p, 1e-3), 1 - 1e-3) / (1 - min(max(p, 1e-3), 1 - 1e-3))) for p in probs]
        top = max(z)
        return [gamma * (zi - top) for zi in z]
    raise ValueError(f"unknown bias form {form!r}; choose from {FORMS}")


def peer_char_spans(texts: Sequence[str], prompt: str, char_limit: int = CHAR_LIMIT) -> list[tuple[int, int]]:
    """Character span of each peer's text inside a rendered prompt (the block layout of memory_generator.build_messages:
    a header ``Peer i`` [optionally followed by a parenthesised note], a colon, a newline, then the clipped text)."""
    from feedback_state.memory_generator import _clip

    spans, cursor = [], 0
    for i, t in enumerate(texts):
        body = _clip(t, char_limit)
        head = prompt.find(f"Peer {i + 1}", cursor)
        start = -1
        if head >= 0:
            colon = prompt.find(":\n", head)
            if colon >= 0 and prompt.startswith(body, colon + 2):
                start = colon + 2
        if start < 0:
            j = prompt.find(body, cursor) if body.strip() else -1
            start = j
        if start < 0 or not body:
            spans.append((0, 0))
            continue
        spans.append((start, start + len(body)))
        cursor = start + len(body)
    return spans


def token_bias(offsets: Sequence[tuple[int, int]], spans: Sequence[tuple[int, int]], values: Sequence[float]) -> np.ndarray:
    """Per-token bias from character spans and per-span values (a token counts when it lies inside the span)."""
    out = np.zeros(len(offsets), dtype=np.float32)
    if not any(values):
        return out
    starts = np.array([a for a, _ in offsets]); ends = np.array([b for _, b in offsets])
    for (a, e), v in zip(spans, values):
        if e <= a or v == 0:
            continue
        out[(ends > starts) & (starts >= a) & (ends <= e)] = v
    return out


def prompt_token_bias(tokenizer, prompt: str, texts: Sequence[str], probs: Sequence[float], gamma: float, form: str = "logratio",
                      char_limit: int = CHAR_LIMIT) -> tuple[list[int], np.ndarray]:
    """Token ids of a rendered prompt and the tilt over them."""
    enc = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    return ids, token_bias(offs, peer_char_spans(texts, prompt, char_limit), bias_values(probs, gamma, form))


class AttentionSteer:
    """Adds a per-key-position bias to the additive attention mask of the hooked HF attention modules.

    ``bias`` is [B, >= key length] (float); rows follow the batch order of the forward call.  Works for prefill and for
    cached decoding (the key length comes from ``cache_position``), with a static cache (mask spans the cache) and
    with a 2-D / bool / float / absent incoming mask.
    """

    def __init__(self, layers: set[int] | None = None):
        self.layers = layers
        self.bias: torch.Tensor | None = None

    def hook(self, module, args, kwargs):
        if self.bias is None or (self.layers is not None and getattr(module, "_steer_idx", -1) not in self.layers):
            return None
        hs = kwargs.get("hidden_states", args[0] if args else None)
        B, Q = hs.shape[0], hs.shape[1]
        if self.bias.shape[0] != B:
            raise RuntimeError(f"attention bias has {self.bias.shape[0]} rows for a batch of {B}")
        cp = kwargs.get("cache_position")
        K = int(cp[-1]) + 1 if cp is not None else Q
        mask = kwargs.get("attention_mask")
        neg = torch.finfo(hs.dtype).min
        if mask is not None and mask.dim() == 4 and mask.shape[-1] > K:   # static cache: the mask spans the whole cache
            K = mask.shape[-1]
        b = self.bias
        if b.shape[1] < K:
            b = torch.nn.functional.pad(b, (0, K - b.shape[1]))
        add = b[:, :K][:, None, None, :].to(dtype=hs.dtype, device=hs.device)   # [B,1,1,K]
        if mask is None or mask.dim() != 4:
            full = torch.zeros(B, 1, Q, K, dtype=hs.dtype, device=hs.device)
            if Q > 1:
                qi = torch.arange(Q, device=hs.device)[:, None] + (K - Q)
                kj = torch.arange(K, device=hs.device)[None, :]
                full = full.masked_fill((kj > qi)[None, None], neg)
            if mask is not None and mask.dim() == 2:   # padding mask [B, K]
                pad = mask[:, :K] if mask.shape[1] >= K else torch.nn.functional.pad(mask, (0, K - mask.shape[1]), value=1)
                full = full.masked_fill((pad == 0)[:, None, None, :], neg)
            mask = full
        elif mask.dtype == torch.bool:
            mask = torch.where(mask, torch.zeros((), dtype=hs.dtype, device=hs.device), torch.full((), neg, dtype=hs.dtype, device=hs.device))
        else:
            mask = mask.to(hs.dtype)
        if mask.shape[-1] != K:
            mask = mask[..., :K] if mask.shape[-1] > K else torch.nn.functional.pad(mask, (0, K - mask.shape[-1]))
        kwargs["attention_mask"] = mask + add
        return args, kwargs


HF_STEER: AttentionSteer | None = None   # the steer of the last install_hf_hooks in this process (a forward pass sets its .bias)


def install_hf_hooks(model, layers: str | set[int] | None = "all") -> AttentionSteer:
    """Register the steer on every decoder layer's attention module of a HF causal LM; returns the steer (set .bias)."""
    global HF_STEER
    base = model.get_decoder() if hasattr(model, "get_decoder") else model.model
    layer_list = base.layers
    if layers is None or layers == "all":
        chosen = None
    elif isinstance(layers, str):
        a, b = layers.split(":")
        chosen = set(range(int(a), int(b)))
    else:
        chosen = set(layers)
    # one steer per process, shared by every model installed here (models in one process read the same .bias and never
    # run forward at the same time)
    steer = HF_STEER if (HF_STEER is not None and HF_STEER.layers == chosen) else AttentionSteer(chosen)
    hooked = 0
    for idx, layer in enumerate(layer_list):
        # hybrid models (Qwen3.5, Qwen3-Next) interleave linear-attention layers (gated delta net: no softmax over the keys,
        # so no per-key additive score exists) with full softmax attention; the tilt goes on the full-attention layers only
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            continue
        attn._steer_idx = idx
        attn.register_forward_pre_hook(steer.hook, with_kwargs=True)
        hooked += 1
    steer.hooked_layers, steer.total_layers = hooked, len(layer_list)
    print(f"[attn-bias] tilt installed on {hooked} of {len(layer_list)} layers" + (f" (the other {len(layer_list) - hooked} are linear attention)" if hooked < len(layer_list) else ""), flush=True)
    HF_STEER = steer
    return steer
