"""Per-token additive attention bias inside vLLM (V1 engine, TRITON_ATTN_VLLM_V1 backend) -- the memory's tilt at vLLM speed.

vLLM's stock attention kernels take no per-token bias.  The Triton backend's two kernels are Python source, so this
module rewrites them at install time (a few inserted lines, anchored on the installed source, nothing else changed):

  * decode  : vllm.attention.ops.chunked_prefill_paged_decode.kernel_paged_attention_2d
  * prefill : vllm.attention.ops.prefix_prefill._fwd_kernel   (cached context blocks + the chunk's own tokens)

Each KV slot gets one float32 "bias" next to the KV cache (``_BIAS_CACHE[num_blocks * block_size]``), written when the
token's K/V are written (so chunked prefill and prefix caching just work: a cached block keeps the bias it was
computed with); the kernels add bias[j] to the scaled score of every query on key j.  Requests get their bias
through an in-process registry keyed by the prompt token ids -- the evaluator runs the engine in-process
(VLLM_ENABLE_V1_MULTIPROCESSING=0).

  install()                                   before the LLM is built (sets VLLM_ATTENTION_BACKEND)
  register(prompt_token_ids, bias_over_prompt) float32, one value per prompt token (generated tokens get 0)
  clear()

Requires enforce_eager (the evaluator uses it); kv_cache_dtype auto; no speculative decoding.
"""
from __future__ import annotations

import hashlib
import importlib.util
import inspect
import os
import sys
from pathlib import Path

import numpy as np
import torch

_REGISTRY: dict[tuple[int, ...], np.ndarray] = {}
_BIAS_CACHE: torch.Tensor | None = None
_KERNELS = None
_INSTALLED = False
_MISSING = object()


# ----------------------------------------------------------------------------------------------- registry
def register(prompt_token_ids, bias) -> None:
    b = np.ascontiguousarray(np.asarray(bias, dtype=np.float32))
    ids = tuple(int(t) for t in prompt_token_ids)
    assert len(ids) == len(b), f"bias has {len(b)} entries for {len(ids)} prompt tokens"
    if b.any():
        _REGISTRY[ids] = b


def clear() -> None:
    _REGISTRY.clear()


def lookup(prompt_token_ids) -> np.ndarray | None:
    if not _REGISTRY:
        return None
    return _REGISTRY.get(tuple(int(t) for t in prompt_token_ids))


# ----------------------------------------------------------------------------------------------- kernels
def _replace_once(src: str, anchor: str, new: str, count: int = 1) -> str:
    n = src.count(anchor)
    if n != count:
        raise RuntimeError(f"kernel source anchor found {n} times (expected {count}): {anchor!r}")
    return src.replace(anchor, new)


def _patched_kernel_source() -> str:
    import vllm.attention.ops.chunked_prefill_paged_decode as cpd
    import vllm.attention.ops.prefix_prefill as pp

    dec = inspect.getsource(cpd.kernel_paged_attention_2d.fn)
    dec = _replace_once(dec, "def kernel_paged_attention_2d(", "def kernel_paged_attention_2d_bias(")
    dec = _replace_once(dec, "        alibi_slopes_ptr,  # [num_query_heads]\n",
                        "        alibi_slopes_ptr,  # [num_query_heads]\n        bias_ptr,  # [num_blks * blk_size] float32: additive score per KV slot\n")
    dec = _replace_once(dec, "        USE_ALIBI_SLOPES: tl.constexpr,  # bool\n",
                        "        USE_ALIBI_SLOPES: tl.constexpr,  # bool\n        USE_BIAS: tl.constexpr,  # bool\n")
    dec = _replace_once(dec, "        S += scale * tl.dot(Q, K)\n",
                        "        S += scale * tl.dot(Q, K)\n"
                        "        if USE_BIAS:\n"
                        "            S += tl.load(bias_ptr + physical_block_idx * BLOCK_SIZE + offs_n)[None, :]\n")

    pre = inspect.getsource(pp._fwd_kernel.fn)
    pre = _replace_once(pre, "def _fwd_kernel(", "def _fwd_kernel_bias(")
    pre = _replace_once(pre, "                B_Seqlen,\n", "                B_Seqlen,\n                bias_cache_ptr,\n                token_bias_ptr,\n")
    pre = _replace_once(pre, "                SKIP_DECODE: tl.constexpr,\n", "                SKIP_DECODE: tl.constexpr,\n                USE_BIAS: tl.constexpr,\n")
    parts = pre.split("        qk *= sm_scale\n")
    if len(parts) != 3:
        raise RuntimeError(f"expected two score-scaling lines in the prefill kernel, found {len(parts) - 1}")
    ctx_bias = ("        qk *= sm_scale\n"
                "        if USE_BIAS:\n"
                "            qk += tl.load(bias_cache_ptr + bn * BLOCK_SIZE +\n"
                "                          ((start_n + offs_bs_n) % BLOCK_SIZE),\n"
                "                          mask=(start_n + offs_bs_n) < cur_batch_ctx_len,\n"
                "                          other=0.0)[None, :]\n")
    req_bias = ("        qk *= sm_scale\n"
                "        if USE_BIAS:\n"
                "            qk += tl.load(token_bias_ptr + cur_batch_in_all_start_index +\n"
                "                          start_n + offs_n,\n"
                "                          mask=(start_n + offs_n) < cur_batch_query_len,\n"
                "                          other=0.0)[None, :]\n")
    pre = parts[0] + ctx_bias + parts[1] + req_bias + parts[2]
    header = ("# generated by feedback_state.vllm_attn_bias from the installed vLLM kernels; do not edit\n"
              "import triton\nimport triton.language as tl\n"
              "from vllm.attention.ops.chunked_prefill_paged_decode import cdiv_fn\n\n\n")
    return header + dec + "\n\n" + pre


def _load_kernels():
    """Write the patched kernels to a cache file (Triton needs real source) and import them."""
    global _KERNELS
    if _KERNELS is not None:
        return _KERNELS
    import vllm
    src = _patched_kernel_source()
    tag = hashlib.sha1((vllm.__version__ + src).encode()).hexdigest()[:12]
    cache = Path(os.environ.get("KALMAN_KERNEL_CACHE", Path.home() / ".cache" / "kalman_vllm_bias"))
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"kernels_{tag}.py"
    if not path.exists():
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(src)
        os.replace(tmp, path)
    name = f"kalman_vllm_bias_kernels_{tag}"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    _KERNELS = mod
    return mod


# ----------------------------------------------------------------------------------------------- launchers
def _prefill(q, k, v, o, k_cache, v_cache, b_loc, b_start_loc, b_seq_len, max_input_len, k_scale, v_scale, sm_scale,
             bias_cache, token_bias, use_bias):
    import triton

    K = _load_kernels()
    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
    assert Lq == Lk == Lv
    batch, head = b_seq_len.shape[0], q.shape[1]
    grid = lambda META: (batch, head, triton.cdiv(max_input_len, META["BLOCK_M"]))   # noqa: E731
    K._fwd_kernel_bias[grid](
        q, k, v, k_cache, v_cache, b_loc, sm_scale, k_scale, v_scale, b_start_loc, b_seq_len,
        bias_cache, token_bias,
        k_cache.shape[4], o,
        b_loc.stride(0), b_loc.stride(1),
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        o.stride(0), o.stride(1), o.stride(2),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), k_cache.stride(3), k_cache.stride(4),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), v_cache.stride(3),
        BLOCK_SIZE=v_cache.shape[3],
        num_queries_per_kv=q.shape[1] // k.shape[1],
        IN_PRECISION=None,
        BLOCK_DMODEL=Lk,
        BLOCK_DMODEL_PADDED=triton.next_power_of_2(Lk),
        SLIDING_WINDOW=0,
        SKIP_DECODE=True,
        USE_BIAS=use_bias,
        BLOCK_M=128,
        BLOCK_N=64,
        num_unroll_cache=4,
        num_unroll_request=1,
        num_warps=4,
        num_stages=1,
    )


def _decode(query, key, output, key_cache, value_cache, block_table, query_start_loc, seq_lens, k_scale, v_scale, sm_scale,
            bias_cache, use_bias):
    import triton

    K = _load_kernels()
    block_size = value_cache.shape[3]
    num_seqs = seq_lens.shape[0]
    num_query_heads, num_kv_heads = query.shape[1], key.shape[1]
    num_queries_per_kv = num_query_heads // num_kv_heads
    head_size = query.shape[2]
    K.kernel_paged_attention_2d_bias[(num_seqs, num_kv_heads)](
        output_ptr=output,
        query_ptr=query,
        key_cache_ptr=key_cache,
        value_cache_ptr=value_cache,
        block_tables_ptr=block_table,
        seq_lens_ptr=seq_lens,
        alibi_slopes_ptr=None,
        bias_ptr=bias_cache,
        scale=sm_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        num_query_heads=num_query_heads,
        num_queries_per_kv=num_queries_per_kv,
        num_queries_per_kv_padded=max(triton.next_power_of_2(num_queries_per_kv), 16),
        block_table_stride=block_table.stride(0),
        query_stride_0=query.stride(0),
        query_stride_1=query.stride(1),
        output_stride_0=output.stride(0),
        output_stride_1=output.stride(1),
        BLOCK_SIZE=block_size,
        HEAD_SIZE=head_size,
        HEAD_SIZE_PADDED=triton.next_power_of_2(head_size),
        USE_ALIBI_SLOPES=False,
        USE_BIAS=use_bias,
        SLIDING_WINDOW=0,
        x=key_cache.shape[4],
        stride_k_cache_0=key_cache.stride(0),
        stride_k_cache_1=key_cache.stride(1),
        stride_k_cache_2=key_cache.stride(2),
        stride_k_cache_3=key_cache.stride(3),
        stride_k_cache_4=key_cache.stride(4),
        stride_v_cache_0=value_cache.stride(0),
        stride_v_cache_1=value_cache.stride(1),
        stride_v_cache_2=value_cache.stride(2),
        stride_v_cache_3=value_cache.stride(3),
        filter_by_query_len=True,
        query_start_len_ptr=query_start_loc,
    )


def _bias_cache(num_blocks: int, block_size: int, device) -> torch.Tensor:
    global _BIAS_CACHE
    n = num_blocks * block_size
    if _BIAS_CACHE is None or _BIAS_CACHE.numel() != n or _BIAS_CACHE.device != device:
        _BIAS_CACHE = torch.zeros(n, dtype=torch.float32, device=device)
    return _BIAS_CACHE


def _attention_forward(self, layer, query, key, value, kv_cache, attn_metadata, output=None):
    """TritonAttentionImpl.forward with the bias cache (same structure as the stock method)."""
    from vllm.attention.ops.paged_attn import PagedAttention

    assert output is not None, "Output tensor must be provided."
    if attn_metadata is None:   # profiling run
        return output
    assert attn_metadata.use_cascade is False
    assert self.alibi_slopes is None and self.sliding_window[0] <= 0 and not self.use_irope, \
        "bias kernels: plain causal attention only (build the LLM with disable_sliding_window=True for windowed families such as Mistral; ALiBi and iRoPE are not covered)"
    assert "fp8" not in self.kv_cache_dtype, "bias kernels: kv_cache_dtype auto only"
    num_actual_tokens = attn_metadata.num_actual_tokens
    key_cache, value_cache = PagedAttention.split_kv_cache(kv_cache, self.num_kv_heads, self.head_size)
    PagedAttention.write_to_paged_cache(key, value, key_cache, value_cache, attn_metadata.slot_mapping, self.kv_cache_dtype,
                                        layer._k_scale, layer._v_scale)
    bias_cache = _bias_cache(key_cache.shape[0], value_cache.shape[3], key.device)
    token_bias = getattr(attn_metadata, "kalman_token_bias", None)
    use_bias = bool(getattr(attn_metadata, "kalman_use_bias", False))
    if token_bias is not None:
        bias_cache[attn_metadata.slot_mapping[:num_actual_tokens]] = token_bias[:num_actual_tokens]
    q, k, v, o = query[:num_actual_tokens], key[:num_actual_tokens], value[:num_actual_tokens], output[:num_actual_tokens]
    if attn_metadata.max_query_len > 1:
        _prefill(q, k, v, o, key_cache, value_cache, attn_metadata.block_table, attn_metadata.query_start_loc, attn_metadata.seq_lens,
                 attn_metadata.max_query_len, layer._k_scale, layer._v_scale, self.scale, bias_cache, token_bias, use_bias)
    _decode(q, k, o, key_cache, value_cache, attn_metadata.block_table, attn_metadata.query_start_loc, attn_metadata.seq_lens,
            layer._k_scale, layer._v_scale, self.scale, bias_cache, use_bias)
    return output


def _wrap_prepare_inputs(orig):
    def _prepare_inputs(self, scheduler_output):
        out = orig(self, scheduler_output)
        attn_metadata = out[0]
        num_reqs = self.input_batch.num_reqs
        total = scheduler_output.total_num_scheduled_tokens
        req_ids = self.input_batch.req_ids[:num_reqs]
        counts = np.array([scheduler_output.num_scheduled_tokens[r] for r in req_ids], dtype=np.int64)
        starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
        bias = np.zeros(total, dtype=np.float32)
        use = False
        for i, rid in enumerate(req_ids):
            st = self.requests[rid]
            vec = getattr(st, "kalman_attn_bias", _MISSING)
            if vec is _MISSING:
                vec = lookup(st.prompt_token_ids)
                st.kalman_attn_bias = vec
            if vec is None:
                continue
            use = True
            pos = self.positions_np[starts[i]: starts[i] + counts[i]]
            inside = pos < len(vec)
            seg = bias[starts[i]: starts[i] + counts[i]]
            seg[inside] = vec[pos[inside]]
        attn_metadata.kalman_use_bias = use
        attn_metadata.kalman_token_bias = torch.from_numpy(bias).to(self.device, non_blocking=True) if (use or _REGISTRY or _BIAS_CACHE is not None) else None
        return out
    return _prepare_inputs


def _wrap_hash_request_tokens(orig, hash_block_tokens):
    """Prefix caching must not share KV blocks across different tilts: the K/V of every token after a tilted block depend
    on the bias, so each block's hash also covers the bias over the prompt up to that block's end (zero prefix -> plain hash,
    shared by every request; the 8 samples of one prompt share everything)."""

    def hash_request_tokens(hash_function, block_size, request):
        hashes = orig(hash_function, block_size, request)
        vec = lookup(getattr(request, "prompt_token_ids", ()))
        if vec is None or not hashes:
            return hashes
        out, parent = [], None
        for k, h in enumerate(hashes):
            prefix = vec[: (k + 1) * block_size]
            extra = h.extra_keys
            if prefix.any():
                extra = tuple(extra or ()) + ("kalman_bias", hashlib.sha1(prefix.tobytes()).hexdigest())
            nh = hash_block_tokens(hash_function, parent, list(h.token_ids), extra)
            out.append(nh)
            parent = nh.hash_value
        return out

    return hash_request_tokens


def install() -> None:
    """Route attention through the bias kernels for every LLM built afterwards in this process."""
    global _INSTALLED
    os.environ["VLLM_ATTENTION_BACKEND"] = "TRITON_ATTN_VLLM_V1"
    os.environ.setdefault("VLLM_USE_V1", "1")
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    if _INSTALLED:
        return
    from vllm.v1.attention.backends import triton_attn
    from vllm.v1.core import kv_cache_manager, kv_cache_utils
    from vllm.v1.worker import gpu_model_runner

    _load_kernels()
    triton_attn.TritonAttentionImpl.forward = _attention_forward
    gpu_model_runner.GPUModelRunner._prepare_inputs = _wrap_prepare_inputs(gpu_model_runner.GPUModelRunner._prepare_inputs)
    kv_cache_manager.hash_request_tokens = _wrap_hash_request_tokens(kv_cache_manager.hash_request_tokens, kv_cache_utils.hash_block_tokens)
    _INSTALLED = True
