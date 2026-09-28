"""The frozen judge's features of one event: what pipeline.features stores per event, and what an online run computes live.

For an event the judge reads the question alone (q_mean / q_last: mean-pooled and last-token hidden states at three layers),
then one Yes/No judge prompt per answer (feedback_state.judge_prompt); the last-token hidden states of those prompts at the
same layers are peer_hidden, their mean and spread over the answers sem and spread, and the judge's own Yes-No log-odds per
answer the margins. No labels, dataset names or peer identities enter.
"""
from __future__ import annotations

import torch


def selected_layers(count: int) -> list[int]:
    return sorted({max(1, count // 3), max(1, 2 * count // 3), count - 1})


@torch.inference_mode()
def event_features(model, tokenizer, record: dict, texts: list[str], *, num_peers: int, yes_id: int, no_id: int,
                   max_length: int, question_max_length: int, device, layers: list[int] | None = None) -> dict:
    """The features of one event whose answers are `texts` (canonical order, the real ones only), padded to num_peers.

    Returns float32 tensors on `device`: q_mean, q_last [3H]; sem, spread [3H]; margins [num_peers]; peer_hidden
    [num_peers, 3H]; and the layers used (chosen from the model's depth on the first call when not given).
    """
    from feedback_state.judge_prompt import context_text, judge_batch

    real = len(texts)
    question = str(record.get("problem", record.get("question", "")))
    enc = tokenizer(question if question.strip() else " ", return_tensors="pt", truncation=True, max_length=int(question_max_length))
    out = model(**{k: v.to(device) for k, v in enc.items()}, output_hidden_states=True, use_cache=False, return_dict=True)
    if layers is None:
        layers = selected_layers(len(out.hidden_states))
    q_mean = torch.cat([out.hidden_states[l][0].float().mean(0) for l in layers])
    q_last = torch.cat([out.hidden_states[l][0, -1].float() for l in layers])

    input_ids, attention_mask = judge_batch(tokenizer, question, texts, context=context_text(record) or None,
                                            max_length=max_length, device=device)
    out = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, use_cache=False, return_dict=True)
    last = attention_mask.long().sum(dim=1).clamp_min(1) - 1
    rows = torch.arange(real, device=device)
    logp = torch.log_softmax(out.logits[rows, last, :].float(), dim=-1)
    margins = logp[:, int(yes_id)] - logp[:, int(no_id)]
    hidden = torch.stack([out.hidden_states[l][rows, last, :].float() for l in layers])   # [layers, peers, H]
    padded = torch.zeros(num_peers, dtype=torch.float32, device=device)
    padded[:real] = margins
    per_peer = torch.zeros(num_peers, hidden.shape[0] * hidden.shape[2], device=device)
    per_peer[:real] = hidden.permute(1, 0, 2).reshape(real, -1)
    return {"q_mean": q_mean, "q_last": q_last, "sem": hidden.mean(dim=1).reshape(-1), "spread": hidden.std(dim=1, unbiased=False).reshape(-1),
            "margins": padded, "peer_hidden": per_peer, "layers": layers}
