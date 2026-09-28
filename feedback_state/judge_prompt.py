"""The frozen judge's prompt: one plain-text Yes/No question per peer, whose hidden states address the record.

For every peer i of an event the judge reads the question, the passage where there is one, all the answers as
``Response 0 ... Response k`` with peer i marked ``[candidate under review]``, and the instruction to answer Yes or No.
No chat template, no reliability information, no peer identities.
"""
from __future__ import annotations

from typing import Sequence

import torch

from feedback_state.tasks import _rag_context_text

INSTRUCTION = "Should the candidate response under review be selected as the best answer for the task? Answer only Yes or No."


def yes_no_token_ids(tokenizer) -> tuple[list[int], list[int]]:
    return tokenizer.encode(" Yes", add_special_tokens=False), tokenizer.encode(" No", add_special_tokens=False)


def context_text(record: dict) -> str:
    return _rag_context_text(record)


def judge_prompt(question: str, texts: Sequence[str], candidate: int, *, context: str | None = None) -> str:
    n = len(texts)
    if not 0 <= int(candidate) < n:
        raise ValueError(f"candidate={candidate} out of range for {n} responses")
    parts = [f"Question:\n{str(question).strip()}"]
    if context:
        parts.append(f"Context / Evidence:\n{str(context).strip()}")
    parts.append("Peer responses:")
    for slot in range(n):
        head = f"Response {slot}" + (" [candidate under review]" if slot == int(candidate) else "")
        parts.append(f"{head}:\n{str(texts[slot]).strip()}")
    parts.append(f"Candidate under review: Response {int(candidate)}")
    parts.append(f"Instruction:\n{INSTRUCTION}")
    parts.append("Answer:")
    return "\n\n".join(parts)


def judge_batch(tokenizer, question: str, texts: Sequence[str], *, context: str | None, max_length: int,
                device: torch.device | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """One judge prompt per response, tokenised into a right-padded batch; a prompt longer than max_length is an error."""
    encoded = []
    for slot in range(len(texts)):
        ids = tokenizer(judge_prompt(question, texts, slot, context=context), add_special_tokens=True, truncation=False)["input_ids"]
        if len(ids) > int(max_length):
            raise ValueError(f"judge prompt exceeds max_length: slot={slot}, length={len(ids)}, max_length={int(max_length)}")
        encoded.append(ids)
    if not encoded:
        raise ValueError("an event needs at least one response")
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (tokenizer.eos_token_id or 0)
    width = max(len(ids) for ids in encoded)
    input_ids = torch.full((len(encoded), width), int(pad_id), dtype=torch.long)
    attention_mask = torch.zeros((len(encoded), width), dtype=torch.long)
    for row, ids in enumerate(encoded):
        input_ids[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        attention_mask[row, : len(ids)] = 1
    if device is not None:
        input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
    return input_ids, attention_mask
