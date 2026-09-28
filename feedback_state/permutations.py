"""Peer identity and slot order.

* Canonical peer id: the index into ``sorted(peer_responses)`` (string order of the keys peer_0 ... peer_5), padded to
  the configured number of peers. The record is indexed by it; it never changes.
* Slot order: ``perm[slot] = peer id`` shown in that slot of a prompt; drawn per event with ``random_order`` so the
  prompt position carries no identity.
"""
from __future__ import annotations

import random
from typing import Any


def random_order(num_peers: int, seed: int) -> list[int]:
    """A reproducible slot -> peer id permutation."""
    perm = list(range(num_peers))
    random.Random(seed).shuffle(perm)
    return perm


def canonical_peer_view(record: dict[str, Any], num_peers: int, setting: str = "A") -> dict[str, Any]:
    """The peers of an event in canonical order: keys, model names, answer texts (padded to num_peers) and the real count."""
    responses = dict(record.get("peer_responses", {}))
    meta = dict(record.get("peer_metadata", {}))
    keys = sorted(responses)[:num_peers]
    real = len(keys)
    names = [str(dict(meta.get(k, {})).get("model") or k) for k in keys]
    texts = [str(responses[k]) for k in keys]
    while len(keys) < num_peers:
        pad = f"__pad_{len(keys)}"
        keys.append(pad); names.append(pad); texts.append("")
    return {"keys": keys, "names": names, "texts": texts, "real": real}
