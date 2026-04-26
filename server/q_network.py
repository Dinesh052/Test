"""DialogXpert-style Q-network action ranker.

A 2-layer MLP trained via TD-learning that scores the 10 candidate action
types given a BERT-encoded observation. 10× more sample-efficient than
full GRPO fine-tuning.

Reference: DialogXpert (AAAI 2025, arXiv:2505.17795)
"""
from __future__ import annotations
import json
import os
from typing import List, Optional, Tuple

ACTION_TYPES = [
    "speak", "request_demand", "acknowledge_demand", "offer_concession",
    "ask_proof_of_life", "buy_time", "push_back_commander",
    "emotional_label", "mirror", "open_question",
]
ACTION_TO_IDX = {a: i for i, a in enumerate(ACTION_TYPES)}

_encoder = None
_q_net = None


def _load_encoder():
    global _encoder
    if _encoder is not None:
        return
    # sentence_transformers triggers pyarrow C extension crash on some Windows
    # installs. Default to unavailable; set EMOTION_USE_TRANSFORMER=1 to try.
    import os
    if os.environ.get("EMOTION_USE_TRANSFORMER") != "1":
        _encoder = "unavailable"
        return
    try:
        from sentence_transformers import SentenceTransformer
        _encoder = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L6-v2")
    except ImportError:
        _encoder = "unavailable"


def _build_q_net():
    """Build a simple 2-layer MLP: 384 → 64 → 10."""
    global _q_net
    try:
        import torch
        import torch.nn as nn
        _q_net = nn.Sequential(
            nn.Linear(384, 64),
            nn.ReLU(),
            nn.Linear(64, len(ACTION_TYPES)),
        )
        # Try to load saved weights
        weights_path = os.path.join(os.path.dirname(__file__), "..", "q_network.pt")
        if os.path.exists(weights_path):
            _q_net.load_state_dict(torch.load(weights_path, weights_only=True))
        _q_net.eval()
    except ImportError:
        _q_net = "unavailable"


def encode_observation(obs_text: str):
    """Encode observation text to 384-dim vector."""
    _load_encoder()
    if _encoder == "unavailable":
        return None
    return _encoder.encode([obs_text])[0]


def rank_actions(obs_text: str) -> Optional[List[Tuple[str, float]]]:
    """Rank all 10 action types by Q-value for the given observation.

    Returns: sorted list of (action_type, q_value) or None if unavailable.
    """
    _load_encoder()
    if _encoder == "unavailable":
        return None
    _build_q_net() if _q_net is None else None
    if _q_net == "unavailable":
        return None

    import torch
    emb = encode_observation(obs_text)
    with torch.no_grad():
        q_values = _q_net(torch.tensor(emb, dtype=torch.float32))
    ranked = sorted(
        [(ACTION_TYPES[i], q_values[i].item()) for i in range(len(ACTION_TYPES))],
        key=lambda x: x[1], reverse=True,
    )
    return ranked


def td_update(
    obs_text: str,
    action_type: str,
    reward: float,
    next_obs_text: str,
    done: bool,
    lr: float = 1e-3,
    gamma: float = 0.95,
) -> Optional[float]:
    """Single TD(0) update step. Returns loss or None if unavailable."""
    _load_encoder()
    if _encoder == "unavailable":
        return None
    _build_q_net() if _q_net is None else None
    if _q_net == "unavailable":
        return None

    import torch
    import torch.nn.functional as F

    _q_net.train()
    emb = torch.tensor(encode_observation(obs_text), dtype=torch.float32)
    next_emb = torch.tensor(encode_observation(next_obs_text), dtype=torch.float32)
    action_idx = ACTION_TO_IDX.get(action_type, 0)

    q_values = _q_net(emb)
    q_sa = q_values[action_idx]

    with torch.no_grad():
        next_q = _q_net(next_emb)
        target = reward + (0.0 if done else gamma * next_q.max())

    loss = F.mse_loss(q_sa, torch.tensor(target, dtype=torch.float32))

    if not hasattr(td_update, '_optimizer'):
        td_update._optimizer = torch.optim.Adam(_q_net.parameters(), lr=lr)
    td_update._optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    _q_net.eval()
    return loss.item()


def save_q_network(path: str = "q_network.pt"):
    """Save trained Q-network weights."""
    if _q_net is None or _q_net == "unavailable":
        return
    import torch
    torch.save(_q_net.state_dict(), path)
