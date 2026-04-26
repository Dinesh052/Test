"""
env_extensions.py — Monkey-patch CrisisNegotiatorEnvironment with performance helpers.

Import ONCE at training startup before any env is created:
    import env_extensions  # noqa — applies patches

New methods:
    env.hidden_snapshot()  → (agitation, trust, demands_list, personality) [no deepcopy]
    env.fast_rollout(action, heuristic_cycle, n_steps) → (total_reward, done, outcome_msg)
    env.peek_step_reward(action) → float [shallow-clone, no state commit]
"""
from __future__ import annotations
import copy
from typing import Any, Dict, List, Optional, Tuple

from models import NegotiatorAction
from server.environment import CrisisNegotiatorEnvironment


def _hidden_snapshot(self) -> Optional[Tuple[float, float, List[Dict], str]]:
    h = self._hidden
    if h is None:
        return None
    demands_info = [
        {"text": d.text, "acknowledged": d.acknowledged, "priority": d.priority}
        for d in h.demands
    ]
    return (h.agitation, h.trust, demands_info, h.personality)


def _fast_rollout(
    self,
    first_action: NegotiatorAction,
    heuristic_cycle: List[Tuple[str, str]],
    n_steps: int = 6,
) -> Tuple[float, bool, str]:
    """Shallow-clone rollout. ~5-8x faster than deepcopy for 6-step trajectories."""
    clone = object.__new__(CrisisNegotiatorEnvironment)
    d = self.__dict__.copy()
    d["_hidden"] = copy.deepcopy(self._hidden)
    d["_state"] = copy.deepcopy(self._state)
    d["_dialogue"] = list(self._dialogue)
    d["_commander_msgs"] = list(self._commander_msgs)
    d["_supervisor_flags"] = list(self._supervisor_flags)
    d["_all_supervisor_flags"] = list(self._all_supervisor_flags)
    d["_agitation_history"] = list(self._agitation_history)
    d["_actions_taken"] = list(self._actions_taken)
    d["_actor_msgs"] = list(self._actor_msgs)
    clone.__dict__.update(d)

    total_reward = 0.0
    obs = None
    try:
        obs = clone.step(first_action)
        total_reward += float(getattr(obs, "reward", 0.0))
        for t in range(n_steps - 1):
            if getattr(obs, "done", False):
                break
            at, content = heuristic_cycle[t % len(heuristic_cycle)]
            obs = clone.step(NegotiatorAction(
                action_type=at, content=content,
                reasoning="fast_rollout", target="hostage_taker"))
            total_reward += float(getattr(obs, "reward", 0.0))
    except Exception:
        pass

    done = bool(getattr(obs, "done", False)) if obs else False
    msg = getattr(obs, "message", "") or "" if obs else ""
    return total_reward, done, msg


def _peek_step_reward(self, action: NegotiatorAction) -> float:
    """Single-step lookahead without state commit."""
    reward, _, _ = self.fast_rollout(action, [], n_steps=1)
    return reward


CrisisNegotiatorEnvironment.hidden_snapshot = _hidden_snapshot
CrisisNegotiatorEnvironment.fast_rollout = _fast_rollout
CrisisNegotiatorEnvironment.peek_step_reward = _peek_step_reward
