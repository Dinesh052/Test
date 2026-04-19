"""Crisis Negotiator OpenEnv Client."""
from typing import Any, Dict

try:
    from openenv.core.env_client import EnvClient
    from openenv.core.client_types import StepResult
except ImportError as _e:
    raise ImportError("openenv-core is required: pip install openenv-core") from _e

from models import NegotiatorAction, CrisisObservation, CrisisState


class CrisisNegotiatorEnv(EnvClient[NegotiatorAction, CrisisObservation, CrisisState]):
    def _step_payload(self, action: NegotiatorAction) -> Dict[str, Any]:
        return action.model_dump(exclude={"metadata"})

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[CrisisObservation]:
        obs_data = payload.get("observation", payload)
        return StepResult(
            observation=CrisisObservation(**obs_data),
            reward=payload.get("reward") or obs_data.get("reward") or 0.0,
            done=payload.get("done", obs_data.get("done", False)),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> CrisisState:
        return CrisisState(**payload)
