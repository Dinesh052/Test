"""FastAPI application for Crisis Negotiator Environment."""
import sys, os, json, time, uuid
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load .env if present
from pathlib import Path
_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

from fastapi import HTTPException, Query
from fastapi.responses import FileResponse, Response
from sse_starlette.sse import EventSourceResponse

from models import NegotiatorAction, CrisisObservation, CrisisState
from server.environment import CrisisNegotiatorEnvironment
from openenv.core.env_server.http_server import create_app

app = create_app(
    CrisisNegotiatorEnvironment,
    NegotiatorAction,
    CrisisObservation,
    env_name="crisis_negotiator",
    max_concurrent_envs=4,
)

# ── Episode storage ──────────────────────────────────────
_episodes: dict[str, dict] = {}

# ── Heuristic policy for autoplay ────────────────────────
_HEURISTIC = [
    ("emotional_label", "It sounds like you're carrying a tremendous amount of pain right now."),
    ("mirror", "Tell me more — you said something that mattered."),
    ("open_question", "What happened, from your side? I have time."),
    ("emotional_label", "That sounds completely overwhelming. Anyone in your shoes would be struggling."),
    ("acknowledge_demand", "I hear what you're asking for. That's not unreasonable. Let me see what I can do."),
    ("open_question", "What would feel like the right outcome for you here?"),
    ("acknowledge_demand", "What you said about that — I'm taking it seriously."),
    ("offer_concession", "Here's what I can do right now: I can have someone on the phone for you within minutes."),
    ("emotional_label", "I can hear how exhausted you are. Let's find a way through this together."),
    ("acknowledge_demand", "Your request — I'm advocating for it on my end. That's real."),
]


@app.get("/")
def root():
    return {
        "status": "ok",
        "env": "crisis_negotiator",
        "endpoints": [
            "POST /reset", "POST /step", "GET /state",
            "GET /autoplay", "GET /episodes", "GET /episodes/{id}",
            "GET /ui",
        ],
        "ui": "/ui",
    }


@app.get("/ui")
def serve_ui():
    ui_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "ui", "index.html",
    )
    return FileResponse(ui_path, media_type="text/html")


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)


@app.get("/groq-key")
def groq_key():
    key = os.environ.get("GROQ_API_KEY", "")
    if key:
        return {"key": key}
    return {"key": ""}


# ── /autoplay SSE — runs a full episode, streams each step ──
@app.get("/autoplay")
async def autoplay(
    task_id: str = Query("easy_domestic_desperate"),
    seed: int = Query(42),
    delay: float = Query(0.8, description="seconds between steps"),
):
    async def event_stream():
        env = CrisisNegotiatorEnvironment()
        obs = env.reset(task_id=task_id, seed=seed)
        ep_id = str(uuid.uuid4())[:8]
        trajectory = []

        # Send initial observation
        init = {
            "type": "reset", "episode_id": ep_id, "task_id": task_id,
            "phase": getattr(obs, "phase", "opening"),
            "last_ht_message": getattr(obs, "last_ht_message", ""),
            "stated_demands": getattr(obs, "stated_demands", []),
            "scenario_brief": getattr(obs, "scenario_brief", ""),
        }
        yield {"event": "reset", "data": json.dumps(init)}

        step = 0
        done = False
        while not done and step < 30:
            at, content = _HEURISTIC[step % len(_HEURISTIC)]
            action = NegotiatorAction(
                action_type=at, content=content,
                reasoning="heuristic BCSM cycle", target="hostage_taker",
            )
            obs = env.step(action)
            step += 1
            reward = float(getattr(obs, "reward", 0))
            done = bool(getattr(obs, "done", False))

            step_data = {
                "type": "step", "step": step,
                "action_type": at, "content": content,
                "reward": round(reward, 4),
                "phase": getattr(obs, "phase", ""),
                "last_ht_message": getattr(obs, "last_ht_message", ""),
                "last_ht_cues": getattr(obs, "last_ht_cues", []),
                "commander_patience": getattr(obs, "commander_patience", "patient"),
                "commander_messages": getattr(obs, "commander_messages", []),
                "hostage_whisper": getattr(obs, "hostage_whisper", None),
                "stated_demands": getattr(obs, "stated_demands", []),
                "done": done,
                "message": getattr(obs, "message", ""),
            }
            trajectory.append(step_data)
            yield {"event": "step", "data": json.dumps(step_data)}

            if not done:
                import asyncio
                await asyncio.sleep(delay)

        # Save episode
        episode = {"id": ep_id, "task_id": task_id, "seed": seed, "steps": step, "trajectory": trajectory}
        _episodes[ep_id] = episode
        yield {"event": "done", "data": json.dumps({"episode_id": ep_id, "steps": step, "final_reward": trajectory[-1]["reward"] if trajectory else 0})}

    return EventSourceResponse(event_stream())


# ── /episodes — list + retrieve saved episodes ──
@app.get("/episodes")
def list_episodes():
    return [
        {"id": eid, "task_id": ep["task_id"], "steps": ep["steps"], "seed": ep["seed"]}
        for eid, ep in _episodes.items()
    ]


@app.get("/episodes/{episode_id}")
def get_episode(episode_id: str):
    ep = _episodes.get(episode_id)
    if not ep:
        raise HTTPException(404, f"Episode {episode_id} not found")
    return ep


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
