"""FastAPI application for Crisis Negotiator Environment."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import HTTPException
from fastapi.responses import FileResponse, Response

from models import NegotiatorAction, CrisisObservation, CrisisState
from server.environment import CrisisNegotiatorEnvironment
from openenv.core.env_server.http_server import create_app

# Allow up to 8 concurrent training workers / parallel evaluators
app = create_app(
    CrisisNegotiatorEnvironment,
    NegotiatorAction,
    CrisisObservation,
    env_name="crisis_negotiator",
    max_concurrent_envs=8,
)


@app.get("/")
def root():
    return {
        "status": "ok",
        "env": "crisis_negotiator",
        "endpoints": [
            "POST /envs/crisis_negotiator/reset",
            "POST /envs/crisis_negotiator/step",
            "GET  /envs/crisis_negotiator/state",
            "GET  /ui",
        ],
        "ui": "/ui",
    }


@app.get("/ui")
def serve_ui():
    ui_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "ui",
        "index.html",
    )
    return FileResponse(ui_path, media_type="text/html")


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    # Tiny no-op favicon to silence 404 spam in logs
    return Response(status_code=204)


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
