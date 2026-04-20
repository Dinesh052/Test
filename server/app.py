"""FastAPI application for Crisis Negotiator Environment."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import NegotiatorAction, CrisisObservation, CrisisState
from server.environment import CrisisNegotiatorEnvironment
from openenv.core.env_server.http_server import create_app

app = create_app(CrisisNegotiatorEnvironment, NegotiatorAction, CrisisObservation, env_name="crisis_negotiator", max_concurrent_envs=1)


@app.get("/")
def root():
    return {"status": "ok", "env": "crisis_negotiator"}


@app.get("/ui")
def serve_ui():
    from fastapi.responses import FileResponse
    import os
    ui_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ui", "index.html")
    return FileResponse(ui_path, media_type="text/html")


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
