"""
FastAPI application for the CropRL Environment.

Usage:
    uvicorn crop_env.server.app:app --reload --host 0.0.0.0 --port 8000
"""

try:
    from openenv.core.env_server.http_server import create_app
except ImportError:
    # Fallback: if http_server is not available, create a minimal app
    create_app = None

from ..models import CropAction, CropObservation
from .crop_environment import CropEnvironment

if create_app is not None:
    app = create_app(
        CropEnvironment, CropAction, CropObservation, env_name="crop_env"
    )
else:
    # Minimal FastAPI fallback for local development without full OpenEnv
    from fastapi import FastAPI

    app = FastAPI(title="CropRL Environment")

    @app.get("/health")
    def health():
        return {"status": "healthy"}


def main():
    """Entry point for direct execution."""
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
