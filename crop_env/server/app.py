"""
FastAPI application for the CropRL Environment.

Usage:
    uvicorn crop_env.server.app:app --reload --host 0.0.0.0 --port 8000
"""

try:
    from openenv.core.env_server.http_server import create_app
except ImportError:
    import traceback
    traceback.print_exc()
    # Fallback: if http_server is not available, create a minimal app
    create_app = None

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
try:
    import pandas as pd
    warnings.filterwarnings("ignore", category=pd.errors.Pandas4Warning)
except ImportError:
    pass

import sys
import os

if __package__ == "server" or __package__ is None:
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    __package__ = "crop_env.server"

from ..models import CropAction, CropObservation
from .environment import CropEnvironment

if create_app is not None:
    app = create_app(
        CropEnvironment, CropAction, CropObservation, env_name="crop_env"
    )

    from fastapi.responses import RedirectResponse
    @app.get("/", include_in_schema=False)
    def root():
        return RedirectResponse(url="/docs")

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
