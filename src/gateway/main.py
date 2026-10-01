"""FastAPI application factory."""

from fastapi import FastAPI

from gateway import __version__
from gateway.api.routes import router


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    app = FastAPI(
        title="LLM Inference Gateway",
        version=__version__,
        description="Semantic cache and dynamic batching in front of LLM inference engines.",
    )
    app.include_router(router)
    return app


app = create_app()
