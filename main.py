"""
FastAPI application for word familiarity scoring.
"""

from __future__ import annotations

# Fix PyTorch 2.6+ weights_only for Stanza model loading (must run before stanza import)
import torch

_original_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)


torch.load = _patched_torch_load

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from core.constants import API_VERSION, SUPPORTED_LANGUAGES
from core.language_codes import alias_map_for_api, normalize_language_request
from core.score_model import familiarity_scorer
from core.tokenizer import tokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FamiliarityRequest(BaseModel):
    """Request model for familiarity scoring."""
    learning_language: str = Field(
        ...,
        description="Target language: canonical ISO 639-3 (e.g. spa, jpn) or locale alias (e.g. es-US, zh-Hans)",
    )
    native_language: str = Field(
        ...,
        description="User language: canonical ISO 639-3 or locale alias (see GET /languages locale_aliases)",
    )
    content: str = Field(..., description="The text content to analyze for familiarity")

    @field_validator("learning_language", "native_language", mode="before")
    @classmethod
    def _normalize_language_codes(cls, v: object) -> str:
        return normalize_language_request(str(v))


class TokenScore(BaseModel):
    """Model for individual token scores."""
    text: str = Field(..., description="The token text")
    familiarity_score: float = Field(..., description="Familiarity score (0-1) based on word frequency")
    cognate_boosted_familiarity_score: Optional[float] = Field(None, description="Familiarity score with cognate boost applied, null if no cognate found")
    cognate_before_LLM: Optional[str] = Field(None, description="The cognate word found in dataset before LLM validation")
    cognate_after_LLM: Optional[str] = Field(None, description="The cognate word in native language after LLM validation (null if rejected by LLM)")
    cognate_similarity: Optional[float] = Field(None, description="Jaro-Winkler similarity score between search word and cognate (0-1), null if no cognate found")
    entity: Optional[str] = Field(None, description="Named entity type (PER, LOC, ORG, etc.) if token is part of a named entity, null otherwise")


class SentenceScore(BaseModel):
    """Model for sentence-level scores."""
    text: str = Field(..., description="The sentence text")
    index: int = Field(..., description="The index of the sentence in the document")
    tokens: list[TokenScore] = Field(..., description="List of token scores for this sentence")


class FamiliarityResponse(BaseModel):
    """Response model for familiarity scoring."""
    content: str = Field(..., description="The analyzed content")
    learning_language: str = Field(..., description="Learning language code of the content")
    native_language: str = Field(..., description="Native language code of the user")
    timestamp: str = Field(..., description="ISO timestamp of analysis")
    sentences: list[SentenceScore] = Field(..., description="List of sentence scores")
    total_tokens: int = Field(..., description="Total number of tokens across all sentences")



@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan event handler for startup and shutdown."""
    import os

    logger.info("Starting API - loading datasets...")

    # Stanza: lazy-load by default to stay under 4GB RAM. Set PRELOAD_LANGUAGES=spa,eng to
    # preload specific languages (useful on 8GB+ instances for faster first requests).
    preload_langs = os.getenv("PRELOAD_LANGUAGES", "").strip()
    if preload_langs:
        langs = [
            normalize_language_request(x.strip())
            for x in preload_langs.split(",")
            if x.strip()
        ]
        if langs:
            try:
                tokenizer.preload_pipelines(langs)
                logger.info("Preloaded Stanza pipelines for: %s", langs)
            except Exception as e:
                logger.error("Failed to preload Stanza pipelines: %s", str(e))
                raise RuntimeError(f"Startup failed: {str(e)}") from e

    # Load cognates dataset (optional - run `git lfs pull` if file is LFS pointer)
    try:
        familiarity_scorer.load_cognates_dataset()
        logger.info("Successfully loaded cognates dataset")
    except Exception as e:
        logger.warning("Cognates dataset not loaded (cognate boosting disabled): %s", str(e))

    logger.info("API startup complete - all models preloaded")

    yield

    # Shutdown (if needed)
    logger.info("API shutdown")


# Initialize FastAPI app
app = FastAPI(
    title="Word Familiarity API",
    version=API_VERSION,
    description="Computes per-token familiarity scores for phrases in target languages based on word frequency",
    lifespan=lifespan
)

# Browser clients (SPA dev + deployed frontends) require CORS; Postman/curl do not.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:5175",
        "http://mediumseagreen-rail-433210.hostingersite.com",
        "https://mediumseagreen-rail-433210.hostingersite.com",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)


@app.get("/")
async def root() -> dict:
    """Root endpoint with API information."""
    return {
        "message": "Word Familiarity API",
        "version": API_VERSION,
        "supported_languages": SUPPORTED_LANGUAGES,
        "locale_aliases": alias_map_for_api(),
        "endpoints": {
            "familiarity": "/familiarity (POST) - Analyze content familiarity scores",
            "languages": "/languages (GET)",
            "health": "/health (GET)"
        }
    }


@app.get("/health")
async def health_check() -> dict:
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/familiarity", response_model=FamiliarityResponse)
async def compute_familiarity(request: FamiliarityRequest) -> FamiliarityResponse:
    """
    Compute familiarity scores for all tokens in the content.

    The endpoint analyzes each token in each sentence and returns:
    - Familiarity score based on word frequency
    - Cognate-boosted familiarity score if cognates are found
    - Results organized by sentence structure

    Args:
        request: FamiliarityRequest with content and language information

    Returns:
        FamiliarityResponse with detailed sentence and token scores

    Raises:
        HTTPException: If language is not supported or processing fails
    """
    logger.info("Received familiarity request for %s -> %s", request.learning_language, request.native_language)

    try:
        # Validate languages
        if request.learning_language not in SUPPORTED_LANGUAGES:
            logger.error("Unsupported learning language: %s", request.learning_language)
            raise HTTPException(
                status_code=400,
                detail=f"Learning language '{request.learning_language}' not supported. "
                       f"Supported: {list(SUPPORTED_LANGUAGES.keys())}"
            )

        if request.native_language not in SUPPORTED_LANGUAGES:
            logger.error("Unsupported native language: %s", request.native_language)
            raise HTTPException(
                status_code=400,
                detail=f"Native language '{request.native_language}' not supported. "
                       f"Supported: {list(SUPPORTED_LANGUAGES.keys())}"
            )

        # Validate content
        if not request.content.strip():
            logger.error("Empty content provided")
            raise HTTPException(
                status_code=400,
                detail="Content cannot be empty"
            )

        logger.info("Processing content with %d words", len(request.content.split()))

        # Compute familiarity scores for the content
        result = familiarity_scorer.compute_document_familiarity(
            document=request.content,
            learning_language=request.learning_language,
            native_language=request.native_language
        )

        logger.info("Successfully processed %d tokens in %d sentences", result['total_tokens'], len(result['sentences']))
        return FamiliarityResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Unexpected error processing request: %s", str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        ) from e




@app.get("/languages")
async def get_supported_languages() -> dict:
    """Get list of supported languages."""
    return {
        "supported_languages": SUPPORTED_LANGUAGES,
        "locale_aliases": alias_map_for_api(),
    }


if __name__ == "__main__":
    import os

    import uvicorn

    # Check for development mode from environment variable
    dev_mode = os.getenv('DEV_MODE', 'false').lower() == 'true'

    if dev_mode:
        port = int(os.getenv("PORT", "8000"))
        logger.info("Starting uvicorn server in DEVELOPMENT mode with hot reload")
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=port,
            reload=True,
            # More targeted reload settings to avoid performance issues
            reload_dirs=[".", "core"],
            reload_delay=0.25
        )
    else:
        port = int(os.getenv("PORT", "8080"))
        logger.info("Starting uvicorn server in PRODUCTION mode (no reload)")
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=port,
            reload=False
        )
