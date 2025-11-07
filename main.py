"""
FastAPI application for word familiarity scoring.
"""

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from core.constants import API_VERSION, SUPPORTED_LANGUAGES
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
    learning_language: str = Field(..., description="Target language code (e.g., 'spa', 'ita', 'fra', 'eng')")
    native_language: str = Field(..., description="Native language code (e.g., 'spa', 'ita', 'fra', 'eng')")
    content: str = Field(..., description="The text content to analyze for familiarity")


class TokenScore(BaseModel):
    """Model for individual token scores."""
    text: str = Field(..., description="The token text")
    familiarity_score: float = Field(..., description="Familiarity score (0-1) based on word frequency")
    cognate_boosted_familiarity_score: float | None = Field(None, description="Familiarity score with cognate boost applied, null if no cognate found")
    cognate_before_LLM: str | None = Field(None, description="The cognate word found in dataset before LLM validation")
    cognate_after_LLM: str | None = Field(None, description="The cognate word in native language after LLM validation (null if rejected by LLM)")
    cognate_similarity: float | None = Field(None, description="Jaro-Winkler similarity score between search word and cognate (0-1), null if no cognate found")
    entity: str | None = Field(None, description="Named entity type (PER, LOC, ORG, etc.) if token is part of a named entity, null otherwise")


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
    # Startup
    logger.info("Starting API - preloading datasets and models...")

    # Preload all Stanza pipelines
    try:
        tokenizer.preload_all_pipelines()
        logger.info("Successfully preloaded all Stanza pipelines")
    except Exception as e:
        logger.error("Failed to preload Stanza pipelines: %s", str(e))
        raise RuntimeError(f"Startup failed: Could not preload Stanza pipelines - {str(e)}") from e

    # Load cognates dataset
    try:
        familiarity_scorer.load_cognates_dataset()
        logger.info("Successfully loaded cognates dataset")
    except Exception as e:
        logger.error("Failed to load cognates dataset: %s", str(e))
        raise RuntimeError(f"Startup failed: Could not load cognates dataset - {str(e)}") from e

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


@app.get("/")
async def root() -> dict:
    """Root endpoint with API information."""
    return {
        "message": "Word Familiarity API",
        "version": API_VERSION,
        "supported_languages": SUPPORTED_LANGUAGES,
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
        "supported_languages": SUPPORTED_LANGUAGES
    }


if __name__ == "__main__":
    import os

    import uvicorn

    # Check for development mode from environment variable
    dev_mode = os.getenv('DEV_MODE', 'false').lower() == 'true'

    if dev_mode:
        logger.info("Starting uvicorn server in DEVELOPMENT mode with hot reload")
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            # More targeted reload settings to avoid performance issues
            reload_dirs=[".", "core"],
            reload_delay=0.25
        )
    else:
        logger.info("Starting uvicorn server in PRODUCTION mode (no reload)")
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=7000,
            reload=False
        )
