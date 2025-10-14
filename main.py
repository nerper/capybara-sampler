"""
FastAPI application for word familiarity scoring.
"""

import logging
import time
from contextlib import asynccontextmanager
from typing import Optional, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from core.constants import API_VERSION, SUPPORTED_LANGUAGES
from core.score_model import familiarity_scorer
from core.cognate_lookup import cognate_loader
from core.tokenizer import tokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FamiliarityRequest(BaseModel):
    """Request model for familiarity scoring."""
    phrase: str = Field(..., description="The phrase to analyze for familiarity")
    learning_language: str = Field(..., description="Target language code (e.g., 'spa', 'ita', 'fra', 'eng')")
    native_language: str = Field(
        description="Native language code for cognate boosting"
    )


class TokenScore(BaseModel):
    """Model for individual token scores."""
    text: str = Field(..., description="The token text")
    familiarity_score: float = Field(..., description="Base familiarity score (0-1)")
    cognate_familiarity_score: Optional[float] = Field(
        None, 
        description="Cognate-boosted familiarity score (0-1), present if cognates found"
    )
    cognate: Optional[str] = Field(
        None,
        description="Cognate word in the native language, present if cognate found"
    )


class FamiliarityResponse(BaseModel):
    """Response model for familiarity scoring."""
    phrase: str = Field(..., description="The analyzed phrase")
    learning_language: str = Field(..., description="Learning language code of the phrase")
    native_language: str = Field(..., description="Native language code used for cognate boosting")
    timestamp: str = Field(..., description="ISO timestamp of analysis")
    tokens: List[TokenScore] = Field(..., description="List of token scores")


class BatchFamiliarityRequest(BaseModel):
    """Request model for batch familiarity scoring."""
    requests: List[FamiliarityRequest] = Field(..., description="List of familiarity requests to process")


class BatchFamiliarityResponse(BaseModel):
    """Response model for batch familiarity scoring."""
    responses: List[FamiliarityResponse] = Field(..., description="List of familiarity responses")
    total_processed: int = Field(..., description="Total number of requests processed")
    processing_time_ms: float = Field(..., description="Total processing time in milliseconds")


@asynccontextmanager
async def lifespan(app: FastAPI):
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
    
    # Preload the CogNet dataset
    try:
        cognate_loader.preload_dataset()
        logger.info("Successfully preloaded CogNet dataset")
    except Exception as e:
        logger.error("Failed to preload CogNet dataset: %s", str(e))
        raise RuntimeError(f"Startup failed: Could not preload CogNet dataset - {str(e)}") from e
    
    logger.info("API startup complete - all models and datasets preloaded")
    
    yield
    
    # Shutdown (if needed)
    logger.info("API shutdown")


# Initialize FastAPI app
app = FastAPI(
    title="Word Familiarity API",
    version=API_VERSION,
    description="Computes per-token familiarity scores for phrases in target languages with cognate boosting",
    lifespan=lifespan
)


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Word Familiarity API",
        "version": API_VERSION,
        "supported_languages": SUPPORTED_LANGUAGES,
        "endpoints": {
            "familiarity": "/familiarity (POST)",
            "batch_familiarity": "/familiarity-batch (POST)",
            "languages": "/languages (GET)",
            "health": "/health (GET)"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/familiarity", response_model=FamiliarityResponse)
async def compute_familiarity(request: FamiliarityRequest):
    """
    Compute familiarity scores for all tokens in a phrase.
    
    The endpoint analyzes each token in the phrase and returns:
    - Base familiarity score based on word frequency
    - Cognate-boosted score if cognates are found with the native language
    
    Args:
        request: FamiliarityRequest with phrase and language information
        
    Returns:
        FamiliarityResponse with detailed token scores
        
    Raises:
        HTTPException: If language is not supported or processing fails
    """
    logger.info("Received familiarity request for %s -> %s", request.learning_language, request.native_language)
    
    try:
        # Validate languages
        if request.learning_language not in SUPPORTED_LANGUAGES:
            logger.error(f"Unsupported learning language: {request.learning_language}")
            raise HTTPException(
                status_code=400,
                detail=f"Learning language '{request.learning_language}' not supported. "
                       f"Supported: {list(SUPPORTED_LANGUAGES.keys())}"
            )
        
        if request.native_language not in SUPPORTED_LANGUAGES:
            logger.error(f"Unsupported native language: {request.native_language}")
            raise HTTPException(
                status_code=400,
                detail=f"Native language '{request.native_language}' not supported. "
                       f"Supported: {list(SUPPORTED_LANGUAGES.keys())}"
            )
        
        # Validate phrase
        if not request.phrase.strip():
            logger.error("Empty phrase provided")
            raise HTTPException(
                status_code=400,
                detail="Phrase cannot be empty"
            )
        
        logger.info(f"Processing phrase with {len(request.phrase.split())} words")
        
        # Compute familiarity scores
        result = familiarity_scorer.compute_familiarity(
            phrase=request.phrase,
            learning_language=request.learning_language,
            native_language=request.native_language
        )
        
        logger.info(f"Successfully processed {len(result['tokens'])} tokens")
        return FamiliarityResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error processing request: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@app.post("/familiarity-batch", response_model=BatchFamiliarityResponse)
async def compute_familiarity_batch(batch_request: BatchFamiliarityRequest):
    """
    Compute familiarity scores for multiple phrases in batch.
    
    The endpoint processes a list of familiarity requests and returns responses for each.
    All requests are processed sequentially with shared model loading for efficiency.
    
    Args:
        batch_request: BatchFamiliarityRequest with list of familiarity requests
        
    Returns:
        BatchFamiliarityResponse with list of responses and processing metadata
        
    Raises:
        HTTPException: If any request has invalid parameters or processing fails
    """
    start_time = time.time()
    logger.info("Received batch familiarity request with %d phrases", len(batch_request.requests))
    
    if not batch_request.requests:
        raise HTTPException(
            status_code=400,
            detail="Batch request cannot be empty"
        )
    
    responses = []
    
    try:
        for i, request in enumerate(batch_request.requests):
            logger.debug("Processing batch item %d/%d", i + 1, len(batch_request.requests))
            
            # Validate languages
            if request.learning_language not in SUPPORTED_LANGUAGES:
                logger.error(f"Unsupported learning language in batch item {i+1}: {request.learning_language}")
                raise HTTPException(
                    status_code=400,
                    detail=f"Learning language '{request.learning_language}' in request {i+1} not supported. "
                           f"Supported: {list(SUPPORTED_LANGUAGES.keys())}"
                )
            
            if request.native_language not in SUPPORTED_LANGUAGES:
                logger.error(f"Unsupported native language in batch item {i+1}: {request.native_language}")
                raise HTTPException(
                    status_code=400,
                    detail=f"Native language '{request.native_language}' in request {i+1} not supported. "
                           f"Supported: {list(SUPPORTED_LANGUAGES.keys())}"
                )
            
            # Validate phrase
            if not request.phrase.strip():
                logger.error(f"Empty phrase in batch item {i+1}")
                raise HTTPException(
                    status_code=400,
                    detail=f"Phrase in request {i+1} cannot be empty"
                )
            
            # Compute familiarity scores
            result = familiarity_scorer.compute_familiarity(
                phrase=request.phrase,
                learning_language=request.learning_language,
                native_language=request.native_language
            )
            
            responses.append(FamiliarityResponse(**result))
        
        processing_time_ms = (time.time() - start_time) * 1000
        logger.info("Batch processing complete - %d requests processed in %.2f ms", 
                   len(responses), processing_time_ms)
        
        return BatchFamiliarityResponse(
            responses=responses,
            total_processed=len(responses),
            processing_time_ms=round(processing_time_ms, 2)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error processing batch request: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error during batch processing: {str(e)}"
        )


@app.get("/languages")
async def get_supported_languages():
    """Get list of supported languages."""
    return {
        "supported_languages": SUPPORTED_LANGUAGES
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)