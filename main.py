"""
FastAPI application for word familiarity scoring.
"""

import logging
import time
from contextlib import asynccontextmanager
from typing import Optional, List
import asyncio
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Configure detailed logging with timing info
logging.basicConfig(
    level=logging.DEBUG,  # More verbose logging
    format='%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - [TIMING] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Track overall startup timing
STARTUP_START_TIME = time.time()
logger.info("=== APPLICATION STARTUP BEGAN ===")

# Time each import step
logger.info("Starting imports...")
import_start = time.time()

logger.info("Importing core.constants...")
constants_start = time.time()
from core.constants import API_VERSION, SUPPORTED_LANGUAGES
logger.info("core.constants imported in %.3f seconds", time.time() - constants_start)

logger.info("Importing core.score_model...")
score_model_start = time.time()
from core.score_model import familiarity_scorer
logger.info("core.score_model imported in %.3f seconds", time.time() - score_model_start)

logger.info("Importing core.tokenizer...")
tokenizer_start = time.time()
from core.tokenizer import tokenizer
logger.info("core.tokenizer imported in %.3f seconds", time.time() - tokenizer_start)

total_import_time = time.time() - import_start
logger.info("All imports completed in %.3f seconds", total_import_time)


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
    lifespan_start = time.time()
    logger.info("=== LIFESPAN STARTUP PHASE BEGAN ===")
    logger.info("Starting API - preloading datasets and models...")
    
    # Preload all Stanza pipelines with detailed timing
    preload_start = time.time()
    logger.info("Beginning Stanza pipeline preloading...")
    
    try:
        tokenizer.preload_all_pipelines()
        
        preload_duration = time.time() - preload_start
        logger.info("Successfully preloaded all Stanza pipelines in %.3f seconds", preload_duration)
        
    except Exception as e:
        preload_duration = time.time() - preload_start
        logger.error("Failed to preload Stanza pipelines after %.3f seconds: %s", preload_duration, str(e))
        raise RuntimeError(f"Startup failed: Could not preload Stanza pipelines - {str(e)}") from e
    
    # Test OpenAI connectivity with timing
    openai_test_start = time.time()
    logger.info("Testing OpenAI API connectivity...")
    try:
        # Import and test the OpenAI detector
        from core.openai_cognate_detector import openai_cognate_detector
        
        if openai_cognate_detector.client is None:
            logger.warning("OpenAI client not initialized - cognate detection will be disabled")
        else:
            logger.info("OpenAI client ready for cognate detection")
            
        openai_test_duration = time.time() - openai_test_start
        logger.info("OpenAI connectivity test completed in %.3f seconds", openai_test_duration)
        
    except Exception as e:
        openai_test_duration = time.time() - openai_test_start
        logger.error("OpenAI initialization test failed after %.3f seconds: %s", openai_test_duration, str(e))
    
    total_lifespan_duration = time.time() - lifespan_start
    total_startup_duration = time.time() - STARTUP_START_TIME
    
    logger.info("=== API STARTUP COMPLETE ===")
    logger.info("Lifespan phase duration: %.3f seconds", total_lifespan_duration)
    logger.info("Total application startup duration: %.3f seconds", total_startup_duration)
    logger.info("API is ready to handle requests")
    
    yield
    
    # Shutdown (if needed)
    shutdown_start = time.time()
    logger.info("=== API SHUTDOWN INITIATED ===")
    logger.info("API shutdown completed in %.3f seconds", time.time() - shutdown_start)


# Initialize FastAPI app
logger.info("Creating FastAPI application...")
app_creation_start = time.time()

app = FastAPI(
    title="Word Familiarity API",
    version=API_VERSION,
    description="Computes per-token familiarity scores for phrases in target languages with cognate boosting",
    lifespan=lifespan
)

app_creation_time = time.time() - app_creation_start
logger.info("FastAPI application created in %.3f seconds", app_creation_time)


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
    
    # Create a thread pool executor for concurrent processing
    async def process_single_request(request: FamiliarityRequest, index: int) -> FamiliarityResponse:
        """Process a single familiarity request."""
        logger.debug("Processing batch item %d/%d", index + 1, len(batch_request.requests))
        
        # Validate languages
        if request.learning_language not in SUPPORTED_LANGUAGES:
            logger.error(f"Unsupported learning language in batch item {index+1}: {request.learning_language}")
            raise HTTPException(
                status_code=400,
                detail=f"Learning language '{request.learning_language}' in request {index+1} not supported. "
                       f"Supported: {list(SUPPORTED_LANGUAGES.keys())}"
            )
        
        if request.native_language not in SUPPORTED_LANGUAGES:
            logger.error(f"Unsupported native language in batch item {index+1}: {request.native_language}")
            raise HTTPException(
                status_code=400,
                detail=f"Native language '{request.native_language}' in request {index+1} not supported. "
                       f"Supported: {list(SUPPORTED_LANGUAGES.keys())}"
            )
        
        # Validate phrase
        if not request.phrase.strip():
            logger.error(f"Empty phrase in batch item {index+1}")
            raise HTTPException(
                status_code=400,
                detail=f"Phrase in request {index+1} cannot be empty"
            )
        
        # Compute familiarity scores in thread pool (since it's CPU-bound)
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            result = await loop.run_in_executor(
                executor,
                familiarity_scorer.compute_familiarity,
                request.phrase,
                request.learning_language,
                request.native_language
            )
        
        return FamiliarityResponse(**result)
    
    try:
        # Process all requests concurrently
        tasks = [
            process_single_request(request, i) 
            for i, request in enumerate(batch_request.requests)
        ]
        
        logger.info("Starting concurrent processing of %d requests", len(tasks))
        responses = await asyncio.gather(*tasks)
        
        processing_time_ms = (time.time() - start_time) * 1000
        logger.info("Batch processing complete - %d requests processed concurrently in %.2f ms", 
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
    logger.info("=== STARTING UVICORN SERVER ===")
    uvicorn_start = time.time()
    
    import uvicorn
    
    logger.info("Starting uvicorn server on http://0.0.0.0:8000 with reload=True")
    logger.info("Application startup phase complete - %.3f seconds elapsed", 
               time.time() - STARTUP_START_TIME)
    
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000
    )