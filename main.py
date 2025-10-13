"""
FastAPI application for word familiarity scoring.
"""

import logging
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


class FamiliarityResponse(BaseModel):
    """Response model for familiarity scoring."""
    phrase: str = Field(..., description="The analyzed phrase")
    language: str = Field(..., description="Language code of the phrase")
    timestamp: str = Field(..., description="ISO timestamp of analysis")
    tokens: List[TokenScore] = Field(..., description="List of token scores")


# Initialize FastAPI app
app = FastAPI(
    title="Word Familiarity API",
    version=API_VERSION,
    description="Computes per-token familiarity scores for phrases in target languages with cognate boosting"
)


@app.on_event("startup")
async def startup_event():
    """Preload datasets and models on startup."""
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


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Word Familiarity API",
        "version": API_VERSION,
        "supported_languages": SUPPORTED_LANGUAGES,
        "endpoints": {
            "familiarity": "/familiarity (POST)"
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


@app.get("/languages")
async def get_supported_languages():
    """Get list of supported languages."""
    return {
        "supported_languages": SUPPORTED_LANGUAGES
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)