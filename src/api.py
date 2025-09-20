# src/api.py
from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
import time
import uuid

from src.logging_config import get_logger, with_error_handling, with_performance_logging
from src.exceptions import ValidationError, APIError
from src.batch_processor import get_batch_processor, paraphrase_batch
from src.evaluation import evaluate_paraphrase, evaluate_paraphrase_batch
from src.cache import get_cache
from src.rate_limiter import get_rate_limiter

logger = get_logger("api")

# Pydantic models
class ParaphraseRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000, description="Text to paraphrase")
    provider: Optional[str] = Field("openrouter", description="Provider to use")
    model: Optional[str] = Field(None, description="Model override")
    skip_cache: bool = Field(False, description="Skip cache")

    @validator('text')
    def validate_text(cls, v):
        if not v.strip():
            raise ValueError('Text cannot be empty or whitespace only')
        return v.strip()

class BatchParaphraseRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=100, description="Texts to paraphrase")
    provider: Optional[str] = Field("openrouter", description="Provider to use")
    model: Optional[str] = Field(None, description="Model override")
    skip_cache: bool = Field(False, description="Skip cache")

    @validator('texts')
    def validate_texts(cls, v):
        if not v:
            raise ValueError('At least one text is required')
        for i, text in enumerate(v):
            if not text.strip():
                raise ValueError(f'Text at index {i} cannot be empty')
            if len(text) > 10000:
                raise ValueError(f'Text at index {i} is too long (max 10000 chars)')
        return [text.strip() for text in v]

class EvaluationRequest(BaseModel):
    original: str = Field(..., min_length=1, max_length=10000, description="Original text")
    paraphrase: str = Field(..., min_length=1, max_length=10000, description="Paraphrased text")
    include_semantic: bool = Field(True, description="Include semantic similarity")

class BatchEvaluationRequest(BaseModel):
    originals: List[str] = Field(..., description="Original texts")
    paraphrases: List[str] = Field(..., description="Paraphrased texts")
    include_semantic: bool = Field(True, description="Include semantic similarity")

    @validator('originals', 'paraphrases')
    def validate_lists(cls, v, field):
        if not v:
            raise ValueError(f'{field.name} cannot be empty')
        for i, text in enumerate(v):
            if not text.strip():
                raise ValueError(f'{field.name} at index {i} cannot be empty')
        return [text.strip() for text in v]

    @validator('paraphrases')
    def validate_lengths(cls, v, values):
        if 'originals' in values and len(v) != len(values['originals']):
            raise ValueError('Originals and paraphrases must have the same length')
        return v

class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    total_items: int
    successful: int
    failed: int
    success_rate: float
    results: Optional[List[Optional[str]]]
    errors: List[Dict[str, Any]]
    created_at: float
    completed_at: Optional[float]
    processing_time: Optional[float]

# Create FastAPI app
app = FastAPI(
    title="Paraphrase Generation API",
    description="A comprehensive API for generating and evaluating paraphrases using multiple AI providers",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
jobs = {}

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Paraphrase Generation API", "version": "2.0.0"}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": "2.0.0"
    }

@app.post("/paraphrase", response_model=Dict[str, Any])
@with_error_handling("api")
@with_performance_logging("api")
async def generate_paraphrase(request: ParaphraseRequest):
    """Generate a paraphrase for the given text."""
    try:
        # Rate limiting
        rate_limiter = get_rate_limiter()
        rate_limiter.wait_if_needed(request.provider or "openrouter")

        # Import here to avoid circular imports
        from src.provider_facade import generate_paraphrase as generate_func

        result = generate_func(
            text=request.text,
            provider=request.provider,
            model=request.model,
            skip_cache=request.skip_cache
        )

        if not result:
            raise HTTPException(status_code=500, detail="Failed to generate paraphrase")

        return {
            "original": request.text,
            "paraphrase": result,
            "provider": request.provider or "openrouter",
            "model": request.model,
            "cached": not request.skip_cache and get_cache().exists(
                f"paraphrase:{hash(request.text + str(request.provider) + str(request.model))}"
            )
        }

    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Paraphrase generation failed", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/paraphrase/batch", response_model=Dict[str, Any])
@with_error_handling("api")
@with_performance_logging("api")
async def generate_paraphrase_batch(request: BatchParaphraseRequest):
    """Generate paraphrases for multiple texts."""
    try:
        # Rate limiting (more tokens for batch)
        rate_limiter = get_rate_limiter()
        rate_limiter.wait_if_needed(
            request.provider or "openrouter",
            tokens=min(len(request.texts), 10)  # Cap at 10 tokens
        )

        results = paraphrase_batch(
            texts=request.texts,
            # Note: skip_cache not directly supported in batch function
        )

        successful = sum(1 for r in results if r is not None)
        failed = len(results) - successful

        return {
            "results": [
                {
                    "original": orig,
                    "paraphrase": para,
                    "success": para is not None
                }
                for orig, para in zip(request.texts, results)
            ],
            "summary": {
                "total": len(request.texts),
                "successful": successful,
                "failed": failed,
                "success_rate": successful / len(request.texts) if request.texts else 0
            },
            "provider": request.provider or "openrouter",
            "model": request.model
        }

    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Batch paraphrase generation failed", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/paraphrase/batch/async", response_model=Dict[str, Any])
@with_error_handling("api")
async def generate_paraphrase_batch_async(
    request: BatchParaphraseRequest,
    background_tasks: BackgroundTasks
):
    """Start an asynchronous batch paraphrase job."""
    try:
        processor = get_batch_processor()
        job_id = processor.create_job(request.texts)

        # Start processing in background
        background_tasks.add_task(processor.start_job, job_id)

        return {
            "job_id": job_id,
            "status": "started",
            "total_items": len(request.texts),
            "message": "Job started. Use /jobs/{job_id} to check status."
        }

    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Async batch job creation failed", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
@with_error_handling("api")
async def get_job_status(job_id: str):
    """Get the status of a batch job."""
    processor = get_batch_processor()
    job_result = processor.get_job_results(job_id)

    if not job_result:
        raise HTTPException(status_code=404, detail="Job not found")

    return JobStatusResponse(**job_result)

@app.post("/evaluate", response_model=Dict[str, Any])
@with_error_handling("api")
@with_performance_logging("api")
async def evaluate_single_paraphrase(request: EvaluationRequest):
    """Evaluate the quality of a single paraphrase."""
    try:
        evaluation = evaluate_paraphrase(
            original=request.original,
            paraphrase=request.paraphrase,
            include_semantic=request.include_semantic
        )

        return {
            "original": request.original,
            "paraphrase": request.paraphrase,
            "evaluation": evaluation
        }

    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Evaluation failed", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/evaluate/batch", response_model=List[Dict[str, Any]])
@with_error_handling("api")
@with_performance_logging("api")
async def evaluate_batch_paraphrases(request: BatchEvaluationRequest):
    """Evaluate the quality of multiple paraphrases."""
    try:
        evaluations = evaluate_paraphrase_batch(
            originals=request.originals,
            paraphrases=request.paraphrases,
            include_semantic=request.include_semantic
        )

        return [
            {
                "original": orig,
                "paraphrase": para,
                "evaluation": eval_result
            }
            for orig, para, eval_result in zip(request.originals, request.paraphrases, evaluations)
        ]

    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Batch evaluation failed", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/cache/stats", response_model=Dict[str, Any])
@with_error_handling("api")
async def get_cache_stats():
    """Get cache statistics."""
    try:
        cache = get_cache()
        # Note: This is a simplified version. In practice, you might want
        # more detailed cache statistics depending on the cache implementation.
        return {
            "status": "Cache system active",
            "type": type(cache).__name__
        }
    except Exception as e:
        logger.error("Cache stats retrieval failed", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/providers", response_model=Dict[str, Any])
@with_error_handling("api")
async def get_available_providers():
    """Get information about available providers."""
    return {
        "providers": ["openrouter", "gemini"],
        "default": "openrouter",
        "features": {
            "openrouter": ["paraphrase", "classification"],
            "gemini": ["paraphrase"]
        }
    }

@app.get("/metrics", response_model=Dict[str, Any])
@with_error_handling("api")
async def get_metrics():
    """Get API usage metrics."""
    try:
        # This is a placeholder. In a real implementation, you would
        # collect and return actual metrics from your monitoring system.
        return {
            "requests_total": 0,
            "requests_successful": 0,
            "requests_failed": 0,
            "average_response_time": 0.0,
            "cache_hit_rate": 0.0
        }
    except Exception as e:
        logger.error("Metrics retrieval failed", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

# Error handlers
@app.exception_handler(ValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse(
        status_code=400,
        content={
            "error": "Validation Error",
            "message": str(exc),
            "type": "validation_error"
        }
    )

@app.exception_handler(APIError)
async def api_exception_handler(request, exc):
    return JSONResponse(
        status_code=502,
        content={
            "error": "API Error",
            "message": str(exc),
            "type": "api_error",
            "provider": getattr(exc, 'provider', 'unknown')
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error("Unhandled exception", error=str(exc), exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred",
            "type": "internal_error"
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)