"""
Batch processing system for paraphrase generation.

This module provides high-performance batch processing capabilities for generating
multiple paraphrases concurrently. It includes job management, async processing,
caching integration, and comprehensive error handling.

Key Features:
- Concurrent batch processing with thread pool execution
- Asynchronous processing support
- Job-based processing with status tracking
- Retry logic with exponential backoff
- Caching integration for improved performance
- Progress tracking and error reporting
- Memory-efficient processing for large datasets
"""
# src/batch_processor.py
import asyncio
import time
from typing import List, Dict, Any, Optional, Callable, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from src.logging_config import get_logger
from src.exceptions import BatchProcessingError, ValidationError
from src.cache import cached

logger = get_logger("batch_processor")

@dataclass
class BatchJob:
    """Represents a single batch processing job."""
    id: str
    items: List[str]
    results: List[Optional[str]] = field(default_factory=list)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "pending"  # pending, processing, completed, failed
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def success_count(self) -> int:
        return len([r for r in self.results if r is not None])

    @property
    def error_count(self) -> int:
        return len(self.errors)

    @property
    def total_count(self) -> int:
        return len(self.items)

    @property
    def success_rate(self) -> float:
        return self.success_count / self.total_count if self.total_count > 0 else 0.0

class BatchProcessor:
    """High-performance batch processor for paraphrase generation."""

    def __init__(
        self,
        process_func: Callable[[str], str],
        max_batch_size: int = 10,
        max_workers: int = 4,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        self.process_func = process_func
        self.max_batch_size = max_batch_size
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self._jobs: Dict[str, BatchJob] = {}

    def validate_input(self, texts: List[str]) -> None:
        """Validate input texts."""
        if not texts:
            raise ValidationError("texts", texts, "Input list cannot be empty")

        if len(texts) > 1000:
            raise ValidationError("texts", len(texts), "Too many texts (max 1000)")

        for i, text in enumerate(texts):
            if not isinstance(text, str):
                raise ValidationError(f"texts[{i}]", text, "Must be a string")
            if not text.strip():
                raise ValidationError(f"texts[{i}]", text, "Cannot be empty")
            if len(text) > 10000:
                raise ValidationError(f"texts[{i}]", len(text), "Text too long (max 10000 chars)")

    def create_job(self, texts: List[str]) -> str:
        """Create a new batch job."""
        import uuid
        job_id = str(uuid.uuid4())

        job = BatchJob(
            id=job_id,
            items=texts,
            results=[None] * len(texts)
        )

        self._jobs[job_id] = job
        logger.info("Created batch job", job_id=job_id, item_count=len(texts))
        return job_id

    def get_job(self, job_id: str) -> Optional[BatchJob]:
        """Get job by ID."""
        return self._jobs.get(job_id)

    def process_sync(self, texts: List[str]) -> List[Optional[str]]:
        """Process texts synchronously."""
        self.validate_input(texts)

        if len(texts) <= self.max_batch_size:
            return self._process_batch(texts)
        else:
            # Split into batches
            results = []
            for i in range(0, len(texts), self.max_batch_size):
                batch = texts[i:i + self.max_batch_size]
                batch_results = self._process_batch(batch)
                results.extend(batch_results)
            return results

    async def process_async(self, texts: List[str]) -> List[Optional[str]]:
        """Process texts asynchronously."""
        self.validate_input(texts)

        if len(texts) <= self.max_batch_size:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self.executor,
                self._process_batch,
                texts
            )
        else:
            # Process batches concurrently
            tasks = []
            loop = asyncio.get_event_loop()

            for i in range(0, len(texts), self.max_batch_size):
                batch = texts[i:i + self.max_batch_size]
                task = loop.run_in_executor(
                    self.executor,
                    self._process_batch,
                    batch
                )
                tasks.append(task)

            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Flatten results and handle exceptions
            results = []
            for batch_result in batch_results:
                if isinstance(batch_result, Exception):
                    logger.error("Batch processing error", error=str(batch_result))
                    results.extend([None] * self.max_batch_size)
                else:
                    results.extend(batch_result)

            return results[:len(texts)]  # Trim to exact size

    def _process_batch(self, texts: List[str]) -> List[Optional[str]]:
        """Process a single batch of texts."""
        results = []

        # Submit all tasks
        future_to_index = {
            self.executor.submit(self._process_with_retry, text): i
            for i, text in enumerate(texts)
        }

        # Collect results in order
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            try:
                result = future.result()
                results.append((index, result))
            except Exception as e:
                logger.error(
                    "Text processing failed",
                    index=index,
                    text=texts[index][:100],
                    error=str(e)
                )
                results.append((index, None))

        # Sort by index and extract results
        results.sort(key=lambda x: x[0])
        return [result for _, result in results]

    def _process_with_retry(self, text: str) -> Optional[str]:
        """Process single text with retry logic."""
        for attempt in range(self.max_retries + 1):
            try:
                result = self.process_func(text)
                return result
            except Exception as e:
                if attempt == self.max_retries:
                    logger.error(
                        "Max retries exceeded",
                        text=text[:100],
                        attempt=attempt,
                        error=str(e)
                    )
                    raise

                logger.warning(
                    "Processing attempt failed, retrying",
                    text=text[:100],
                    attempt=attempt,
                    error=str(e)
                )

                if self.retry_delay > 0:
                    time.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff

        return None

    def start_job(self, job_id: str) -> bool:
        """Start processing a job asynchronously."""
        job = self._jobs.get(job_id)
        if not job:
            return False

        if job.status != "pending":
            return False

        job.status = "processing"

        # Start processing in background
        def process_job():
            try:
                results = self.process_sync(job.items)
                job.results = results
                job.status = "completed"
                job.completed_at = time.time()

                success_count = sum(1 for r in results if r is not None)
                logger.info(
                    "Job completed",
                    job_id=job_id,
                    total=job.total_count,
                    successful=success_count,
                    failed=job.total_count - success_count
                )
            except Exception as e:
                job.status = "failed"
                job.completed_at = time.time()
                job.errors.append({
                    "error": str(e),
                    "timestamp": time.time()
                })
                logger.error("Job failed", job_id=job_id, error=str(e))

        self.executor.submit(process_job)
        return True

    def get_job_results(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job results if completed."""
        job = self._jobs.get(job_id)
        if not job:
            return None

        return {
            "job_id": job.id,
            "status": job.status,
            "total_items": job.total_count,
            "successful": job.success_count,
            "failed": job.error_count,
            "success_rate": job.success_rate,
            "results": job.results if job.status == "completed" else None,
            "errors": job.errors,
            "created_at": job.created_at,
            "completed_at": job.completed_at,
            "processing_time": (
                job.completed_at - job.created_at
                if job.completed_at else None
            )
        }

class ParaphraseBatchProcessor(BatchProcessor):
    """Batch processor specifically for paraphrase generation."""

    def __init__(self, **kwargs):
        from src.provider_facade import generate_paraphrase
        super().__init__(process_func=generate_paraphrase, **kwargs)

    @cached(ttl=3600, key_prefix="batch_paraphrase")
    def process_with_cache(self, texts: List[str]) -> List[Optional[str]]:
        """Process with caching enabled."""
        return self.process_sync(texts)

# Global batch processor instance
_batch_processor_instance = None

def get_batch_processor() -> ParaphraseBatchProcessor:
    """Get global batch processor instance."""
    global _batch_processor_instance

    if _batch_processor_instance is None:
        _batch_processor_instance = ParaphraseBatchProcessor(
            max_batch_size=5,  # Conservative batch size for API limits
            max_workers=2,
            max_retries=3,
            retry_delay=1.0
        )
        logger.info("Initialized global batch processor")

    return _batch_processor_instance

def paraphrase_batch(
    texts: List[str],
    use_cache: bool = True
) -> List[Optional[str]]:
    """Convenience function for batch paraphrase generation."""
    processor = get_batch_processor()

    if use_cache:
        return processor.process_with_cache(texts)
    else:
        return processor.process_sync(texts)

async def paraphrase_batch_async(
    texts: List[str],
    use_cache: bool = True
) -> List[Optional[str]]:
    """Async convenience function for batch paraphrase generation."""
    processor = get_batch_processor()

    if use_cache:
        # Note: Caching doesn't work well with async, so we'll skip it
        return await processor.process_async(texts)
    else:
        return await processor.process_async(texts)