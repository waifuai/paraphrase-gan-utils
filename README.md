# Paraphrase Generation System

A comprehensive, production-ready paraphrase generation system with advanced features including batch processing, quality evaluation, caching, rate limiting, and REST API support.

## 🚀 Key Features

- **Multiple AI Providers**: Support for Google Gemini and OpenRouter providers
- **Batch Processing**: Efficiently process multiple texts with concurrent execution
- **Quality Evaluation**: Comprehensive paraphrase quality scoring with lexical, structural, and semantic metrics
- **Intelligent Caching**: Redis/memory caching to reduce API calls and improve performance
- **Rate Limiting**: Built-in rate limiting to respect API quotas and prevent abuse
- **REST API**: Full REST API with interactive documentation
- **Error Handling**: Robust error handling with detailed logging and recovery mechanisms
- **Async Support**: Asynchronous processing for high-throughput scenarios
- **Interactive Mode**: CLI interactive mode for quick testing and experimentation

## Project Structure

```
.
├── data/
│   ├── custom_train.tsv      # Custom training data (tab-separated: source<TAB>target) - Used for structure reference
│   └── custom_eval.tsv       # Custom evaluation data (tab-separated: source<TAB>target) - Used for structure reference
├── src/
│   ├── api.py                # FastAPI REST endpoints with automatic docs
│   ├── batch_processor.py    # High-performance batch processing with concurrency
│   ├── cache.py              # Redis/memory caching system with TTL support
│   ├── config.py             # Comprehensive configuration management
│   ├── evaluation.py         # Paraphrase quality evaluation with multiple metrics
│   ├── exceptions.py         # Custom exception hierarchy with detailed error info
│   ├── gemini_api.py         # Wrapper around Google GenAI SDK Client
│   ├── logging_config.py     # Structured logging with performance monitoring
│   ├── main.py               # Enhanced CLI with multiple operation modes
│   ├── provider_facade.py    # Provider abstraction with fallback mechanisms
│   ├── rate_limiter.py       # Multi-provider rate limiting system
│   ├── data_processing/      # Legacy data processing modules
│   └── provider_openrouter.py # OpenRouter API integration
├── requirements.txt          # Runtime dependencies (includes FastAPI, caching, ML libraries)
├── requirements-dev.txt      # Development and testing dependencies
├── logs/                     # Application logs (created automatically)
├── .venv/                    # Virtual environment (created by uv)
├── model_output/             # Default directory for output (less critical now)
└── README.md
```

## Installation

### Prerequisites
- Python 3.8+
- API keys for your preferred providers (Gemini or OpenRouter)

### Setup Steps

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

2.  **Obtain API Keys:**
    *   **Google Gemini**: Get an API key from [Google AI Studio](https://aistudio.google.com/app/apikey)
    *   **OpenRouter**: Get an API key from [OpenRouter](https://openrouter.ai/)

3.  **Configure API Keys:**
    ```bash
    # Option 1: Environment variables (recommended)
    export GEMINI_API_KEY="your-gemini-key"
    # or
    export OPENROUTER_API_KEY="your-openrouter-key"

    # Option 2: Key files
    echo "your-gemini-key" > ~/.api-gemini
    echo "your-openrouter-key" > ~/.api-openrouter
    ```

4.  **Create Virtual Environment:**
    ```bash
    python -m uv venv .venv
    .venv/Scripts/python.exe -m ensurepip
    .venv/Scripts/python.exe -m pip install uv
    ```

5.  **Install Dependencies:**
    ```bash
    # Install runtime dependencies
    .venv/Scripts/python.exe -m uv pip install -r requirements.txt

    # Install development dependencies (optional)
    .venv/Scripts/python.exe -m uv pip install -r requirements-dev.txt
    ```

6.  **Setup Optional Components:**
    ```bash
    # For Redis caching (optional)
    # Install and start Redis server, then set REDIS_URL=redis://localhost:6379/0

    # For sentence transformers (optional, for semantic evaluation)
    .venv/Scripts/python.exe -m uv pip install sentence-transformers
    ```

## Custom Dataset Format

The custom dataset files (`data/custom_train.tsv`, `data/custom_eval.tsv`) are included for historical context and potential future use, but are not directly used by the current Gemini API-based paraphrase generation logic. They contain tab-separated pairs of sentences, where the first column is the source sentence and the second column is the target paraphrase. Example:

```tsv
Original sentence one.<TAB>Paraphrased sentence one.
Original sentence two.<TAB>Paraphrased sentence two.
```

## Usage

The system supports multiple operation modes for different use cases.

### Command Line Interface

#### Single Paraphrase Generation
```bash
.venv/Scripts/python.exe -m src.main --mode cli --decode_input "This is the sentence to paraphrase."
```

#### Batch Processing
```bash
# Process multiple texts from a file
.venv/Scripts/python.exe -m src.main --mode batch --batch_input input.txt --batch_output results.txt

# Process a single text in batch mode
.venv/Scripts/python.exe -m src.main --mode batch --decode_input "Text to paraphrase"
```

#### Interactive Mode
```bash
.venv/Scripts/python.exe -m src.main --mode interactive
```

#### Quality Evaluation
```bash
.venv/Scripts/python.exe -m src.main --mode evaluate \
    --evaluate_original "Original text" \
    --evaluate_paraphrase "Generated paraphrase"
```

### REST API Server

Start the REST API server with automatic documentation:

```bash
.venv/Scripts/python.exe -m src.main --mode api --port 8000
```

The API will be available at:
- **Main API**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

#### API Endpoints

**POST /paraphrase**
Generate a single paraphrase
```json
{
  "text": "This is the sentence to paraphrase.",
  "provider": "openrouter",
  "model": "anthropic/claude-3-sonnet"
}
```

**POST /paraphrase/batch**
Generate multiple paraphrases
```json
{
  "texts": ["Text 1", "Text 2", "Text 3"],
  "provider": "gemini"
}
```

**POST /evaluate**
Evaluate paraphrase quality
```json
{
  "original": "Original text",
  "paraphrase": "Generated paraphrase",
  "include_semantic": true
}
```

### Python API Usage

```python
from src.main import generate_paraphrase, paraphrase_batch
from src.evaluation import evaluate_paraphrase
from src.config import setup_system

# Setup system components
setup_system()

# Single paraphrase
result = generate_paraphrase("This is a test sentence.")
print(f"Paraphrase: {result}")

# Batch processing
texts = ["Text 1", "Text 2", "Text 3"]
results = paraphrase_batch(texts)
print(f"Results: {results}")

# Quality evaluation
evaluation = evaluate_paraphrase(
    original="Original text",
    paraphrase="Generated paraphrase"
)
print(f"Quality score: {evaluation['overall_score']}")
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GEMINI_API_KEY` | Google Gemini API key | None |
| `GOOGLE_API_KEY` | Alternative Gemini API key | None |
| `OPENROUTER_API_KEY` | OpenRouter API key | None |
| `REDIS_URL` | Redis connection URL | `redis://localhost:6379/0` |
| `LOG_LEVEL` | Logging level | `INFO` |

### Configuration File

Advanced configuration can be modified in `src/config.py`:

```python
# API Settings
API_HOST = "0.0.0.0"
API_PORT = 8000
API_DEBUG = False

# Cache Settings
CACHE_TYPE = "redis"  # "redis" or "memory"
CACHE_TTL = 3600  # 1 hour default

# Rate Limiting
RATE_LIMIT_REQUESTS_PER_MINUTE = 60

# Batch Processing
BATCH_MAX_SIZE = 10
BATCH_MAX_WORKERS = 4
```

## Advanced Features

### Quality Evaluation Metrics

The system provides comprehensive quality evaluation:

- **Lexical Similarity**: Word overlap, Jaccard similarity, cosine similarity
- **Structural Similarity**: Length ratio, sentence count differences
- **Semantic Similarity**: BERT-based sentence embeddings (when available)
- **Overall Quality Score**: Weighted combination of all metrics

### Caching System

- **Memory Cache**: Fast in-process caching with TTL support
- **Redis Cache**: Distributed caching for multi-instance deployments
- **Automatic Fallback**: Falls back to memory cache if Redis unavailable
- **Cache Warming**: Intelligent cache population for frequently used texts

### Rate Limiting

- **Provider-Specific Limits**: Different limits for Gemini vs OpenRouter
- **Adaptive Throttling**: Automatically adjusts based on API response headers
- **Token-Based Limiting**: Supports both request count and token-based limits
- **Exponential Backoff**: Smart retry logic with exponential backoff

### Batch Processing

- **Concurrent Execution**: Multi-threaded processing for high throughput
- **Progress Tracking**: Real-time progress updates for long-running jobs
- **Error Resilience**: Continues processing even if some items fail
- **Result Aggregation**: Comprehensive statistics and error reporting

### Error Handling

- **Custom Exception Hierarchy**: Specific exceptions for different error types
- **Structured Logging**: Detailed error logging with context
- **Graceful Degradation**: Continues operation when non-critical components fail
- **Recovery Mechanisms**: Automatic retry with exponential backoff

## Docker Deployment

Create a `Dockerfile` for containerized deployment:

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ ./src/
COPY data/ ./data/

EXPOSE 8000
CMD ["python", "-m", "src.main", "--mode", "api", "--port", "8000"]
```

## Monitoring and Observability

The system includes comprehensive monitoring:

- **Structured Logging**: JSON-formatted logs with correlation IDs
- **Performance Metrics**: Request latency, throughput, error rates
- **Health Checks**: Built-in health endpoints for load balancers
- **Cache Statistics**: Hit rates, memory usage, eviction counts
- **Rate Limit Monitoring**: Current usage vs limits

## Development

### Running Tests

```bash
# Install development dependencies
.venv/Scripts/python.exe -m uv pip install -r requirements-dev.txt

# Run tests
.venv/Scripts/python.exe -m pytest tests/

# Run with coverage
.venv/Scripts/python.exe -m pytest --cov=src tests/
```

### Code Quality

```bash
# Format code
.venv/Scripts/python.exe -m black src/

# Type checking
.venv/Scripts/python.exe -m mypy src/

# Linting
.venv/Scripts/python.exe -m flake8 src/
```

## Troubleshooting

### Common Issues

1. **API Key Errors**: Ensure API keys are properly configured in environment variables or key files
2. **Rate Limiting**: System will automatically handle rate limits with backoff - check logs for details
3. **Memory Issues**: Reduce batch size or enable Redis caching for large workloads
4. **Performance**: Use Redis instead of memory cache for multi-process deployments

### Debug Mode

Enable verbose logging for troubleshooting:

```bash
.venv/Scripts/python.exe -m src.main --mode api --verbose
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## License

This project is licensed under the MIT-0 License. See the LICENSE file for details.
