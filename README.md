# LLM Inference Gateway

Async, OpenAI-compatible gateway that sits in front of LLM inference engines
(vLLM, Ollama) and adds **semantic caching** and **dynamic batching**.

> 🚧 Work in progress. Full documentation, architecture diagram and
> benchmarks will be added as the project is completed.

## Planned features

- OpenAI-compatible `POST /v1/chat/completions`
- Semantic cache (sentence-transformers + Qdrant), cosine threshold 0.90
- Dynamic batching (`asyncio.Queue`, size/timeout flush)
- Prometheus metrics, Docker Compose, GitHub Actions CI

## Quick start

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
uvicorn gateway.main:app --reload
curl http://localhost:8000/health
```

## License

MIT
