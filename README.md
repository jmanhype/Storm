# Storm

A CLI tool that generates long-form articles on a given topic by running a multi-step pipeline: research, perspective generation, simulated multi-perspective conversation, outline creation, and section-by-section article writing. Built on DSPy and uses Claude 3 Haiku via OpenRouter.

## Status

Prototype. The codebase has multiple versions of the core logic (`Storm.py`, `storm.py`, `storm_enhanced.py`, `1.py`, `2.py`) without clear indication of which is canonical. There is a `self-improve.yml` GitHub Actions workflow committed but it is a generic template, not Storm-specific. Log files (`azure_openai_usage.log`, `openai_usage.log`) and a 2.4 MB Jupyter notebook are checked into the repo.

## How it works

1. **Research** -- Fetches Wikipedia links and tables of contents for context
2. **Perspectives** -- Generates multiple viewpoints on the topic
3. **Conversation** -- Simulates Q&A between perspectives to build depth
4. **Outline** -- Creates a structured article outline from the conversation
5. **Writing** -- Generates the article section by section with word count targeting

## Usage

```bash
export OPENROUTER_API_KEY='your-key'
pip install dspy-ai pydantic
python storm "Quantum Computing"
python storm "Topic" --words 1500
python storm "Topic" -o output.md --format md
```

## Dependencies

| Package | Purpose |
|---|---|
| dspy-ai | LLM pipeline framework |
| pydantic | Data validation |
| requests | HTTP calls |
| transformers | QA model (used in Storm.py evaluation) |
| spacy | NLP (used in Storm.py) |
| sentence-transformers | Semantic similarity scoring |
| textstat | Readability metrics |

## Files

| File | Lines | Purpose |
|---|---|---|
| storm (executable) | ~500 | CLI entry point with argparse |
| storm.py | ~250 | Core pipeline with DSPy signatures |
| Storm.py | ~400 | Alternate version with evaluation metrics |
| storm_enhanced.py | ~400 | Another variant |
| main.py | ~70 | Simplified orchestrator |
| app.py | ~130 | Web/app interface |
| delphi.py | ~450 | Separate prediction/oracle module |
| grok_storm.py | ~200 | Grok API variant |

## Limitations

- Multiple overlapping entry points; unclear which to use
- Wikipedia is the only research source
- No rate limiting or retry logic for API calls
- API key is hardcoded in `Storm.py` (placeholder value)
- Log files and a large notebook are committed to the repo
- No tests that run without API keys

## License

None specified.