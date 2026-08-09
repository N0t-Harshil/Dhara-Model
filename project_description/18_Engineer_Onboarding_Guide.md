# Engineer Onboarding Guide

## Repository Layout

```
specialized-coding-model/
├── config.yaml              # Primary training config (7B, 4x A100)
├── config_foundation.yaml   # Foundation pretraining config (160M, 1x A100)
├── config_small.yaml        # Debug config (173M, 1x GPU)
├── main.py                  # CLI entrypoint (8 commands)
├── src/
│   ├── data/                # Data pipeline (MOST POLISHED)
│   │   ├── registry.py      # 55-dataset registry
│   │   ├── streaming.py     # Dataset loading with fallbacks
│   │   ├── drivers.py       # File/Script/Local/Streaming driver families + builder cache
│   │   ├── metadata_cache.py # Metadata records, fingerprints, URI rewriting
│   │   ├── shards.py        # Shard-parallel progress store
│   │   ├── pipeline.py      # Pipeline orchestration (2250 lines)
│   │   ├── doc_builder.py   # 18-source web scraper (808 lines)
│   │   ├── quality.py       # Quality scoring + dedup
│   │   ├── ast_filter.py    # Code quality via AST parsing
│   │   ├── function_sampler.py # Function-level code sampling
│   │   ├── sanity.py        # Sanity checks
│   │   └── health_reporter.py # Dataset health stats
│   ├── methos_v3/           # Core model (22 files)
│   ├── nslt/                # Alternative model (10 files)
│   ├── config/              # Pydantic schema (666 lines)
│   ├── training/            # Training orchestration
│   ├── models/              # Model factory
│   ├── alignment/           # DPO + Constitutional AI
│   ├── evaluation/          # Benchmarks + safety
│   ├── infrastructure/      # Distributed + tracking
│   └── utils/               # Logging + reproducibility
├── scripts/                 # 13 utility/validation scripts
├── tests/                   # 12 test files
├── data/docs/               # Scraped documentation datasets
├── notebooks/               # Jupyter notebooks
└── models/tokenizer/        # Tokenizer files (empty - downloaded)
```

## Day 1 Checklist

1. [ ] Read README.md
2. [ ] Read ARCHITECTURE.md
3. [ ] Read PROJECT_DOCUMENTATION.md (config field reference)
4. [ ] Review this onboarding guide
5. [ ] Install dependencies: `pip install -r requirements.txt`
6. [ ] Run config validation: `python main.py config-validate --config config_foundation.yaml`
7. [ ] Run dataset verification: `python scripts/verify_datasets.py --token $HF_TOKEN`
8. [ ] Run production validation: `python scripts/production_validation.py --token $HF_TOKEN --smoke-steps 10`
9. [ ] Run tests: `python -m pytest tests/ -v`

## Development Workflow

1. **Setup**: Works on Windows (CPU, dev server) and Linux (GPU, training)
2. **Edit**: Change files on your local machine
3. **Test**: Run `python -m pytest tests/` for Python tests
4. **Validate**: After data changes, run validation scripts
5. **Sync**: Use rsync to push to GPU server:
   ```
   rsync -avz src/data/*.py scripts/*.py $SERVER:~/work/specialized-coding-model/src/data/
   ```
6. **Train**: `python main.py full-training --config config_foundation.yaml` or `python main.py pretrain --config config_foundation.yaml`

## Important Files by Task

**To add a dataset**:
- Edit `src/data/registry.py` → add entry in appropriate _register_* function
- Test: `python scripts/verify_datasets.py --token $HF_TOKEN`

**To add a documentation scraper**:
- Edit `src/data/doc_builder.py` → create new scraper class (inherit DocScraper or SphinxScraper)
- Add to SCRAPERS dict in doc_builder.py
- Test: `python src/data/doc_builder.py --sources [name] --max-per-source 50`

**To add a model architecture**:
- Create files in `src/models/` or `src/methos_v3/`
- Register in `src/models/factory.py`
- Add config parameters in `src/config/schema.py`
- Add config in YAML config files

**To run training**:
- Foundation: `python main.py full-training --config config_foundation.yaml` (1x A100)
- Full: `bash scripts/train_4gpu.sh` (4x A100 via torchrun)

**To verify changes**:
- Run tests: `python -m pytest tests/ -v -k [test_name]`
- Run validation: `python scripts/production_validation.py --token $HF_TOKEN`

## Best Practices

1. **Follow existing patterns**: The codebase has consistent patterns (logging, type hints, error handling). Match them.
2. **Always use logger.exception()**: All exception handlers should use `logger.exception()` for full traceback.
3. **Test before PR**: Run relevant tests before submitting changes.
4. **Update config docs**: When adding config parameters, update PROJECT_DOCUMENTATION.md.
5. **One fix, one file**: Keep changes minimal and focused.
6. **Don't break the data pipeline**: The data pipeline is the most critical subsystem. Always run verify_datasets.py after data changes.
7. **Metadata preservation**: Never strip metadata from dataset samples. Always yield `{**sample, "text": text}`.
8. **Use registry mode**: When `use_registry: true`, the registry is the single source of truth. Never bypass it.

## Common Tasks

### Adding a new dataset to the registry

```python
# In registry.py _register_code() or appropriate category function
registry.register(DatasetInfo(
    path="new/dataset",
    name="subset-name",       # optional
    category="code",
    weight=round(total * 0.05, 4),
    quality_score=0.90,
    domain="ml",
    text_fields=["text"],
    license="MIT",
    fallbacks=["fallback/dataset"],
    priority=10,
))
```

### Adding a new scraper

```python
class MyNewDocScraper(SphinxScraper):
    SOURCE_NAME = "my-new-docs"
    BASE_URL = "https://docs.example.com/"
    LANGUAGE = "python"
    CATEGORY = "docs"
    URL_PATTERNS = ["guide/", "api/", "tutorial/"]
    EXCLUDE_PATTERNS = [".pdf", "_sources", "search.html"]

# Then add to SCRAPERS dict
```

## Debugging Quick Reference

| Symptom | Likely Cause | First Check |
|---------|-------------|-------------|
| Dataset won't load | Gated or missing | verify_datasets.py output |
| Scraper returns 0 pages | Site redesign or Cloudflare | Run scraper individually |
| Training loss NaN | Learning rate too high or corrupt data | Check gradient norms |
| OOM during training | Batch size too large | Reduce batch_size or max_seq_length |
| Checkpoint won't load | Config mismatch | Compare save/load configs |
| Slow data loading | Network-bound (streaming) | Enable cache_dir |