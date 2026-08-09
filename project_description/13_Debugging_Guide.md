# Debugging Guide

## Common Failures & Diagnosis

### 1. Dataset Loading Failures
- **Symptom**: `datasets` library raises `GatedDataset` error
- **Cause**: Dataset requires HF access token or approval
- **Check**: Is HF_TOKEN set? Has access been granted at huggingface.co/datasets/{path}?
- **Fix**: Set HF_TOKEN env var or token in config; request access for gated datasets
- **Log location**: Console output from streaming.py line ~40 `_handle_load_error()`

### 2. Dataset Not Found (404)
- **Symptom**: `bigcode/the-stack-v2-dedup does not exist` error
- **Cause**: URL encoding issue with `#` in config name (C# case)
- **Fix**: Remove C# from STACK_V2_LANGUAGES (already done)
- **Verify**: Run `python scripts/verify_datasets.py --token $HF_TOKEN`

### 3. Documentation Scraper Failures
- **Symptom**: "0 pages" scraped for a source
- **Causes**: 
  - Cloudflare challenge (PyTorch)
  - Site redesign (RFC, PostgreSQL)
  - JavaScript-rendered navigation (Docker, MDN)
  - Timeout (server too slow)
- **Check**: Run scraper individually with `--sources [name]`
- **Verify**: Check JSONL output files in data/docs/{source_name}/documents.jsonl

### 4. Packing Validation Failures
- **Symptom**: Padding > 5% or quality score too low
- **Check**: Phase 6 output in production report
- **Diagnosis**: 
  - Too many short sequences → increase packing efficiency
  - Too many long sequences → increase max_seq_length or improve window sampling
- **Fix**: Adjust max_seq_length or improve random_window_sample

### 5. Checkpoint Load Failures
- **Symptom**: Model fails to resume from checkpoint
- **Check**: Phase 8 output for checkpoint_found and resume_return_code
- **Causes**: 
  - Incompatible config change between save and resume
  - Missing optimizer state keys
  - FSDP world size mismatch
- **Fix**: Ensure identical config, world size, and FSDP settings

### 6. Training Divergence
- **Symptom**: Loss goes to NaN or infinity
- **Check**: Gradient norms, learning rate, mixed precision settings
- **Diagnosis**:
  - Try max_grad_norm lower (0.5 instead of 1.0)
  - Reduce learning rate
  - Disable mixed precision (use_full_precision)
  - Check for corrupted data samples
- **Verify**: Re-run with --smoke-steps 50 to isolate

### Where Logs Are Stored
- Console: stdout/stderr (always visible)
- File: configurable via cfg.output.log_dir (default: logs/)
- Experiment tracker: wandb/mlflow/tensorboard (configurable in cfg.tracking)

### How to Reproduce Issues
- Check `scripts/final_verify.sh` for the standard reproduction sequence
- Each Phase in production_validation.py can be run independently
- For dataset issues: `python scripts/verify_datasets.py`
- For pipeline issues: `python scripts/test_pipeline.py`

### How to Verify Fixes
1. Run `python src/data/registry.py` standalone to check registration
2. Run `python scripts/verify_datasets.py --token $HF_TOKEN` to check dataset access
3. Run `python scripts/production_validation.py --smoke-steps 50` for full validation
4. Run specific test: `pytest tests/test_data_pipeline.py -v`

### Troubleshooting Flowcharts

#### Dataset Loading Flowchart
```
Dataset Load Error
    ├── Gated error?
    │   ├── Yes → Is HF_TOKEN set?
    │   │       ├── Yes → Request access at HF Hub
    │   │       └── No → Set HF_TOKEN in config or env
    │   └── No → Continue
    ├── "does not exist" error?
    │   ├── Check path for special characters (#, etc.)
    │   ├── Check dataset name spelling
    │   └── Verify dataset still exists on HF Hub
    ├── Timeout?
    │   ├── Increase datasets timeout
    │   └── Check network connectivity to HF Hub
    └── Other error?
        ├── Check datasets library version (need 2.20+)
        ├── Python-script-based datasets don't work without trust_remote_code
        └── Check for arrow/parquet file corruption
```

#### Training Failure Flowchart
```
Training fails
    ├── NaN loss?
    │   ├── Reduce learning rate
    │   ├── Increase gradient clipping (lower max_grad_norm)
    │   ├── Check for corrupted data samples
    │   └── Disable mixed precision
    ├── OOM (CUDA out of memory)?
    │   ├── Reduce batch_size
    │   ├── Enable gradient checkpointing
    │   ├── Reduce max_seq_length
    │   └── Use FSDP with CPU offload
    ├── Checkpoint save fails?
    │   ├── Check disk space
    │   ├── Check write permissions
    │   └── Reduce save frequency
    └── Slow convergence?
        ├── Increase learning rate
        ├── Check data quality
        ├── Verify weight initialization
        └── Increase warmup steps
```
