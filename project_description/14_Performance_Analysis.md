# Performance Analysis

## Data Pipeline Bottlenecks

### CPU Bottlenecks
1. **Text extraction in doc_builder.py**: Regex-based cleaning of raw HTML is CPU-bound. For 2000 pages/source, this adds ~30s per source.
   - **Optimization**: Use `lxml.html.clean` or `readability-lxml` for faster extraction
   - **Status**: Not critical (one-time build, not training)

2. **Tokenization**: `tokenizer()` calls are CPU-bound on long sequences.
   - **Optimization**: Use `tokenizer.encode_batch()` with parallelism
   - **Status**: Packaged into DataLoader workers

3. **Deduplication**: MinHash/SimHash for large datasets.
   - **Optimization**: Reduce threshold or use exact dedup only
   - **Status**: Acceptable for current scale

### GPU Bottlenecks
1. **SSM scan kernel**: The selective_scan operation is memory-bandwidth-bound.
   - **Optimization**: Use Triton kernel (mode="triton" in ssm_scan.py)
   - **Status**: CUDA and sequential modes available, Triton backend scaffolded

2. **LatentSandbox energy minimization**: Requires K× more compute than a single forward pass.
   - **Optimization**: Use LatentSandboxEfficient (sequential trajectories)
   - **Status**: Both parallel and sequential modes available

3. **Packing overhead**: Random window sampling + EOS concatenation adds negligible GPU cost (<1%).

4. **Decoder evaluation redundancy (fixed)**: `DharaModel.forward` previously ran the hierarchical decoder stack + dense vocab projection twice per micro-batch (once for `_vocab_logits`, again inside `hierarchical_log_prob`). It now runs exactly once and reuses the logits for the LM loss (causal shift applied via a target mask), roughly halving the dominant per-step cost. Dense-mode loss math is unchanged. The opt-in `head_ce: topk` can further shrink the loss/backward target set to the decoder's adaptive top-k indices union the target token, but the dense vocab projection still runs in the forward pass for candidate scoring.

### I/O Bottlenecks
1. **Dataset loading from HF Hub**: Network-bound for streaming datasets.
   - **Optimization**: Enable local caching (`cache_dir` in config)
   - **Status**: Cache hit after first epoch

2. **JSONL write/read in doc_builder**: Disk-bound for 2000 pages/source.
   - **Optimization**: Use GZip compression for JSONL files
   - **Status**: Already implemented (gzip in DocumentBuilder)

3. **Checkpoint save/load**: Disk-bound for multi-GB state dicts.
   - **Optimization**: Use `save_pretrained()` with sharded checkpoints
   - **Status**: Basic save/load implemented

### Memory Bottlenecks
1. **WeightedMixedDataset index array**: O(total_samples) memory for assignments.
   - **Optimization**: Generate indices on-the-fly or use memory-mapped arrays
   - **Status**: Acceptable for 100M+ samples

2. **LatentSandbox parallel trajectories**: O(K × batch × d_hidden) GPU memory.
   - **Optimization**: Use sequential processing (LatentSandboxEfficient)
   - **Status**: Both modes available

3. **SSM compression state**: O(1) — fixed d_state regardless of sequence length.
   - **Status**: No issue, this is the architecture's advantage

## Recommended Optimizations by Priority

| Priority | Optimization | Expected Gain | Complexity |
|----------|-------------|---------------|------------|
| HIGH | Enable Triton SSM scan kernel | 2-3x SSM throughput | HIGH |
| HIGH | Use cache_dir for dataset caching | 5-10x faster reload | LOW |
| HIGH | Implement streaming dataloader parallel workers | 2x training throughput | MEDIUM |
| MEDIUM | Add tokenization parallelism | 2-3x tokenization speed | LOW |
| MEDIUM | Use efficient LatentSandbox for limited GPU | 50% GPU memory savings | LOW |
| LOW | GZip compress JSONL doc files | 5-10x smaller files | LOW |
| LOW | Sharded checkpoint format | Faster save/load for large models | MEDIUM |

## Current Performance Baseline (from code analysis)
- Registry build: ~0.04s
- Doc builder: ~2500 pages across 18 sources in ~6 minutes
- Dataset streaming: First-sample latency ~2-10s per dataset (network-bound)
- Token distribution measurement: ~2000 samples in ~30s
- Smoke training: 100 steps in ~5 minutes (single A100, small config)
