# API Reference

## `src.data.registry`

### `DatasetInfo` (dataclass)

Container for metadata about a single dataset entry in the registry.

**Fields:**

| Field | Type | Default | Description |
|---|---|---|---|
| `path` | `str` | — | HuggingFace dataset path (e.g. `"bigcode/the-stack-v2-dedup"`) or `"json"` for local JSONL |
| `category` | `str` | — | One of 8 category names: `"code"`, `"web_text"`, `"docs"`, `"wiki"`, `"math"`, `"science"`, `"books"`, `"structured_knowledge"` |
| `weight` | `float` | — | Relative sampling weight (normalized per category) |
| `quality_score` | `float` | — | Quality assessment in [0, 1] (used as effective quality threshold) |
| `name` | `Optional[str]` | `None` | Dataset config/subset name (e.g. `"20231101.en"` for Wikipedia) |
| `split` | `str` | `"train"` | Dataset split to load |
| `data_dir` | `Optional[str]` | `None` | Local data directory (used for JSON datasets) |
| `language` | `Optional[str]` | `None` | Programming language (e.g. `"python"`, `"cpp"`) |
| `domain` | `str` | `"general"` | Content domain (e.g. `"backend"`, `"ml"`, `"algorithms"`) |
| `fallbacks` | `List[str]` | `[]` | List of fallback dataset paths |
| `text_fields` | `List[str]` | All `TEXT_FIELD_CANDIDATES` | Candidate text field names for text extraction |
| `license` | `str` | `"unknown"` | Dataset license |
| `max_samples` | `Optional[int]` | `None` | Maximum number of samples to load |
| `function_sampling` | `bool` | `False` | Enable function-level code sampling |
| `streaming` | `bool` | `True` | Use streaming mode |
| `priority` | `int` | `10` | Dataset priority (lower = higher priority) |
| `long_context` | `bool` | `False` | Dataset uses long context (>8K tokens) |
| `estimated_tokens` | `Optional[int]` | `None` | Approximate token count |
| `doc_type` | `str` | `"general"` | Documentation type |

**Methods:**

- `load_kwargs() -> dict` — Returns keyword arguments for `datasets.load_dataset()`, including `path`, `split`, `streaming`, and optionally `name` and `data_dir`.

### `DatasetRegistry`

A registry that holds all `DatasetInfo` entries, tracks fallback usage, and provides category-based queries and weight normalization.

**Constructor:**

- `__init__() -> None` — Initializes empty registry with `_entries`, `_used_fallbacks`, and `_counter`.

**Methods:**

- `register(info: DatasetInfo) -> None` — Registers a dataset entry under a composite key `{path}/{name or 'default'}/{category}/{counter}`.
- `get(key: str) -> Optional[DatasetInfo]` — Retrieves an entry by its composite key.
- `get_by_path_category(path: str, category: str) -> Optional[DatasetInfo]` — Finds an entry by exact path + category match.
- `all_entries() -> List[DatasetInfo]` — Returns all registered entries.
- `by_category(cat: str) -> List[DatasetInfo]` — Returns entries filtered by category name.
- `log_fallback(primary: str, fallback: str) -> None` — Logs a fallback usage (warns via logger, stores in `_used_fallbacks`).
- `normalize_weights(category_targets: Optional[Dict[str, float]]) -> float` — Scales weights within each category so the sum matches the target. Returns the total sum of all weights after normalization. Modifies `DatasetInfo.weight` in place.
- `summary() -> dict` — Returns `{"total_registered": int, "total_weight": float, "fallbacks_used": dict}`.

### Module-Level Constants

- `TEXT_FIELD_CANDIDATES: List[str]` — `["text", "content", "body", "code", "document", "article", "source", "output", "problem", "solution", "abstract", "section"]`
- `TEXT_FIELD_PRIORITY: List[str]` — Priority-ordered list of field names for text detection (includes `"func_code_string"`, `"whole_func_string"`, `"documentation"`, `"doc_string"`, `"func_documentation_string"`).
- `STACK_V2_LANGUAGES: Dict[str, str]` — Maps language names (e.g. `"Python"`, `"Rust"`) to short keys (e.g. `"python"`, `"rust"`). 17 languages total.
- `CODE_LANG_TARGETS: Dict[str, float]` — Target language distribution within code category (e.g. `"python": 0.25`).
- `WEB_FALLBACKS: List[str]` — `["allenai/dolma", "togethercomputer/RedPajama-Data-1T", "tiiuae/falcon-refinedweb"]`
- `DOC_SOURCES: Dict[str, Dict]` — Maps documentation source names to metadata (language, quality_score, weight, domain). 18 sources: `python-docs`, `pytorch-docs`, `numpy-docs`, `rust-book`, `go-docs`, `mdn-docs`, `fastapi-docs`, `cuda-docs`, `linux-kernel-docs`, `opencv-docs`, `kubernetes-docs`, `docker-docs`, `postgresql-docs`, `sqlite-docs`, `cudnn-docs`, `onnx-docs`, `rfcs`, `lang-specs`.
- `CATEGORY_WEIGHTS: Dict[str, float]` — Target token distribution per category (sum = 1.0): `code: 0.30`, `web_text: 0.20`, `docs: 0.15`, `wiki: 0.10`, `math: 0.10`, `science: 0.05`, `books: 0.05`, `structured_knowledge: 0.05`.

### Module-Level Functions

- `detect_text_fields(sample: dict, known_fields: Optional[list]) -> list` — Scans a sample dict for text fields. First checks `known_fields` if provided, then `TEXT_FIELD_PRIORITY`, then all keys (looking for strings > 50 chars). Returns a list with one field name (or `["text"]` as fallback).
- `extract_text(sample: dict, fields: Optional[list]) -> str` — Extracts text from a sample using the given fields or priority list. Falls back to concatenating all long string values (100+ chars) separated by double newlines.
- `build_registry() -> DatasetRegistry` — Constructs the full registry by calling all `_register_*` helpers in sequence: `_register_code` (code: 30%), registers FineWeb (web_text: 20%), `_register_docs` (docs: 15%), `_register_wiki` (wiki: 10%), `_register_math` (math: 10%), `_register_science` (science: 5%), `_register_books` (books: 5%), `_register_structured` (structured_knowledge: 5%). Normalizes weights to match `CATEGORY_WEIGHTS`. Returns populated `DatasetRegistry` with ~55 entries.

### Registration Helpers (internal)

- `_register_code(registry: DatasetRegistry) -> float` — Registers code datasets: OpenCoder-LLM/opc-fineweb-code-corpus (15%), bigcode/the-stack-v2-dedup per-language entries (30%), code-search-net/code_search_net (20%), codeparrot/codeparrot-clean (20%), deepmind/code_contests (15%). Returns accumulated weight.
- `_register_docs(registry: DatasetRegistry) -> float` — Registers FineWeb-Edu sample-10BT as doc proxy (3%) and 18 `DOC_SOURCES` entries as JSON datasets pointing to `data/docs/{name}` (weighted by `cfg["w"]`).
- `_register_wiki(registry: DatasetRegistry) -> float` — Registers `wikimedia/wikipedia` "20231101.en" with full wiki weight.
- `_register_math(registry: DatasetRegistry) -> float` — Registers 6 math datasets: open-web-math (42%), NuminaMath-CoT (20%), NuminaMath-1.5 (15%), MathPile (10%), FineWeb-Edu (10%), leandojo (3%).
- `_register_science(registry: DatasetRegistry) -> float` — Registers FineWeb-Edu sample-10BT as science corpus (100%).
- `_register_books(registry: DatasetRegistry) -> float` — Registers FineWeb-Edu (50%) and FineWeb (30%) for books category.
- `_register_structured(registry: DatasetRegistry) -> float` — Registers FineWeb-Edu (50%) and FineWeb (50%) for structured_knowledge category.

---

## `src.data.streaming`

### Module-Level Constants

- `GATED_DATASET_HELP: str` — Help text for gated dataset access instructions. Re-exported from `src.data.metadata_cache`.

### Module-Level Functions

- `_is_gated_error(e: Exception) -> bool` — Checks if an exception message contains gated-related keywords. Re-exported from `src.data.metadata_cache`.
- `_handle_load_error(path: str, e: Exception) -> None` — Logs appropriate error message for gated vs. non-gated dataset load failures.
- `stream_dataset(path: str, split: str = "train", name: Optional[str] = None, data_dir: Optional[str] = None, streaming: bool = True, limit: Optional[int] = None, text_fields: Optional[List[str]] = None) -> Generator[Dict[str, Any], None, None]` — Loads a dataset via HuggingFace `datasets.load_dataset()`, auto-detects text fields, extracts text, yields samples with an added `"text"` key. Supports DatasetDict and IterableDatasetDict. Respects `limit` for maximum sample count.

**Raises:** Logs errors via `_handle_load_error` for gated/not-found/network issues; does not raise — returns empty on failure.

- `stream_dataset_with_fallbacks(info: DatasetInfo, registry: DatasetRegistry, limit: Optional[int] = None) -> Generator[Dict[str, Any], None, None]` — Builds a fallback chain from `info.fallbacks`, resolves fallback paths via `registry.get_by_path_category()`, attempts to stream from each entry in sequence. First successful non-empty dataset stops the chain. Logs fallback activation via `registry.log_fallback()`. Logs error if all fallbacks exhausted.

**Raises:** Logs per-dataset exceptions and continues to next fallback; never raises.

### `StreamingManager`

**Constructor:**

- `__init__(self, registry: DatasetRegistry) -> None` — Stores registry reference and initializes skip counters.

**Methods:**

- `stream_all(limit_per_dataset: Optional[int] = None) -> Generator[Dict[str, Any], None, None]` — Streams all datasets in the registry in order via `stream_dataset_with_fallbacks`.
- `stream_category(category: str, limit_per_dataset: Optional[int] = None) -> Generator[Dict[str, Any], None, None]` — Streams only datasets matching a specific category.
- `skip_rate() -> Dict[str, float]` — Returns the proportion of skipped samples per dataset key.

### `MassiveDataCollector` (Legacy)

**Constructor:**

- `__init__(self, datasets_cfg: Optional[List] = None) -> None` — Stores dataset config list.

**Methods:**

- `get_dataset_list() -> list` — Returns the stored dataset list.
- `stream_single_dataset(ds_info, limit=None, theme="all", skip_samples=0, raw_text=True) -> Generator` — Legacy wrapper that constructs a `DatasetInfo` from the given object and delegates to `stream_dataset_with_fallbacks`.

---

## `src.data.drivers`

Unified dataset loading abstraction. Every dataset entry resolves to exactly one driver family; the driver owns fingerprinting, warm-start reuse, iterator resume, and streaming.

### Module-Level Constants

- `BUILDER_SCHEMA_VERSION: int` — Schema version of builder records (currently `1`).
- `DRIVER_KIND_FILE` / `DRIVER_KIND_SCRIPT` / `DRIVER_KIND_LOCAL` / `DRIVER_KIND_STREAMING: str` — Driver family identifiers (`"file"`, `"script"`, `"local"`, `"streaming"`).

### Module-Level Functions

- `compute_builder_fingerprint(preprocess_sig: str, token_sig: str, identity: Dict[str, Any]) -> str` — Fingerprint of a script builder from preprocess/tokenizer signatures plus identity fields (repo/name/split/data_dir/revision/script_revision/builder_class). Resume state is excluded.
- `inspect_builder(builder, split: str) -> Tuple[List[str], Optional[str]]` — Inspects a builder for its resolved source shards: `config.data_files` (split-filtered, auto-converted parquet/arrow repos) first, then the streaming iterable's exposed shard sources (including callable `shard_data_sources()`). Returns `(files, loader)` where `files` is empty for pure generator (IterableDataset) builders. Used by `detect_driver` to classify File vs Script family.
- `script_identity(builder) -> Tuple[str, Optional[str]]` — Returns `(script_revision, builder_class)` for builder-record identity.
- `local_file_list(path: str) -> Tuple[List[str], Optional[str]]` — Lists recognized data files in a local directory and detects the loader (parquet/arrow/csv/json/text).
- `detect_driver(info: DatasetInfo, meta_cache: Optional[DatasetMetadataCache], builder_cache: Optional[BuilderCache], local_dir: Optional[str] = None) -> DatasetDriver` — Classifies an entry in order: existing local dir → `LocalDatasetDriver`; verifying metadata record → `FileDatasetDriver`; verifying builder record → `ScriptDatasetDriver`; live `load_dataset_builder()` inspection → File (if shard sources) or Script (if none); inspection failure (gated/not-found/offline) → `StreamingDatasetDriver` with a diagnostics dict (including `resolve_error`). A fully warm run makes zero network or builder-resolution calls.
- `load_dataset_builder(*args, **kwargs)` — Thin re-export of `datasets.load_dataset_builder` (kept at module level so tests can monkeypatch it).

### `DatasetDriver` (base)

**Constructor:** `__init__(self, kind: str, record: Optional[dict], meta_cache=None, builder_cache=None)` — Stores family kind and the cached record.

**Methods:**
- `fingerprint() -> str` — Stable identity fingerprint.
- `resolve() -> bool` — Attempts to restore a cached record for the entry.
- `cache() -> bool` — Persists the record (file list for File family, builder identity for Script family).
- `invalidate() -> None` — Removes the cached record.
- `resume() -> Dict[str, Any]` — Returns the persisted resume state (script iterator raw-row offset).
- `warmup() -> bool` — Re-verifies the record without network.
- `prefetch() -> None` — Background warm-start of the driver.
- `stream(**kwargs) -> Generator[Dict[str, Any], None, None]` — Yields gated samples tagged `_shard` and `_raw_seq` (identity-preserving vs. the original sequential path).

### `FileDatasetDriver` (`DatasetDriver`)

**Constructor:** `__init__(self, record=None, meta_cache=None, builder_cache=None)` — File-family driver (`kind="file"`).

- Streams via the metadata cache's direct Arrow iterable (`stream_from_record`), shard-parallel, with text-field detection and gating. No HF resolution on warm runs.

### `LocalDatasetDriver` (`FileDatasetDriver`)

**Constructor:** `__init__(self, record=None, meta_cache=None, builder_cache=None)` — File family for existing local directories (`kind="local"`); never contacts the Hub.

### `ScriptDatasetDriver` (`DatasetDriver`)

**Constructor:** `__init__(self, record=None, builder_cache=None, load_builder=None)` — Script-family driver (`kind="script"`) for builders without shard sources.

**Methods:**
- `resumed() -> bool` — True if the stream started from a persisted offset.
- `raw_consumed() -> int` — Number of raw rows consumed this stream (compared against the dataset length for `natural_end`).
- `natural_end() -> bool` — True when the iterator exhausted naturally (raw rows consumed == dataset length).
- `save_resume() -> None` — Persists the current raw-row offset in the builder record.
- `reset_resume() -> None` — Clears the persisted offset.
- `stream(...)` — Streams via `load_dataset_builder()` → `as_streaming_dataset(split)` → `islice(offset)`, gating each sample and tagging `_shard=0`, `_raw_seq=offset+raw`. One module-level builder instantiation per stream; no network on warm runs.

### `StreamingDatasetDriver` (`DatasetDriver`)

**Constructor:** `__init__(self, record=None, meta_cache=None, builder_cache=None, info=None, diagnostics=None)` — Streaming fallback (`kind="streaming"`); mirrors the original cold `load_dataset` path, optionally capturing a metadata record mid-stream. Used when builder inspection failed; the failure short-circuits (no cold-path retry).

### `BuilderCache`

**Constructor:** `__init__(self, root: Path, enabled: bool = True, fingerprint_version: int = 1)` — Cache of builder identity records under `<metadata_cache_dir>/builders`.

**Methods:**
- `safe_dir_name(repo: str, name: Optional[str]) -> str` (classmethod) — Filesystem-safe directory name from repo+name.
- `record_dir(info) -> Path` — Directory for the given entry's record.
- `fingerprint_for(info, preprocess_sig, token_sig) -> str` — Expected fingerprint for an entry.
- `build_record(info, preprocess_sig, token_sig, script_revision, builder_class) -> Dict[str, Any]` — Constructs a record.
- `get(repo, name=None, split="train") -> Optional[Dict[str, Any]]` — Loads the record without verification.
- `verify(info, preprocess_sig, token_sig) -> Optional[Dict[str, Any]]` — Loads the record and verifies schema/version/split/fingerprint.
- `save(rec, info) -> bool` — Atomic write (temp file + `os.replace`).
- `invalidate(repo, name=None, split="train") -> None` — Removes the record directory.
- `save_resume(rec, offset) -> None` / `reset_resume(rec) -> None` — Persists/clears the script iterator raw-row offset (runtime state, excluded from the fingerprint).

---

## `src.data.pipeline`

### Module-Level Constants

- `CODE_LANGUAGE_WEIGHTS: Dict[str, float]` — Weight distribution for code languages.
- `DOMAIN_MARKERS: Dict[str, List[str]]` — Keyword markers for 21 domain categories (e.g. `"backend"`, `"ml"`, `"security"`).
- `QUALITY_THRESHOLDS: Dict[str, float]` — Minimum quality scores per category for filtering.
- `LICENSE_KEYWORDS: List[str]` — Keywords for license/boilerplate detection.
- `BOILERPLATE_FILE_PATTERNS: List[str]` — File path patterns to skip.

### Module-Level Functions

- `detect_domain(text: str, category: str) -> str` — For code category, scores text against `DOMAIN_MARKERS` and returns the best-matching domain. For non-code, returns the category itself.
- `detect_language(text: str, dataset_name: str = "") -> str` — Heuristic programming language detection using dataset name keywords and code pattern markers. Returns one of `"python"`, `"javascript"`, `"typescript"`, `"java"`, `"rust"`, `"go"`, `"other"`.
- `random_window_sample(tokens: list, max_seq_length: int) -> list` — If token sequence exceeds `max_seq_length`, samples a random contiguous window of `max_seq_length` tokens. Otherwise returns the original list.
- `remove_boilerplate(text: str, keywords: Optional[List[str]] = None) -> str` — Removes license/copyright boilerplate from the beginning of text by scanning the first 50 lines for keyword matches in comment contexts. Also collapses excessive blank lines and trims trailing whitespace.
- `should_skip_file(file_path: str, patterns: Optional[List[str]] = None) -> bool` — Returns `True` if `file_path` contains any boilerplate patterns (e.g. `"_pb2.py"`, `"node_modules/"`, `".git/"`).
- `compute_quality_score(text: str, category: str, language: str = "text") -> float` — Delegates to `document_quality_score()` from `quality.py` and returns the `"final"` score.
- `passes_quality_filter(text: str, category: str, quality_score: Optional[float] = None, lang: str = "text") -> bool` — Returns `True` if `compute_quality_score(text, category, lang)` >= threshold (or the provided `quality_score` threshold).
- `pack_sequences(tokenized_samples: list, max_seq_length: int, eos_token_id: int) -> tuple` — Packs multiple tokenized sequences into fixed-length packed sequences separated by EOS tokens. Returns `(packed_examples, packing_efficiency)`. Each packed example includes `input_ids`, `labels`, `attention_mask`, `_segments` (count), and `_avg_quality`.

### `WeightedMixedDataset(TorchDataset)`

**Constructor:**

- `__init__(datasets_with_weights: list, total_samples: int, balance_languages: bool = False, lang_target: Optional[Dict[str, float]] = None, balance_domains: bool = False, domain_target: Optional[Dict[str, float]] = None, rng_seed: int = 42, quality_scores: Optional[List[float]] = None)` — Creates a mixed dataset that samples from sub-datasets according to effective weights (base_weight * quality_score). Supports language and domain stratification for balanced sampling.

**Methods:**

- `__len__() -> int` — Returns `total_samples`.
- `__getitem__(idx: int) -> dict` — Returns a sample from the pre-computed assignment table.
- `stats() -> Dict[str, Any]` — Returns dataset statistics: `total_samples`, `datasets` count, `weights`, `consumed` counts, `remaining_ratios`.

**Properties (private):**

- `_effective_weights() -> List[float]` — Recomputed weights based on remaining samples per dataset (adaptive resampling).
- `_sample_adaptive(rng, group_map=None, group_name="")` — Builds the assignment table using multinomial sampling with adaptive weight adjustment.
- `_build_from_stratification(rng, group_map, group_samples)` — Builds assignment table stratified by language/domain groups.

### `DataPipeline`

**Constructor:**

- `__init__(cfg: Config, tokenizer: PreTrainedTokenizerBase) -> None` — Initializes the pipeline with config and tokenizer. Sets `HF_TOKEN` env var from config. Configures deduplicator (simhash/minhash/exact), contamination filter, quality scorer, exact dedup, and health reporter.

**Properties:**

- `collector -> MassiveDataCollector` — Lazy-initialized legacy data collector from `cfg.data.datasets`.

**Methods:**

- `format_prompt(language: str, problem: str, style: str = "standard", constitution: Optional[List[str]] = None) -> str` — Formats a prompt for SFT/instruction tuning. Supports `"standard"` and `"constitutional"` styles. Returns a formatted prompt string with optional constitution preamble.
- `tokenize_supervised(texts: List[Tuple[str, str]], max_length: int) -> Dict[str, Any]` — Tokenizes prompt-response pairs. Labels are set to -100 for prompt tokens and response token IDs for response tokens. Pads/truncates to `max_length`. Returns dict with `input_ids`, `attention_mask`, `labels`.
- `build_sft_dataset(data: List[Dict[str, str]], style: str = "standard", constitution: Optional[List[str]] = None, apply_quality_filter: bool = True, apply_dedup: bool = True, apply_contamination: bool = True) -> Dataset` — Builds a supervised fine-tuning dataset from instruction-response data. Applies quality filtering (length/content checks), contamination removal, and deduplication. Returns a HuggingFace `Dataset`.
- `build_preference_dataset(data: List[Dict[str, Any]]) -> Dataset` — Builds a preference dataset (DPO/RLHF) with `chosen_*` and `rejected_*` fields. Returns combined dataset with parallel chosen/rejected columns.
- `build_stage_dataset(raw_samples: List[Dict[str, str]]) -> Dataset` — Tokenizes raw samples into a HuggingFace `Dataset` using `_preprocess_batch`.
- `get_active_datasets_for_stage(stage) -> List[DatasetEntryConfig]` — Filters datasets by stage's `dataset_filter` (category-based).
- `build_pretrain_dataset(stage_name: str = "pretrain", dataset_filter: Optional[List[str]] = None, max_samples_per_dataset: Optional[int] = None) -> Dataset` — Legacy pretrain dataset builder. If `cfg.data.use_registry` is enabled, delegates to `build_pretrain_dataset_from_registry`. Otherwise uses the `MassiveDataCollector` path. Applies boilerplate removal, quality filtering, AST filtering (for code), deduplication, function sampling, tokenization, random window sampling, and sequence packing. Caches results to disk.
- `build_pretrain_dataset_from_registry(dataset_filter: Optional[List[str]] = None, max_samples_per_dataset: Optional[int] = None) -> Dataset` — Builds pretrain dataset using the registry system with `stream_dataset_with_fallbacks`. Runs full filtering pipeline: boilerplate removal, quality scoring (using `document_quality_score`), AST code filtering, deduplication, function sampling. Produces a `WeightedMixedDataset` with adaptive resampling.

**Private Methods:**

- `_get_cache_key(ds_info: DatasetEntryConfig, stage_name: str) -> str` — SHA-256 hash for caching.
- `_cached_dataset_path(ds_info: DatasetEntryConfig, stage_name: str) -> Path` — Cache file path construction.

---

## `src.data.doc_builder`

### Module-Level Functions

- `_session() -> requests.Session` — Creates a requests session with retry adapter (max 3 retries, backoff factor 1, status codes 429/500/502/503/504).
- `_clean_text(html: str) -> str` — Strips script, style, nav, footer tags and all HTML tags from HTML content. Collapses whitespace.
- `_extract_section_text(html: str, section_tag: str = "main", fallback: str = "body") -> str` — Extracts text from a specific HTML section tag (e.g. `<main>`) with optional fallback. Cleans via `_clean_text`.
- `_find_client_redirect(html: str, base_url: str) -> Optional[str]` — Detects `<meta refresh>` and `location.replace/href` client-side redirects and returns the redirect URL.
- `_fetch(url: str, session: Optional[requests.Session] = None) -> Optional[str]` — Fetches a URL with up to 4 redirect-following attempts. Returns HTML string or `None`.
- `scrape_all(output_dir: str, sources: Optional[List[str]] = None, max_per_source: int = 2000, source_timeout: int = 600) -> Dict[str, int]` — Scrapes all (or specified) documentation sources concurrently with per-source timeout. Returns dict mapping source names to page counts (or -1 on timeout).
- `build_doc_registry(base_dir: str, weights: Optional[Dict[str, float]] = None, total_docs_weight: float = 0.15) -> dict` — Builds a registry dict from cached JSONL files. Returns `{"json": [DatasetInfo, ...]}` entries with weights proportional to `total_docs_weight`.

### `DocPage` (dataclass)

- `url: str` — Page URL
- `title: str` — Extracted title
- `text: str` — Cleaned text content
- `sections: List[Dict[str, Any]]` — Optional section metadata

### `DocScraper` (ABC)

**Class Attributes:**

- `SOURCE_NAME: str = ""`
- `BASE_URL: str = ""`
- `LANGUAGE: str = "text"`
- `CATEGORY: str = "docs"`
- `MAX_PAGES: int = 5000`
- `MIN_TEXT_LENGTH: int = 200`
- `MAX_TEXT_LENGTH: int = 100000`

**Constructor:**

- `__init__(self, output_dir: Optional[str] = None)` — Sets up output path, session, and tracking sets/counters.

**Methods:**

- `discover_urls() -> Generator[str, None, None]` — **Abstract.** Yields URLs to scrape.
- `extract_text(html: str, url: str) -> Optional[str]` — **Abstract.** Extracts and cleans text from HTML. Returns `None` if extraction fails.
- `scrape() -> Generator[DocPage, None, None]` — Main scraping loop: discovers URLs, normalizes, deduplicates, fetches, extracts text, yields `DocPage` objects. Respects `MAX_PAGES`, `MIN_TEXT_LENGTH`, `MAX_TEXT_LENGTH`.
- `scrape_to_jsonl(output_path: Path) -> int` — Runs `scrape()` and writes results to a JSONL file with fields: `source`, `language`, `category`, `title`, `url`, `text`, `text_length`. Returns page count.
- `_normalize_url(url: str) -> str` — Normalizes URL (strips fragments, removes trailing slash).
- `_extract_title(html: str) -> Optional[str]` — Extracts `<title>` tag content.
- `_summary(count: int) -> str` — Returns a summary string with discovered, duplicate, and failed counts.

### `SphinxScraper(DocScraper)`

Additional class attributes: `URL_PATTERNS`, `EXCLUDE_PATTERNS`, `CONTENT_SELECTOR`, `CONTENT_FALLBACK`.

- `discover_urls()` — Yields seed URLs from `BASE_URL + URL_PATTERNS`, then discovers linked pages via `href` regex, respecting `EXCLUDE_PATTERNS` and `BASE_URL` prefix.
- `extract_text(html, url)` — Uses `_extract_section_text` with `CONTENT_SELECTOR` and `CONTENT_FALLBACK`.

### Concrete Scrapers (18 total)

All inherit from `SphinxScraper` unless noted:

| Class | Source Name | Language | Base URL | Notes |
|---|---|---|---|---|
| `PythonDocScraper` | python-docs | python | https://docs.python.org/3/ | Tutorial, library, reference, howto, using, whatsnew, faq |
| `PyTorchDocScraper` | pytorch-docs | python | https://pytorch.org/docs/stable/ | Custom discover: tensors, torch, nn, optim + genindex |
| `NumPyDocScraper` | numpy-docs | python | https://numpy.org/doc/stable/ | User, reference, dev, about, release |
| `FastAPIDocScraper` | fastapi-docs | python | https://fastapi.tiangolo.com/ | Tutorial, advanced, reference, deployment, how-to |
| `OpenCVDocScraper` | opencv-docs | cpp | https://docs.opencv.org/4.x/ | Modules, tutorials |
| `ONNXDocScraper` | onnx-docs | python | https://onnx.ai/onnx/ | Intro, tutorials, api, operators, optimizers |
| `DockerDocScraper` | docker-docs | shell | https://docs.docker.com/ | Get-started, build, compose, engine, network, etc. |
| `KubernetesDocScraper` | kubernetes-docs | shell | https://kubernetes.io/docs/ | Setup, concepts, tasks, tutorials, reference |
| `GoDocScraper` | go-docs | go | https://go.dev/doc/ | Tutorial, effective_go, faq, wiki |
| `PostgreSQLDocScraper` | postgresql-docs | sql | https://www.postgresql.org/docs/current/ | Custom discover with version-aware URL matching |
| `SQLiteDocScraper` (direct) | sqlite-docs | sql | https://sqlite.org/docs.html | Custom discover/extract |
| `MDNDocScraper` (direct) | mdn-docs | javascript | https://developer.mozilla.org/en-US/docs/Web | Custom discover with CORE_TOPICS and deprecated exclusion |
| `RustBookScraper` (direct) | rust-book | rust | https://doc.rust-lang.org/book/ | Custom discover via href extraction |
| `CUDADocScraper` (direct) | cuda-docs | cpp | https://docs.nvidia.com/cuda/ | 18+ CUDA doc seed pages with sub-link following |
| `CuDNNDocScraper` (direct) | cudnn-docs | cpp | https://docs.nvidia.com/deeplearning/cudnn/api/ | 6 seed pages, no sub-link following |
| `RFCDocScraper` (direct) | rfcs | text | https://www.rfc-editor.org/rfc/ | Custom discover via rfc-index.txt, plain text extraction |
| `LinuxKernelDocScraper` (direct) | linux-kernel-docs | text | https://www.kernel.org/doc/html/latest/ | Custom discover, div.document selector |
| `LangSpecScraper` (direct) | lang-specs | text | https://docs.python.org/3/reference/ | Hardcoded Python + Rust language spec URLs |

### Module-Level Dictionary

- `SCRAPERS: Dict[str, type[DocScraper]]` — Maps short names to scraper classes. 18 entries.

---

## `src.data.quality`

*(Referenced by pipeline — full API in quality.py)*

Key components used in pipeline:
- `QualityFilter` — Static methods: `check_length(text, min_len, max_len)`, `is_high_quality_content(content)`.
- `QualityScorer` — Computes composite quality scores.
- `ExactDeduplicator` — Exact text deduplication via hash set.
- `SimHashDeduplicator(threshold=0.85)` — Simhash-based fuzzy deduplication.
- `MinHashDeduplicator(threshold=0.85)` — MinHash-based fuzzy deduplication.
- `ContaminationFilter(benchmarks=[...])` — Filters benchmark-contaminated samples.
- `document_quality_score(text, category, language) -> dict` — 8-component quality scoring returning a dict with `"final"` score.

---

## `src.data.ast_filter`

- `extract_functions(text: str, language: str) -> List[dict]` — AST-based function extraction.
- `filter_code(text: str, language: str, ast_filter_cfg: ASTFilterConfig) -> Tuple[bool, str]` — Filters code based on AST parsing, auto-generation detection, identifier ratio, executable ratio. Returns `(is_valid, reason)`.
- `identifier_ratio(text: str, language: str) -> float` — Ratio of identifier tokens to total tokens.
- `executable_ratio(text: str, language: str) -> float` — Ratio of executable statements to total statements.

---

## `src.data.function_sampler`

- `sample_functions_or_fallback(text: str, language: str, cfg: FunctionSamplingConfig) -> str` — Extracts individual functions/classes/methods from code text using AST parsing. Falls back to original text if extraction fails or is empty.
- `is_code_empty_or_trivial(code: str) -> bool` — Returns `True` if code is empty, too short, or contains only trivial content.

---

## `src.data.health_reporter`

- `DatasetHealthReport(name, config_path)` — Tracks per-dataset and global statistics. Methods include `add_dataset_stats(...)`, `add_error(msg)`, `compute_global_stats()`, `save(output_dir)`, `summary_text()`.

---

## `src.data.sanity`

- `run_sanity_checks(dataset, tokenizer, num_samples=10, max_decode=512) -> dict` — Runs sanity checks on a built dataset: decodes samples, checks for empty/truncated content, repetitive patterns, and returns a report dict.

---

## `src.config.schema`

### `Config` (root Pydantic v2 model)

Top-level configuration object. Loaded from YAML via `load_config()`. Contains nested sub-models:

| Field | Type | Default | Description |
|---|---|---|---|
| `project` | `Optional[Dict[str, Any]]` | `None` | Project metadata (e.g. `seed: 42`) |
| `model` | `ModelConfig` | default | Model configuration |
| `training` | `TrainingConfig` | default | Training pipeline configuration |
| `distributed` | `DistributedConfig` | default | Distributed training configuration |
| `data` | `DataConfig` | default | Data pipeline configuration |
| `tokenizer` | `TokenizerConfig` | default | Tokenizer configuration |
| `alignment` | `AlignmentConfig` | default | Alignment/constitutional AI configuration |
| `output` | `OutputConfig` | default | Output paths configuration |
| `generation` | `GenerationConfig` | default | Text generation parameters |
| `evaluation` | `EvaluationConfig` | default | Evaluation benchmark configuration |

**Model Validators:**

- `migrate_v1_config(cls, values)` — `@model_validator(mode="before")`. Migrates v1 config keys (e.g. `data_collection` → `data.datasets`, `fsdp` → `distributed.fsdp`, `languages` removal, `rope_scaling_factor` → `rope_scaling`).

### Sub-Models

| Model | Key Fields |
|---|---|
| `ModelConfig` | `name`, `dtype`, `device`, `train_from_scratch`, `architecture: ModelArchitectureConfig`, `load_in_8bit`, `load_in_4bit` |
| `ModelArchitectureConfig` | `model_type` (llama/mixtral/qwen2_moe/deepseek_v2/nslt/methos_v3), `hidden_size`, `num_hidden_layers`, `num_attention_heads`, `num_key_value_heads`, `intermediate_size`, `moe: MoEConfig`, `nslt: NSLTConfig`, `methos_v3: MethosV3Config`, `multimodal: MultimodalConfig`, `max_position_embeddings`, `rope_theta`, `rope_scaling`, `attention_implementation`, `tie_word_embeddings`, `attention_bias`, `attention_dropout`, `hidden_act`, `rms_norm_eps`, `initializer_range`, `pretraining_tp`, `mlp_bias`, `gradient_checkpointing`, `activation_checkpointing`, `use_compile`, `vocab_size` |
| `MoEConfig` | `num_experts`, `top_k`, `expert_capacity`, `shared_expert_count`, `shared_expert_gate`, `norm_topk_prob`, `output_router_logits`, `aux_loss_coef`, `jitter_noise`, `router_aux_loss_coef` |
| `NSLTConfig` | `d_state`, `d_hidden`, `n_ssm_layers`, `n_ode_steps`, `solver`, `n_trajectories`, `n_sim_steps`, `use_efficient_sandbox`, `sparsity_pct` |
| `MethosV3Config` | 40+ parameters: `d_state`, `d_hidden`, `n_ssm_layers`, `n_hssm_levels`, `n_ode_steps`, `n_trajectories`, `n_sim_steps`, `use_efficient_sandbox`, `sparsity_pct`, `n_token_categories`, `n_languages`, `n_doc_roles`, `n_task_types`, `n_difficulty_levels`, `n_reasoning_types`, `max_subgoals`, `n_domains`, `max_reasoning_steps`, `n_semantic_concepts`, `working_mem_capacity`, `max_episodes`, `n_context_adapter_blocks`, `n_language_groups`, `adaptive_top_k_min`, `adaptive_top_k_max`, `n_experts`, `n_debate_rounds`, `max_refinement_passes`, `max_repair_iters`, `max_entities`, `n_relation_types`, `max_events`, `n_tool_types`, `enable_executive`, `enable_world_model`, `enable_learning_controller`, `enable_tools`, `enable_curiosity`, `enable_aux_losses`, `executive_gate_threshold`, `qa_max_passes`, `qa_converge_threshold`, `loss_weights` |
| `VisionConfig` | `enabled`, `vision_encoder`, `image_size`, `patch_size`, `vision_hidden_size`, `num_vision_layers`, `num_attention_heads`, `intermediate_size`, `projection_dim`, `freeze_vision_encoder`, `tie_vision_embeddings`, `image_token_id`, `max_images_per_sample` |
| `MultimodalConfig` | `vision: VisionConfig` |
| `RopeScalingConfig` | `type`, `factor`, `target_max_length`, `original_max_position_embeddings` |
| `TrainingConfig` | `max_seq_length`, `response_only_loss`, `ignore_data_skip`, `save_steps`, `save_total_limit`, `eval_strategy`, `eval_steps`, `logging_steps`, `pretrain: PretrainStageConfig`, `sft: SFTStageConfig`, `alignment: AlignmentPhaseConfig`, `instruction_tuning: InstructionTuningConfig`, `rlhf: RLHFConfig`, `safety: SafetyConfig` |
| `PretrainStageConfig` | `enabled`, `learning_rate`, `lr_scheduler_type`, `warmup_steps`, `weight_decay`, `batch_size`, `gradient_accumulation_steps`, `max_steps`, `optimizer`, `data_mix` |
| `SFTStageConfig` | `enabled`, `learning_rate`, `lr_scheduler_type`, `warmup_steps`, `weight_decay`, `batch_size`, `gradient_accumulation_steps`, `max_steps`, `optimizer`, `data_mix` |
| `AlignmentPhaseConfig` | `enabled`, `methods` (dpo/orpo/simpo/kto), `method_configs: AlignmentMethodConfig` |
| `AlignemntMethodConfig` | `dpo: Optional[DPOConfig]`, `orpo: Optional[ORPOConfig]`, `simpo: Optional[SimPOConfig]`, `kto: Optional[KTOConfig]` |
| `DataConfig` | `cache_dir`, `streaming`, `max_cache_gb`, `num_download_workers`, `datasets: List[DatasetEntryConfig]`, `quality: QualityPipelineConfig`, `curriculum: CurriculumConfig`, `preprocessing: PreprocessingConfig`, `language_balancing: LanguageBalancingConfig`, `synthetic_reasoning: SyntheticReasoningConfig`, `domain_balancing: DomainBalancingConfig`, `ast_filter: ASTFilterConfig`, `function_sampling: FunctionSamplingConfig`, `sampler: WeightedSamplerConfig`, `sanity_checks: SanityCheckConfig`, `health_reporting: HealthReportingConfig`, `hf_token`, `use_registry` |
| `DatasetEntryConfig` | `path`, `max_samples`, `split`, `name`, `data_dir`, `language`, `category`, `domain`, `weight`, `quality_score`, `token_weight`, `strip_instruction_format`, `function_sampling` |
| `TokenizerConfig` | `source`, `huggingface_model`, `vocab_size`, `max_samples`, `type`, `force`, `add_prefix_space`, `add_bos_token`, `add_eos_token` |
| `AlignmentConfig` | `prompt_style`, `constitution: List[str]` |
| `DistributedConfig` | `strategy` (fsdp/deepspeed/ddp/none), `fsdp: FSDPConfig`, `deepspeed: Optional[DeepSpeedConfig]` |
| `FSDPConfig` | `enabled`, `sharding_strategy`, `transformer_layer_cls`, `backward_prefetch`, `forward_prefetch`, `activation_checkpointing`, `use_orig_params`, `sync_module_states`, `limit_all_gathers`, `mixed_precision`, `cpu_offload` |
| `GenerationConfig` | `max_new_tokens`, `temperature`, `top_p`, `top_k`, `repetition_penalty`, `do_sample`, `num_beams` |
| `EvaluationConfig` | `benchmarks`, `benchmark_configs`, `automated_report`, `report_dir`, `timeout`, `evaluate_during_training`, `eval_frequency` |
| `OutputConfig` | `model_dir`, `data_dir`, `checkpoint_dir`, `log_dir`, `experiment_tracking: ExperimentTrackingConfig` |

### Module-Level Functions

- `load_config(path: str | Path = "config.yaml") -> Config` — Loads YAML config file and validates against `Config` model. Raises `FileNotFoundError` if path doesn't exist.
- `config_to_dict(cfg: Config) -> Dict[str, Any]` — Converts config to Python dict via `model_dump(mode="python")`.

---

## `src.models.factory`

### `ModelFactory`

**Static Methods:**

- `build_model_config(vocab_size: int, arch: ModelArchitectureConfig, max_seq_length: int) -> Any` — Builds a HuggingFace `PretrainedConfig` for standard model types (llama, mixtral, qwen2_moe, deepseek_v2). Returns `LlamaConfig`, `MixtralConfig`, or `AutoConfig.for_model(...)`. Handles rope_scaling configuration.

- `architecture_spec(cfg: Config) -> Dict[str, Any]` — Extracts architecture specification from config as a dict for compatibility checking.

- `saved_architecture_spec(saved: Dict[str, Any]) -> Dict[str, Any]` — Extracts architecture specification from a saved `config.json` dict.

- `is_compatible(cfg: Config, tokenizer: PreTrainedTokenizerBase, model_path: Path) -> bool` — Checks if a saved checkpoint's architecture matches the current config.

- `create_model(cfg: Config, tokenizer: PreTrainedTokenizerBase) -> nn.Module` — Creates a model based on `cfg.model.architecture.model_type`. Supports:
  - `"methos_v3"` → `MethosV3Model` (from `src.methos_v3`)
  - `"nslt"` → `NSLTModel` (from `src.nslt`)
  - `"llama"` → `LlamaForCausalLM`
  - `"mixtral"` → `MixtralForCausalLM`
  - `"qwen2_moe"` / `"deepseek_v2"` → `AutoModelForCausalLM.from_config()`

- `create_methos_v3(cfg) -> MethosV3Model` — Convenience method (inline logic in `create_model`).

- `create_nslt(cfg) -> NSLTModel` — Convenience method (inline logic in `create_model`).

- `load_model(path: str | Path, cfg: Config, tokenizer: Optional[PreTrainedTokenizerBase] = None, strict: bool = False) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]` — Loads a model from checkpoint. Supports methos_v3 (with state dict filtering for shape mismatches), nslt, and standard HF models (via `AutoModelForCausalLM.from_pretrained`). Returns `(model, tokenizer)`.

- `load_tokenizer(path: str | Path = "models/tokenizer", cfg: Optional[Any] = None) -> PreTrainedTokenizerBase` — Loads tokenizer from path. Tries `AutoTokenizer`, then `PreTrainedTokenizerFast`, then falls back to constructing from `vocab.json` + `merges.txt`. Sets pad token and padding side.

- `estimate_model_size(arch: ModelArchitectureConfig, tokenizer_or_vocab: Any = None) -> Dict[str, Any]` — Estimates parameter counts for all architecture types. Returns `{"total_params_b": float, "active_params_b": float, "layers": int, "experts": int, "top_k": int, "architecture": str}`.

- `get_fsdp_layer_cls(model_type: str) -> str` — Returns the transformer layer class name for FSDP wrapping (e.g. `"HierarchicalSSMStack"` for methos_v3, `"SSMCompressionEngine"` for nslt).

**Private Methods:**

- `_resolve_dtype(dtype_str: str) -> torch.dtype` — Maps string dtype to `torch.dtype`.
- `_try_load_tokenizer(path: Path) -> PreTrainedTokenizerBase` — Attempts multiple strategies to load a tokenizer.

### Module-Level Constants

- `ARCH_CONFIG_MAP: Dict[str, tuple]` — Maps model_type to `(ConfigClass, ModelClass)` for standard architectures.
- `ARCH_FSDP_LAYER_MAP: Dict[str, str]` — Maps model_type to FSDP transformer layer class name.

---

## `src.methos_v3.model`

### `MethosV3Config(PretrainedConfig)`

**Class Attributes:**
- `model_type = "methos_v3"`

**Constructor Parameters (40+):**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `vocab_size` | `int` | 128000 | Vocabulary size |
| `hidden_size` | `int` | 10240 | Model hidden dimension |
| `d_state` | `int` | 4096 | SSM state dimension |
| `d_hidden` | `int` | 10240 | Hidden dimension for higher layers |
| `max_position_embeddings` | `int` | 262144 | Maximum sequence length |
| `rope_theta` | `float` | 10000000.0 | RoPE base frequency |
| `sparsity_pct` | `float` | 1.0 | Output sparsity (1.0 = full) |
| `n_ssm_layers` | `int` | 6 | Number of SSM layers |
| `n_hssm_levels` | `int` | 3 | Hierarchical SSM levels |
| `n_ode_steps` | `int` | 8 | ODE integration steps |
| `n_trajectories` | `int` | 7 | Parallel trajectories in sandbox |
| `n_sim_steps` | `int` | 16 | Energy descent simulation steps |
| `use_efficient_sandbox` | `bool` | True | Memory-efficient sandbox variant |
| `n_token_categories` | `int` | 8 | Token category count |
| `n_languages` | `int` | 16 | Language count |
| `n_doc_roles` | `int` | 8 | Document role count |
| `n_task_types` | `int` | 8 | Task type count |
| `n_difficulty_levels` | `int` | 5 | Difficulty levels |
| `n_reasoning_types` | `int` | 8 | Reasoning type count |
| `max_subgoals` | `int` | 64 | Maximum subgoals |
| `n_domains` | `int` | 5 | Domain count |
| `max_reasoning_steps` | `int` | 32 | Max reasoning steps |
| `n_semantic_concepts` | `int` | 4096 | Semantic concept count |
| `working_mem_capacity` | `int` | 512 | Working memory capacity |
| `max_episodes` | `int` | 256 | Max episodes |
| `n_context_adapter_blocks` | `int` | 4 | Context adapter blocks |
| `n_language_groups` | `int` | 8 | Language groups |
| `adaptive_top_k_min` | `int` | 32 | Min top-k for sparse decoder |
| `adaptive_top_k_max` | `int` | 2048 | Max top-k for sparse decoder |
| `n_experts` | `int` | 7 | Number of specialists |
| `n_debate_rounds` | `int` | 3 | Specialist debate rounds |
| `max_refinement_passes` | `int` | 5 | Max refinement passes |
| `max_repair_iters` | `int` | 3 | Max repair iterations |
| `max_entities` | `int` | 64 | Max entities in world model |
| `n_relation_types` | `int` | 16 | Relation types |
| `max_events` | `int` | 32 | Max events |
| `n_tool_types` | `int` | 4 | Tool types |
| `enable_executive` | `bool` | True | Enable executive controller |
| `enable_world_model` | `bool` | True | Enable world model |
| `enable_tools` | `bool` | True | Enable internal tools |
| `enable_curiosity` | `bool` | True | Enable curiosity module |
| `enable_aux_losses` | `bool` | True | Enable auxiliary losses |
| `executive_gate_threshold` | `float` | 0.3 | Gate threshold |
| `qa_max_passes` | `int` | 5 | QA max passes |
| `qa_converge_threshold` | `float` | 0.05 | QA convergence threshold |
| `loss_weights` | `Optional[Dict[str,float]]` | None | Loss weights |
| `is_encoder_decoder` | `bool` | False | HF compatibility flag |

### `MethosV3Model(PreTrainedModel)`

**Class Attributes:**
- `config_class = MethosV3Config`
- `base_model_prefix = "model"`
- `supports_gradient_checkpointing = True`
- `_no_split_modules` — List of module names for FSDP wrapping.

**Architecture (11 layers):**
1. `IntelligentTokenizer` — Token categorization, language identification
2. `AdaptiveSemanticEmbedding` — RoPE-based semantic embeddings with task-type adaptation
3. `HierarchicalMemoryEngine` — Multi-level SSM memory (HSSM) with working memory
4. `IntentUnderstanding` + `AdaptiveDifficultyRouter` — Task intent and difficulty routing
5. `GlobalPlanner` — Subgoal planning
6. `AdaptiveContinuousReasoning` — Domain-specific ODE-based reasoning
7. `CognitiveWorkspace` — Central workspace hub
8. `SpecialistSandbox` — Multi-expert debate sandbox
9. `QualityAssurance` — Self-verification and correction
10. `HierarchicalSparseDecoder` — Sparse vocabulary projection with language-group routing
11. `ExecutiveController` — Resource allocation gating

**Optional modules:** `WorldModel`, `InternalToolInterface`, `AuxiliaryLossComputer`

**Methods:**

- `forward(input_ids, attention_mask=None, labels=None, categories=None, languages=None, doc_roles=None, task_ids=None, language_ids=None, mem_state=None, aux_targets=None, **kwargs) -> CausalLMOutputWithPast` — Full forward pass through all 11 layers. Returns loss if labels provided, logits otherwise.

- `generate(input_ids, max_new_tokens=1024, temperature=0.7, top_k=40, top_p=0.9, eos_token_id=None, **kwargs) -> torch.LongTensor` — Autoregressive generation with top-k + top-p sampling. Uses memory state for efficient incremental decoding.

**Properties:**
- `device -> torch.device`

**Private Methods:**
- `_log_architecture()` — Logs parameter counts and module status.

### `MethosV3ForCausalLM(MethosV3Model)`

Adds a linear `lm_head` for standard causal LM interface.

- `forward(*args, **kwargs) -> CausalLMOutputWithPast` — Delegates to parent, passes through `lm_head` on logits during inference.

### `MoEMethosV3Model(MethosV3Model)`

- Constructor adds `n_experts` and `top_k_experts` parameters for MoE variant.

---

## `src.nslt.model`

### `NSLTModel(nn.Module)`

**4-Layer Architecture:**
1. `SSMCompressionEngine` (stacked) — O(1) memory SSM encoding
2. `LTCRoutingLayer` — Continuous ODE-based routing
3. `LatentSandbox` / `LatentSandboxEfficient` / `MCTSLatentSandbox` — Latent space reasoning
4. `SparseOutputSynthesizer` — Ultra-sparse vocabulary projection (not O(V) logits)

**Supporting modules:** `TokenEmbedding`, `RotaryPositionEncoding`

**Constructor:**

- `__init__(vocab_size=128000, d_model=7168, d_state=2048, d_hidden=7168, n_ssm_layers=4, max_seq_len=262144, rope_base=10000000.0, sparsity_pct=1.0, n_ode_steps=8, n_trajectories=8, n_sim_steps=16, use_efficient_sandbox=False, use_mcts_sandbox=False, device=torch.device("cpu"), dtype=torch.float32)`

**Methods:**

- `forward(input_ids, attention_mask=None, labels=None, return_compressed_state=False, return_trajectories=False, **kwargs) -> CausalLMOutputWithPast` — Forward pass. During training, computes loss using sparse log-probabilities (avoids O(T·V) materialization). During inference, computes per-position logits in chunks.

- `generate(input_ids, max_new_tokens=1024, temperature=0.7, top_k=40, top_p=0.9, eos_token_id=None, **kwargs) -> torch.LongTensor` — Autoregressive generation. Full forward pass each step (O(1) memory — no KV-cache growth).

- `get_compressed_state(input_ids) -> torch.Tensor` — Extracts the O(1) compressed state from the SSM stack for a given input. Returns `[batch, d_state]`.

- `forward_multimodal(input_ids, images, labels=None, **kwargs) -> CausalLMOutputWithPast` — Forward pass with optional visual input. Encodes images via vision encoder and fuses with text-based compressed state.

**Properties:**
- `device -> torch.device`

**Private Methods:**
- `_log_architecture()` — Logs parameter counts and configuration.

### `MoENSLTModel(nslt_model.NSLTModel)`

Replaces SSM layers with `MoE_SSM_Block` instances. Constructor adds `n_experts` (default 8) and `top_k_experts` (default 2) parameters.

---

## `src.training.pipeline`

### `TrainingPipeline`

**Constructor:**

- `__init__(cfg: Config, dist_setup: Optional[DistributedSetup] = None) -> None` — Initializes pipeline with config, distributed setup, experiment tracker, and seed.

**Methods:**

- `initialize(fresh_start: bool = False, resume_checkpoint: Optional[str] = None) -> None` — Loads tokenizer, creates or loads model, sets up data pipeline, alignment pipeline, and experiment tracking.

- `run_pretrain(dataset: Dataset, **overrides) -> Dict[str, float]` — Runs pretraining phase using `PretrainStageConfig`.

- `run_sft(dataset: Dataset, **overrides) -> Dict[str, float]` — Runs supervised fine-tuning phase.

- `run_instruction_tuning(dataset: Dataset, **overrides) -> Dict[str, float]` — Runs instruction tuning phase.

- `run_alignment(preference_dataset: Dataset, kto_dataset: Optional[Dataset] = None) -> Dict[str, Any]` — Runs alignment phase (DPO/ORPO/SimPO/KTO).

- `run_constitutional_alignment(instructions: List[str], responses: List[str]) -> List[Dict[str, str]]` — Runs constitutional AI alignment.

- `run_safety_training() -> Dict[str, float]` — Runs safety training phase.

- `run_red_teaming() -> List[Dict[str, Any]]` — Runs red teaming.

- `run_evaluation() -> Dict[str, Any]` — Runs full evaluation suite (benchmarks + safety).

- `full_training_sequence() -> Dict[str, Any]` — Orchestrates the complete training sequence: pretrain → SFT → instruction tuning, with optional curriculum learning.

**Private Methods:**

- `_train_stage(dataset, stage_name, stage_cfg, **overrides) -> Dict[str, float]` — Builds trainer, trains, saves checkpoint, logs metrics.
- `_build_trainer(dataset, stage_name, stage_cfg, **overrides) -> Trainer` — Constructs the HuggingFace `Trainer`.
- `_cache_key(ds_info, stage_name) -> str` — Cache key hash.
- `_cached_dataset_path(ds_info, stage_name) -> Path` — Cache path construction.

---

## `scripts/production_validation.py`

### Validation Phases

- `phase1_verify_registry(token: Optional[str] = None) -> Dict[str, Any]` — Verifies all registry datasets are accessible. Returns `{"total_entries", "total_weight", "available_weight", "available_pct", "failures", "failure_details"}`.

- `phase2_build_docs(output_dir="data/docs", sources=None, max_per_source=2000) -> Dict[str, int]` — Builds documentation corpus by scraping all doc sources. Returns `{"results": {name: count}, "total_pages", "missing", "empty"}`.

- `phase3_validate_docs(output_dir="data/docs") -> Dict[str, Any]` — Validates documentation quality: checks JSONL existence, document counts, average length, duplicate detection, boilerplate assessment, metadata completeness. Returns `{"results": {source: {...}}, "total_docs", "total_chars"}`.

- `phase4_token_distribution(tokenizer_name="Xenova/claude-tokenizer", max_samples=5000, max_seq_length=2048) -> Dict[str, Any]` — Builds pretrain dataset and measures token distribution per category against targets. Returns `{"total_samples", "total_tokens", "distribution": {cat: {"tokens", "pct", "target_pct", "diff", "status"}}, "overall_status"}`.

- `phase5_decode_samples(tokenizer_name="Xenova/claude-tokenizer", num_samples=20) -> List[Dict[str, Any]]` — Decodes packed samples and inspects for issues (empty, malformed UTF-8, license spam, excessive newlines, repetitive content, missing metadata, segment mismatches, dirty padding). Returns a list of sample inspection dicts.

- `phase6_packing_quality(tokenizer_name="Xenova/claude-tokenizer", max_samples=2000) -> Dict[str, Any]` — Analyzes packing quality: average length, padding percentage, utilization, segment counts, quality scores, padding buckets, zero-segment count, missing metadata, corrupted UTF-8. Returns `{"max_seq_length", "avg_length", "avg_padding", "padding_pct", "utilization_pct", "avg_segments", "avg_quality_packed", "padding_buckets", "zero_segment_samples", "missing_metadata_samples", "corrupted_utf8_samples", "status", "samples_analyzed"}`.

- `phase7_verify_fallbacks(token: Optional[str] = None) -> Dict[str, Any]` — Tests fallback behavior for a docs entry (JSONL not built — should fall back to FineWeb-Edu). Returns `{"docs_fallback": {"status", "fallback_activated", "samples"}}`.

- `phase8_smoke_checkpoint(config_path="config_foundation.yaml", max_steps=100) -> Dict[str, Any]` — Runs training smoke test for `max_steps`, saves checkpoint, verifies checkpoint files exist, and resumes from checkpoint for 10 more steps. Returns `{"train_return_code", "train_stdout_tail", "train_stderr_tail", "checkpoint_files", "checkpoint_found", "resume_return_code", "status"}`.

- `generate_report(all_results: Dict[str, Any]) -> str` — Generates a formatted production readiness report from all phase results. Includes decision (READY / READY WITH MINOR WARNINGS / NOT READY).

- `main()` — Entry point. Parses args, runs all 8 phases, generates report. Returns exit code 0.

---

## `scripts/verify_datasets.py`

### Functions

- `try_load_sample(path: str, name: Optional[str] = None, split: str = "train", timeout_sec: int = 60, max_samples: int = 5, token: Optional[str] = None) -> Dict[str, Any]` — Attempts to load samples from a dataset. Returns a result dict with `path`, `name`, `split`, `status` (ok/gated/not_found/empty/timeout/error/no_split/no_config), `error`, `loaded`, `columns`, `avg_text_len`, `time_sec`.

- `verify_registry(registry: DatasetRegistry, token: Optional[str] = None, max_per_dataset: int = 5) -> List[Dict[str, Any]]` — Verifies all registry entries. Skips JSON datasets (status `local_json`) and deduplicates by path/name. Returns list of result dicts.

- `report(results: List[Dict[str, Any]]) -> None` — Prints formatted availability report by category with weights.

- `main()` — CLI entry point with `--token`, `--max-samples`, `--report` args. Exits with code 1 if any failures exist.
