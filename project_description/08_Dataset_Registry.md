# Dataset Registry

## Overview

The dataset registry (`src/data/registry.py`) defines 55 dataset entries across 8 categories. Each entry is a `DatasetInfo` dataclass specifying the HuggingFace path, category, weight (fraction of total training tokens), quality score, fallback chains, language, domain, text fields, and licensing. The registry is built by `build_registry()` which calls 8 registration helpers, one per category, and normalizes weights to sum to 1.0.

## Category Weight Targets

| Category | Weight | Description |
|----------|--------|-------------|
| code | 0.30 | Programming code across 19 languages |
| web_text | 0.20 | General web text (FineWeb) |
| docs | 0.15 | Documentation (18 sources + FineWeb-Edu fallback) |
| wiki | 0.10 | Wikipedia (English, 20231101) |
| math | 0.10 | Mathematical reasoning and proofs |
| science | 0.05 | Scientific literature proxy |
| books | 0.05 | Books proxy |
| structured_knowledge | 0.05 | Structured knowledge proxy |

## Code (30%)

### OpenCoder-LLM/opc-fineweb-code-corpus
- **Path**: `OpenCoder-LLM/opc-fineweb-code-corpus`
- **Category**: code
- **Weight**: 0.045 (15% of code category)
- **Quality score**: 0.90
- **Domain**: algorithms
- **Text fields**: text
- **License**: MIT
- **Fallbacks**: code-search-net/code_search_net, codeparrot/codeparrot-clean
- **Priority**: 10
- **Notes**: Replaces previously removed FineWeb-Code. Provides high-quality permissively licensed code.

### bigcode/the-stack-v2-dedup (18 languages)
- **Path**: `bigcode/the-stack-v2-dedup`
- **Category**: code
- **Weight**: ~0.09 total (30% of code category, distributed across languages)
- **Quality score**: 0.88
- **Domain**: backend
- **Text fields**: content, text
- **License**: various
- **Priority**: 5
- **Function sampling**: enabled
- **Languages**: Python (0.25), C++ (0.15), JavaScript (0.10), TypeScript (0.08), Java (0.08), Rust (0.07), Go (0.06), SQL (0.05), Shell (0.04), C# (0.04), PHP (0.03), C (0.02), Kotlin (0.01), Swift (0.01), Scala (0.01), R (0.01), Julia (0.01), Lua (0.01), Objective-C (0.01)
- **Language targets**: Defined in `CODE_LANG_TARGETS` dict at `registry.py:157`
- **Fallbacks**: code-search-net/code_search_net, codeparrot/codeparrot-clean (sql: b-mc2/sql-create-context)
- **Notes**: 19 dedicated language entries. Individual language weights are proportional to `CODE_LANG_TARGETS`. Each language has its own registry entry with `name=lang_name` for targeted sampling.

### code-search-net/code_search_net
- **Path**: `code-search-net/code_search_net` (namespaced id of the legacy `code_search_net` repo — newer datasets/huggingface_hub reject namespace-less ids in `hf://` URIs; same repo, same data)
- **Category**: code
- **Weight**: 0.06 (20% of code category)
- **Quality score**: 0.92
- **Domain**: backend
- **Text fields**: code, func_code_string, whole_func_string, text
- **License**: various
- **Priority**: 10
- **Function sampling**: enabled
- **Notes**: Code-search dataset with function-level samples across multiple languages.

### codeparrot/codeparrot-clean
- **Path**: `codeparrot/codeparrot-clean`
- **Category**: code
- **Weight**: 0.06 (20% of code category)
- **Quality score**: 0.84
- **Language**: python
- **Domain**: backend
- **Text fields**: content, text
- **License**: MIT
- **Priority**: 10
- **Function sampling**: enabled
- **Notes**: Cleaned Python code dataset from GitHub. Used as a fallback target for other code datasets.

### deepmind/code_contests
- **Path**: `deepmind/code_contests`
- **Category**: code
- **Weight**: 0.045 (15% of code category)
- **Quality score**: 0.90
- **Domain**: algorithms
- **Text fields**: description, solutions, text
- **License**: Apache-2.0
- **Priority**: 20
- **Fallbacks**: codeparrot/codeparrot-clean
- **Notes**: Competitive programming dataset with problem descriptions, solutions, and test cases.

## Web Text (20%)

### HuggingFaceFW/fineweb
- **Path**: `HuggingFaceFW/fineweb`
- **Category**: web_text
- **Weight**: 0.20
- **Quality score**: 0.93
- **Domain**: web
- **Text fields**: text
- **License**: ODC-BY
- **Priority**: 1
- **Fallbacks**: allenai/dolma, togethercomputer/RedPajama-Data-1T, tiiuae/falcon-refinedweb
- **Notes**: The primary web text corpus. 3-tier fallback chain for resilience.

## Documentation (15%)

### 18 Doc Builder Sources (JSONL format)
- **Path**: `json`
- **Category**: docs
- **Quality scores**: 0.90–0.99 (per source)
- **License**: various
- **Priority**: 5
- **Fallbacks**: HuggingFaceFW/fineweb-edu
- **Notes**: Each of the 18 scraped doc sources gets its own registry entry with `data_dir` pointing to the JSONL output directory. Weights are proportional to `DOC_SOURCES` weights:

| Source | Language | Weight Fraction | Quality Score | Domain |
|--------|----------|----------------|---------------|--------|
| python-docs | python | 0.020 | 0.99 | backend |
| pytorch-docs | python | 0.020 | 0.99 | ml |
| numpy-docs | python | 0.010 | 0.98 | ml |
| rust-book | rust | 0.010 | 0.98 | backend |
| go-docs | go | 0.010 | 0.97 | backend |
| mdn-docs | javascript | 0.010 | 0.95 | web |
| fastapi-docs | python | 0.005 | 0.96 | backend |
| cuda-docs | cpp | 0.005 | 0.95 | ml |
| linux-kernel-docs | text | 0.005 | 0.94 | os |
| opencv-docs | cpp | 0.005 | 0.93 | cv |
| kubernetes-docs | shell | 0.005 | 0.92 | cloud |
| docker-docs | shell | 0.005 | 0.91 | cloud |
| postgresql-docs | sql | 0.003 | 0.92 | backend |
| sqlite-docs | sql | 0.003 | 0.91 | backend |
| cudnn-docs | cpp | 0.003 | 0.92 | ml |
| onnx-docs | python | 0.003 | 0.91 | ml |
| rfcs | text | 0.003 | 0.90 | protocols |
| lang-specs | text | 0.003 | 0.90 | languages |

### HuggingFaceFW/fineweb-edu (docs fallback)
- **Path**: `HuggingFaceFW/fineweb-edu`
- **Name**: sample-10BT
- **Category**: docs
- **Weight**: ~0.02 (3% of docs category)
- **Quality score**: 0.92
- **Domain**: general
- **Text fields**: text
- **License**: ODC-BY
- **Priority**: 10
- **Notes**: Acts as fallback for all 18 doc sources when JSONL files are not yet built.

## Wiki (10%)

### wikimedia/wikipedia 20231101.en
- **Path**: `wikimedia/wikipedia`
- **Name**: 20231101.en
- **Category**: wiki
- **Weight**: 0.10
- **Quality score**: 0.95
- **Domain**: web
- **Text fields**: text
- **License**: CC-BY-SA
- **Priority**: 1
- **Notes**: English Wikipedia dump from November 2023. Single largest non-weight-limited entry.

## Math (10%)

### open-web-math/open-web-math
- **Path**: `open-web-math/open-web-math`
- **Category**: math
- **Weight**: 0.042 (42% of math category)
- **Quality score**: 0.92
- **Domain**: science
- **Text fields**: text, problem, solution
- **License**: various
- **Priority**: 10
- **Notes**: Largest math dataset. General mathematical web content.

### AI-MO/NuminaMath-CoT
- **Path**: `AI-MO/NuminaMath-CoT`
- **Category**: math
- **Weight**: 0.02 (20% of math category)
- **Quality score**: 0.90
- **Domain**: science
- **Text fields**: text, problem, solution
- **Fallbacks**: AI-MO/NuminaMath-1.5
- **License**: various
- **Notes**: Chain-of-thought math reasoning dataset.

### AI-MO/NuminaMath-1.5
- **Path**: `AI-MO/NuminaMath-1.5`
- **Category**: math
- **Weight**: 0.015 (15% of math category)
- **Quality score**: 0.91
- **Domain**: science
- **Text fields**: text, problem, solution
- **Fallbacks**: AI-MO/NuminaMath-CoT
- **License**: various
- **Notes**: Math dataset v1.5. Bidirectional fallback with NuminaMath-CoT.

### GAIR/MathPile
- **Path**: `GAIR/MathPile`
- **Category**: math
- **Weight**: 0.010 (10% of math category)
- **Quality score**: 0.93
- **Domain**: science
- **Text fields**: text, problem, solution
- **Fallbacks**: open-web-math/open-web-math
- **License**: various
- **Authentication**: REQUIRED — gated dataset, must request access at HuggingFace
- **Priority**: 10
- **Notes**: ⚠️ GATED — will fail if HF token does not have approved access.

### HuggingFaceFW/fineweb-edu sample-10BT (math proxy)
- **Path**: `HuggingFaceFW/fineweb-edu`
- **Name**: sample-10BT
- **Category**: math
- **Weight**: 0.01 (10% of math category)
- **Quality score**: 0.85
- **Domain**: science
- **Text fields**: text, problem, solution
- **Fallbacks**: open-web-math/open-web-math
- **Notes**: Educational web text used as math content proxy.

### akjadhav/leandojo-lean4-formal-informal-strings-split
- **Path**: `akjadhav/leandojo-lean4-formal-informal-strings-split`
- **Category**: math
- **Weight**: 0.003 (3% of math category)
- **Quality score**: 0.93
- **Domain**: science
- **Text fields**: text, problem, solution
- **Fallbacks**: open-web-math/open-web-math
- **License**: various
- **Notes**: Lean 4 formal-informal math strings. Provides formal theorem proving data.

## Science (5%)

### HuggingFaceFW/fineweb-edu sample-10BT (science proxy)
- **Path**: `HuggingFaceFW/fineweb-edu`
- **Name**: sample-10BT
- **Category**: science
- **Weight**: 0.05
- **Quality score**: 0.85
- **Domain**: science
- **Text fields**: text, title, abstract
- **Notes**: All science content is proxied through FineWeb-Edu. S2ORC, PubMed, OpenAlex, and ACL Anthology were removed due to compatibility issues with datasets 2.20+.

## Books (5%)

### HuggingFaceFW/fineweb-edu sample-10BT (books proxy)
- **Path**: `HuggingFaceFW/fineweb-edu`
- **Name**: sample-10BT
- **Category**: books
- **Weight**: 0.025 (50% of books category)
- **Quality score**: 0.88
- **Domain**: web
- **Text fields**: text, content
- **Priority**: 10
- **Long context**: true
- **Notes**: Educational web text used as books proxy.

### HuggingFaceFW/fineweb (books proxy)
- **Path**: `HuggingFaceFW/fineweb`
- **Category**: books
- **Weight**: 0.015 (30% of books category, adjusted from target 50%)
- **Quality score**: 0.85
- **Domain**: web
- **Text fields**: text, content
- **Priority**: 10
- **Long context**: true
- **Notes**: General web text used as books proxy. PG19 and Gutenberg were removed due to compatibility issues.

## Structured Knowledge (5%)

### HuggingFaceFW/fineweb-edu sample-10BT (knowledge proxy)
- **Path**: `HuggingFaceFW/fineweb-edu`
- **Name**: sample-10BT
- **Category**: structured_knowledge
- **Weight**: 0.025 (50% of structured_knowledge category)
- **Quality score**: 0.85
- **Domain**: web
- **Text fields**: text, description, title
- **Priority**: 15
- **Notes**: Educational web text used as structured knowledge proxy.

### HuggingFaceFW/fineweb (knowledge proxy)
- **Path**: `HuggingFaceFW/fineweb`
- **Category**: structured_knowledge
- **Weight**: 0.025 (50% of structured_knowledge category)
- **Quality score**: 0.82
- **Domain**: web
- **Text fields**: text, description, title
- **Priority**: 15
- **Notes**: General web text used as structured knowledge proxy. WIT, Wikidata, DBpedia, ConceptNet, and WordNet were removed due to compatibility issues.

## Weight Normalization

After all entries are registered, `DatasetRegistry.normalize_weights()` (`src/data/registry.py:117`) ensures each category's total weight matches the `CATEGORY_WEIGHTS` targets. The normalization:
1. Groups entries by category
2. Calculates the current sum of weights per category
3. Scales each entry's weight by `target / current`
4. Adjusts the last entry in each category to absorb floating-point rounding errors
5. Reports the final normalized total (should be 1.0)

## Fallback System

When a dataset fails to load (gated, 404, network error), the `stream_dataset_with_fallbacks()` function (`src/data/streaming.py:96`) iterates through the fallback chain:
1. Primary dataset → fallback[0] → fallback[1] → ...
2. First successful dataset wins
3. Fallback usage is logged via `registry.log_fallback()`
4. If all fallbacks fail, logs an error and yields nothing

Current fallback chains:
- `fineweb` → dolma → RedPajama → Falcon RefinedWeb
- `NuminaMath-CoT` ↔ `NuminaMath-1.5` (bidirectional)
- `MathPile` → open-web-math
- `code_contests` → codeparrot-clean
- All 18 doc sources → FineWeb-Edu
