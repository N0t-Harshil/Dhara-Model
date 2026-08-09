# Specialized Coding Model — Engineering Documentation Package

This directory contains comprehensive engineering documentation for the Specialized Coding Model project.

## Quick Navigation

| Document | What It Covers |
|----------|---------------|
| [01_Project_Overview.md](01_Project_Overview.md) | What this project is, why it exists, high-level architecture |
| [02_System_Architecture.md](02_System_Architecture.md) | Complete architecture with Mermaid diagrams |
| [03_Codebase_Map.md](03_Codebase_Map.md) | Every file and directory explained |
| [04_Component_Documentation.md](04_Component_Documentation.md) | Deep docs for every major Python file |
| [05_Data_Pipeline.md](05_Data_Pipeline.md) | Registry → Streaming → Filtering → Packing → Training |
| [06_Documentation_Builder.md](06_Documentation_Builder.md) | Web scraping for 18 documentation sources |
| [07_Training_Pipeline.md](07_Training_Pipeline.md) | Config, training stages, checkpointing |
| [08_Dataset_Registry.md](08_Dataset_Registry.md) | All 55 datasets with weights and metadata |
| [09_Validation_System.md](09_Validation_System.md) | 8-phase automated validation |
| [10_Known_Issues.md](10_Known_Issues.md) | 18 issues with root causes and fixes |
| [11_Future_Work.md](11_Future_Work.md) | Roadmap from critical to research ideas |
| [12_Configuration_Guide.md](12_Configuration_Guide.md) | Parameter documentation |
| [13_Debugging_Guide.md](13_Debugging_Guide.md) | Troubleshooting flowcharts and fixes |
| [14_Performance_Analysis.md](14_Performance_Analysis.md) | Bottlenecks and optimization priorities |
| [15_API_Reference.md](15_API_Reference.md) | Complete API reference |
| [16_Project_Status.md](16_Project_Status.md) | Feature completion assessment |
| [17_Release_Readiness_Report.md](17_Release_Readiness_Report.md) | Production readiness (10 categories scored) |
| [18_Engineer_Onboarding_Guide.md](18_Engineer_Onboarding_Guide.md) | Day 1 checklist and common tasks |

## Executive Summary

**What problem does this project solve?**
It provides a complete, production-grade framework for pretraining and aligning large language models with novel SSM-based architectures (MethosV3, NSLT). The framework handles everything from dataset acquisition and quality filtering through distributed training and evaluation.

**How does the system work end-to-end?**
1. Configuration (YAML → Pydantic schema)
2. Dataset registry builds 55-entry catalog across 8 categories
3. Streaming loads datasets with automatic fallback chains
4. Quality pipeline filters, deduplicates, and scores documents
5. Code-specific processing (AST filtering, function sampling)
6. Weighted mixed dataset with adaptive resampling
7. Sequence packing with metadata tracking
8. Tokenization → DataLoader → Training → Checkpointing → Validation

**What are its strongest architectural features?**
- Production-grade data pipeline with registry system, fallback chains, and 8-phase validation
- O(1) memory SSM architecture (no KV-cache scaling issues)
- Novel workspace-centric reasoning paradigm with 11 specialized layers
- Config-driven design with Pydantic validation
- Comprehensive documentation scraping system (18 sources)

**What are its current weaknesses?**
- Incomplete test coverage (many placeholders)
- Training pipeline not thoroughly tested at production scale
- Some legacy/deprecated code (MassiveDataCollector)
- Multi-GPU training validation incomplete
- HF token management (stored in config, not secret manager)

**What should a new engineer learn first?**
1. The data pipeline (src/data/) — it's the most polished and critical subsystem
2. The config system (src/config/schema.py) — controls everything
3. The validation system (scripts/production_validation.py) — how to verify changes

**What issues should be fixed before the next production release?**
1. Complete test coverage (high priority)
2. Fix all production validation failures
3. Track config_foundation.yaml in version control
4. Implement CI/CD for automated validation
5. Clean up deprecated code and duplicate scripts
6. Verify end-to-end training on target hardware

**Production Readiness Level: 6.5/10 (Late Beta)**
- Data pipeline: 8/10 — production-ready
- Model architecture: 7/10 — implemented, HF compatible
- Training infrastructure: 5/10 — scaffolded, needs testing
- Testing: 4/10 — significant gaps
- Documentation: 8/10 — comprehensive
- Security: 5/10 — not explicitly addressed