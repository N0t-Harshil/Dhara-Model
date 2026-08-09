from __future__ import annotations

import json
import logging
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PASS = 0
FAIL = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASS, FAIL
    if condition:
        print(f"  OK  {name}")
        PASS += 1
    else:
        suffix = f" -- {detail}" if detail else ""
        print(f"  FAIL {name}{suffix}")
        FAIL += 1


def test_missing_directory() -> None:
    print("\n--- Missing Local Directory ---")
    from scripts.verify_datasets import _verify_local_dataset
    from src.data.registry import DatasetInfo

    info = DatasetInfo("json", "docs", 0.01, 0.9, name="test",
                        data_dir="/nonexistent/path/that/definitely/does/not/exist")
    result = _verify_local_dataset(info)
    check("status is missing_local", result["status"] == "missing_local",
         f"got {result['status']}: {result['error']}")


def test_missing_jsonl_file() -> None:
    print("\n--- Missing JSONL File ---")
    from scripts.verify_datasets import _verify_local_dataset
    from src.data.registry import DatasetInfo

    with tempfile.TemporaryDirectory() as tmpdir:
        info = DatasetInfo("json", "docs", 0.01, 0.9, name="test", data_dir=tmpdir)
        result = _verify_local_dataset(info)
        check("status is missing_local (no jsonl)", result["status"] == "missing_local",
             f"got {result['status']}: {result['error']}")


def test_empty_jsonl() -> None:
    print("\n--- Empty JSONL File ---")
    from scripts.verify_datasets import _verify_local_dataset
    from src.data.registry import DatasetInfo

    with tempfile.TemporaryDirectory() as tmpdir:
        Path(tmpdir, "documents.jsonl").write_text("", encoding="utf-8")
        info = DatasetInfo("json", "docs", 0.01, 0.9, name="test", data_dir=tmpdir)
        result = _verify_local_dataset(info)
        check("status is empty_local", result["status"] == "empty_local",
             f"got {result['status']}: {result['error']}")


def test_malformed_jsonl_high_ratio() -> None:
    """99 valid + 1 invalid = 99% ratio -> warning_local (fails 95% threshold due to error)."""
    print("\n--- Malformed JSONL: 99% valid, 1 error ---")
    from scripts.verify_datasets import _verify_local_dataset
    from src.data.registry import DatasetInfo

    with tempfile.TemporaryDirectory() as tmpdir:
        lines = [json.dumps({"text": f"doc {i}", "source": "t", "title": "T", "url": "http://x.com"})
                 for i in range(99)]
        lines.append('{"text": "broken}')
        Path(tmpdir, "documents.jsonl").write_text(
            "\n".join(lines) + "\n", encoding="utf-8")
        info = DatasetInfo("json", "docs", 0.01, 0.9, name="test", data_dir=tmpdir)
        result = _verify_local_dataset(info)
        check("status is warning_local (error present)", result["status"] == "warning_local",
             f"got {result['status']}: {result['error']}")
        check("record_count == 99", result["record_count"] == 99,
             f"got {result['record_count']}")
        check("valid_ratio == 0.99", abs(result.get("valid_ratio", 0) - 0.99) < 0.01,
             f"got {result.get('valid_ratio')}")


def test_malformed_jsonl_medium_ratio() -> None:
    """50 valid + 50 invalid = 50% ratio -> warning_local (>= 50%)."""
    print("\n--- Malformed JSONL: 50% valid ---")
    from scripts.verify_datasets import _verify_local_dataset
    from src.data.registry import DatasetInfo

    with tempfile.TemporaryDirectory() as tmpdir:
        lines = [json.dumps({"text": f"doc {i}", "source": "t", "title": "T", "url": "http://x.com"})
                 for i in range(50)]
        lines.extend(['garbage'] * 50)
        Path(tmpdir, "documents.jsonl").write_text(
            "\n".join(lines) + "\n", encoding="utf-8")
        info = DatasetInfo("json", "docs", 0.01, 0.9, name="test", data_dir=tmpdir)
        result = _verify_local_dataset(info)
        check("status is warning_local", result["status"] == "warning_local",
             f"got {result['status']}: {result['error']}")
        check("record_count == 50", result["record_count"] == 50,
             f"got {result['record_count']}")
        check("valid_ratio == 0.50", abs(result.get("valid_ratio", 0) - 0.50) < 0.01,
             f"got {result.get('valid_ratio')}")


def test_malformed_jsonl_low_ratio() -> None:
    """3 valid + 97 invalid = 3% ratio -> invalid_local."""
    print("\n--- Malformed JSONL: 3% valid ---")
    from scripts.verify_datasets import _verify_local_dataset
    from src.data.registry import DatasetInfo

    with tempfile.TemporaryDirectory() as tmpdir:
        lines = [json.dumps({"text": f"doc {i}", "source": "t", "title": "T", "url": "http://x.com"})
                 for i in range(3)]
        lines.extend(['garbage'] * 97)
        Path(tmpdir, "documents.jsonl").write_text(
            "\n".join(lines) + "\n", encoding="utf-8")
        info = DatasetInfo("json", "docs", 0.01, 0.9, name="test", data_dir=tmpdir)
        result = _verify_local_dataset(info)
        check("status is invalid_local", result["status"] == "invalid_local",
             f"got {result['status']}: {result['error']}")
        check("record_count == 3", result["record_count"] == 3,
             f"got {result['record_count']}")
        check("valid_ratio < 0.50", result.get("valid_ratio", 1) < 0.50,
             f"got {result.get('valid_ratio')}")


def test_invalid_utf8() -> None:
    print("\n--- Invalid UTF-8 ---")
    from scripts.verify_datasets import _verify_local_dataset
    from src.data.registry import DatasetInfo

    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir, "documents.jsonl")
        p.write_bytes(b'\xff\xfe\x00{"text": "bad"}\n')
        info = DatasetInfo("json", "docs", 0.01, 0.9, name="test", data_dir=tmpdir)
        result = _verify_local_dataset(info)
        check("status is invalid_local", result["status"] == "invalid_local",
             f"got {result['status']}: {result['error']}")
        check("reports UTF-8 error", "UTF-8" in result.get("error", ""),
             f"error: {result.get('error')}")


def test_warning_schema() -> None:
    """100% valid JSON ratio, but missing required fields -> warning_local."""
    print("\n--- Schema Warning (missing fields, 100% valid ratio) ---")
    from scripts.verify_datasets import _verify_local_dataset
    from src.data.registry import DatasetInfo

    with tempfile.TemporaryDirectory() as tmpdir:
        rec = {"id": 1, "content": "no text field", "not_source": "x", "not_title": "y"}
        Path(tmpdir, "documents.jsonl").write_text(
            json.dumps(rec) + "\n", encoding="utf-8")
        info = DatasetInfo("json", "docs", 0.01, 0.9, name="test", data_dir=tmpdir)
        result = _verify_local_dataset(info)
        check("status is warning_local (schema warning)", result["status"] == "warning_local",
             f"got {result['status']}: {result['error']}")
        check("reports missing fields", "missing required fields" in result.get("error", ""),
             f"error: {result.get('error')}")
        check("valid_ratio == 1.0", abs(result.get("valid_ratio", 0) - 1.0) < 0.01,
             f"got {result.get('valid_ratio')}")


def test_valid_jsonl() -> None:
    """100% valid ratio, no errors -> ok_local."""
    print("\n--- Valid JSONL ---")
    from scripts.verify_datasets import _verify_local_dataset
    from src.data.registry import DatasetInfo

    with tempfile.TemporaryDirectory() as tmpdir:
        recs = [
            {"text": "doc " * 50, "source": "test", "title": "Doc 1", "url": "http://example.com/1"},
            {"text": "doc " * 60, "source": "test", "title": "Doc 2", "url": "http://example.com/2"},
        ]
        Path(tmpdir, "documents.jsonl").write_text(
            "\n".join(json.dumps(r) for r in recs) + "\n", encoding="utf-8")
        info = DatasetInfo("json", "docs", 0.01, 0.9, name="test", data_dir=tmpdir)
        result = _verify_local_dataset(info)
        check("status is ok_local", result["status"] == "ok_local",
             f"got {result['status']}: {result['error']}")
        check("record_count == 2", result["record_count"] == 2,
             f"got {result['record_count']}")
        check("avg_text_len > 0", result["avg_text_len"] > 0,
             f"got {result['avg_text_len']}")
        check("columns include text", "text" in result.get("columns", []),
             f"columns: {result.get('columns')}")
        check("valid_ratio == 1.0", abs(result.get("valid_ratio", 0) - 1.0) < 0.01,
             f"got {result.get('valid_ratio')}")


def test_duplicate_accounting() -> None:
    print("\n--- Duplicate Dataset Accounting ---")
    from scripts.verify_datasets import report

    results = [
        {"path": "ds1", "name": "", "category": "science", "status": "ok",
         "weight": 0.03, "loaded": 3, "columns": ["text"], "avg_text_len": 100},
        {"path": "ds1", "name": "", "category": "books", "status": "skipped_dup",
         "weight": 0.02, "error": "duplicate"},
        {"path": "ds3", "name": "", "category": "math", "status": "ok",
         "weight": 0.05, "loaded": 3, "columns": ["text"], "avg_text_len": 200},
    ]

    w_ok = sum(r["weight"] for r in results if r["status"] in ("ok", "skipped_dup"))
    check("duplicate weight counted in ok", abs(w_ok - 0.10) < 0.001,
         f"got {w_ok}")

    skipped = [r for r in results if r["status"] == "skipped_dup"]
    check("skipped_dup entry exists", len(skipped) == 1)
    check("skipped_dup entry has weight", abs(skipped[0]["weight"] - 0.02) < 0.001)


def test_category_weight_accounting() -> None:
    print("\n--- Category Weight Accounting ---")
    from scripts.verify_datasets import report

    results = [
        {"path": "json", "name": "missing", "category": "docs",
         "status": "missing_local", "weight": 0.10, "error": "not found"},
        {"path": "json", "name": "also-missing", "category": "docs",
         "status": "missing_local", "weight": 0.05, "error": "not found"},
    ]

    w_ok = sum(r["weight"] for r in results if r["status"] == "ok_local")
    w_fail = sum(r["weight"] for r in results if r["status"] != "ok_local")

    check("missing docs have 0 ok_local weight", w_ok == 0.0, f"got {w_ok}")
    check("missing docs have full fail weight", abs(w_fail - 0.15) < 0.001,
         f"got {w_fail}")


def test_warning_local_weight() -> None:
    """warning_local entries should NOT count as available weight."""
    print("\n--- Warning Local Weight Accounting ---")
    from scripts.verify_datasets import report

    results = [
        {"path": "json", "name": "partial", "category": "docs",
         "status": "warning_local", "weight": 0.07, "error": "50% valid"},
        {"path": "json", "name": "good", "category": "docs",
         "status": "ok_local", "weight": 0.08, "error": ""},
    ]

    w_ok = sum(r["weight"] for r in results if r["status"] in ("ok", "skipped_dup", "ok_local"))
    w_fail = sum(r["weight"] for r in results if r["status"] not in ("ok", "skipped_dup", "ok_local"))

    check("warning_local NOT counted in ok", abs(w_ok - 0.08) < 0.001,
         f"got {w_ok}")
    check("warning_local counted in fail", abs(w_fail - 0.07) < 0.001,
         f"got {w_fail}")


def test_doc_builder_zero_pages_report() -> None:
    print("\n--- Doc Builder 0 Pages Warning ---")
    from src.data.doc_builder import PythonDocScraper

    scraper = PythonDocScraper()
    summary = scraper._summary(0)
    check("summary includes 0 pages", "0 pages" in summary)
    check("summary excludes discovered when 0", "discovered=" not in summary)


def test_report_skipped_dup_in_ok_column() -> None:
    print("\n--- Report: skipped_dup counts as OK ---")
    from scripts.verify_datasets import report
    import io

    results = [
        {"path": "real/ds", "name": "", "category": "science", "status": "ok",
         "weight": 0.03},
        {"path": "real/ds", "name": "", "category": "books", "status": "skipped_dup",
         "weight": 0.02},
        {"path": "json", "name": "local", "category": "docs", "status": "ok_local",
         "weight": 0.05},
    ]

    buffer = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buffer
    try:
        report(results)
    finally:
        sys.stdout = old_stdout

    output = buffer.getvalue()
    check("science shows 1 OK+Dup", "science" in output)
    check("books OK+Dup > 0", "books" in output)
    check("Total weight available > 0", "Total weight available: 0.10" in output)


def test_stage_boundary_callback_behavior() -> None:
    print("\n--- StageBoundaryCallback Behavior ---")
    from src.training.pipeline import StageBoundaryCallback

    class Control:
        def __init__(self):
            self.should_save = False
            self.should_training_stop = False

    class State:
        def __init__(self, gs):
            self.global_step = gs
            self.is_world_process_zero = True

    cb = StageBoundaryCallback(1, 2, "code", end_step=100)
    c = Control()
    cb.on_step_end(None, State(50), c)
    check("no stop before boundary", c.should_training_stop is False)
    check("no save before boundary", c.should_save is False)

    c2 = Control()
    cb.on_step_end(None, State(100), c2)
    check("stop at boundary", c2.should_training_stop is True)
    check("save at boundary", c2.should_save is True)

    cb.on_train_begin(None, None, c2)
    check("flags reset on train begin",
          c2.should_training_stop is False and c2.should_save is False)


class _FakeTokenizer:
    eos_token_id = 0
    name_or_path = "fake-tokenizer"
    vocab_size = 32000


def _make_stage_cfg(name="code", categories=("code",), steps=100,
                    max_samples=None, weights=None):
    class SC:
        pass
    sc = SC()
    sc.name = name
    sc.categories = list(categories)
    sc.weights = weights
    sc.steps = steps
    sc.max_samples_per_dataset = max_samples
    return sc


def test_stage_cache_key_sensitivity() -> None:
    print("\n--- Stage Cache Key Sensitivity ---")
    from src.config.schema import Config
    from src.data.pipeline import DataPipeline

    cfg = Config()
    pipe = DataPipeline(cfg, _FakeTokenizer())

    k1 = pipe._stage_cache_key(_make_stage_cfg(), 1)
    k2 = pipe._stage_cache_key(_make_stage_cfg(categories=("code", "docs")), 1)
    check("category change invalidates key", k2 != k1)

    k3 = pipe._stage_cache_key(_make_stage_cfg(steps=200), 1)
    check("steps change invalidates key", k3 != k1)

    k4 = pipe._stage_cache_key(_make_stage_cfg(max_samples=500), 1)
    check("max_samples change invalidates key", k4 != k1)

    k5 = pipe._stage_cache_key(_make_stage_cfg(weights={"code": 0.5}), 1)
    check("weights change invalidates key", k5 != k1)

    k6 = pipe._stage_cache_key(_make_stage_cfg(), 2)
    check("stage index change invalidates key", k6 != k1)

    cfg.data.quality.deduplication.threshold = 0.99
    k7 = pipe._stage_cache_key(_make_stage_cfg(), 1)
    check("config change invalidates key", k7 != k1)


def test_stage_dataset_cache_roundtrip() -> None:
    print("\n--- Stage Dataset Cache Round Trip ---")
    from datasets import Dataset
    from src.config.schema import Config
    from src.data.pipeline import DataPipeline

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = Config()
        cfg.training.pretrain.staging.stage_cache_dir = tmpdir
        cfg.data.use_packed_cache = True
        pipe = DataPipeline(cfg, _FakeTokenizer())

        entries = [
            (Dataset.from_list([{"input_ids": [1, 2, 3], "quality_score": 0.9}]),
             1.0, "ds_a", "code", 0.8),
        ]
        metas = [{"loaded": 10, "accepted": 5, "avg_qs": 0.8}]

        calls = {"n": 0}

        def fake_registry_builder(**kw):
            calls["n"] += 1
            result = pipe._construct_mixed_dataset(entries)
            result._entries = entries
            result._dataset_metas = metas
            result._global_stats = {
                "total_packed": 1, "total_tokens": 3, "total_raw": 10,
                "total_accepted": 5, "total_rejected_boilerplate": 2,
                "total_rejected_short": 1, "total_rejected_quality": 1,
                "total_rejected_dup": 1, "total_rejected_ast": 0,
                "total_rejected_empty": 0,
            }
            result._lang_dist = {"en": 5}
            result._domain_dist = {"code": 5}
            result._rejection_reasons = {}
            result._fallbacks_used = {}
            return result

        pipe.build_pretrain_dataset_from_registry = fake_registry_builder

        stage_cfg = _make_stage_cfg()
        ds1, meta1 = pipe.build_pretrain_stage_dataset(stage_cfg, 1, 8)
        check("registry built on miss", calls["n"] == 1)
        check("packed.pt written", (Path(tmpdir) / "stage1" / "packed.pt").exists())
        check("stage_meta.json written", (Path(tmpdir) / "stage1" / "stage_meta.json").exists())

        ds2, meta2 = pipe.build_pretrain_stage_dataset(stage_cfg, 1, 8)
        check("cache hit, 0 registry calls", calls["n"] == 1)
        check("packed content identical",
              ds2._entries[0][0][0]["input_ids"] == [1, 2, 3])
        check("per-dataset meta replayed", ds2._dataset_metas[0]["avg_qs"] == 0.8)


def test_stage_dataset_passes_filter() -> None:
    print("\n--- Stage Dataset Category Filter ---")
    from src.config.schema import Config
    from src.data.pipeline import DataPipeline

    cfg = Config()
    pipe = DataPipeline(cfg, _FakeTokenizer())
    captured = {}

    def raising_builder(**kw):
        captured.update(kw)
        raise RuntimeError("stop after capturing kwargs")

    pipe.build_pretrain_dataset_from_registry = raising_builder

    stage_cfg = _make_stage_cfg(categories=("code", "docs"), max_samples=123)
    try:
        pipe.build_pretrain_stage_dataset(stage_cfg, 2, 8)
    except RuntimeError:
        pass
    check("dataset_filter == stage categories",
          captured.get("dataset_filter") == ["code", "docs"])
    check("max_samples passed through",
          captured.get("max_samples_per_dataset") == 123)


class _FakeTrainer:
    def __init__(self, checkpoint_dir=None):
        self.train_dataset = None
        self.train_calls = []
        self.saved = False
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None

    def add_callback(self, cb):
        self.callback = cb

    def remove_callback(self, cb):
        pass

    def train(self, resume_from_checkpoint=None):
        self.train_calls.append(resume_from_checkpoint)
        if self.checkpoint_dir is not None:
            ckpt = self.checkpoint_dir / f"checkpoint-{len(self.train_calls)}"
            ckpt.mkdir(parents=True, exist_ok=True)
            (ckpt / "dummy").write_text("x")
        from types import SimpleNamespace
        return SimpleNamespace(metrics={"loss": 0.1, "train_loss": 0.1})

    def save_model(self, path):
        self.saved = True


class _FakeDist:
    def is_main_process(self):
        return True

    def is_distributed(self):
        return False

    def get_training_args(self, output_dir):
        return {}


def _staged_cfg(tmpdir, stage_dir=None, mode="stage", steps=(3, 4)):
    from src.config.schema import Config, PretrainStageGroupConfig
    cfg = Config()
    cfg.output.model_dir = tmpdir
    cfg.training.pretrain.staging.enabled = True
    cfg.training.pretrain.staging.mode = mode
    cfg.training.pretrain.staging.stage_cache_dir = stage_dir or str(Path(tmpdir) / "stages")
    cfg.training.pretrain.staging.stages = [
        PretrainStageGroupConfig(name="code", categories=["code"], steps=steps[0]),
        PretrainStageGroupConfig(name="math", categories=["math"], steps=steps[1]),
    ]
    return cfg


def _stub_pipeline(pipe, fake_trainer, stage_calls, unit_calls=None):
    class FakeTok:
        def save_pretrained(self, path):
            pass

    class FakeDs:
        def __init__(self, n):
            self.n = n

        def __len__(self):
            return self.n

    class FakeDataPipeline:
        def build_pretrain_stage_dataset(self, stage_cfg, i, total):
            stage_calls.append((stage_cfg.name, i, total))
            return FakeDs(10), {}

        def build_pretrain_dataset_unit(self, info, i, j, total):
            if unit_calls is not None:
                unit_calls.append((info.category, j, total))
            return FakeDs(10), {}

        def has_metadata_record(self, info):
            return False

        def warm_metadata_cache(self, info):
            return True

        def unit_cache_packed_count(self, info, i, j):
            return None

        def unit_cache_hit(self, info, i, j):
            return False

    pipe.tokenizer = FakeTok()
    pipe.data_pipeline = FakeDataPipeline()
    built = []

    def fake_build(dataset, stage_name, stage_cfg, **overrides):
        built.append(overrides)
        fake_trainer.train_dataset = dataset
        return fake_trainer

    pipe._build_trainer = fake_build
    pipe._save_checkpoint = lambda p: None
    return built


def test_staged_pretrain_flow() -> None:
    print("\n--- Staged Pretrain Orchestrator (fresh run) ---")
    from src.training.pipeline import TrainingPipeline

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = _staged_cfg(tmpdir)
        trainer = _FakeTrainer()
        stage_calls = []
        pipe = TrainingPipeline(cfg, _FakeDist())
        built = _stub_pipeline(pipe, trainer, stage_calls)

        res = pipe._run_staged_pretrain(cfg.training.pretrain)

        check("both stages built", len(stage_calls) == 2 and stage_calls[0] == ("code", 1, 2))
        check("stage 1 trains fresh (no resume)",
              trainer.train_calls[0] is None)
        check("stage 2 trains", len(trainer.train_calls) == 2)
        check("max_steps = total across stages", built[0]["max_steps"] == 7)
        check("full checkpoint saves", built[0]["save_only_model"] is False)
        check("ignore_data_skip forced", built[0]["ignore_data_skip"] is True)
        check("metrics recorded per stage", "stage_1" in res and "stage_2" in res)
        check("final model saved", trainer.saved is True)


def test_staged_pretrain_skip_and_resume() -> None:
    print("\n--- Staged Pretrain Resume (stage 1 done) ---")
    import json as _json
    from src.training.pipeline import TrainingPipeline

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = _staged_cfg(tmpdir)
        pretrain_dir = Path(tmpdir) / "pretrain"
        pretrain_dir.mkdir(parents=True)
        (pretrain_dir / "trainer_state.json").write_text(_json.dumps({"global_step": 3}))
        ckpt = pretrain_dir / "checkpoint-3"
        ckpt.mkdir()
        (ckpt / "dummy").write_text("x")

        trainer = _FakeTrainer()
        stage_calls = []
        pipe = TrainingPipeline(cfg, _FakeDist())
        built = _stub_pipeline(pipe, trainer, stage_calls)

        res = pipe._run_staged_pretrain(cfg.training.pretrain)

        check("completed stage skipped", [s[0] for s in stage_calls] == ["math"])
        check("resume from stage-1 checkpoint",
              trainer.train_calls == [str(ckpt)])
        check("stage 2 metrics only", "stage_2" in res and "stage_1" not in res)


def test_metadata_cache_fingerprint() -> None:
    print("\n--- Metadata Cache Fingerprint ---")
    from src.data.metadata_cache import compute_fingerprint

    base = dict(repo="bigcode/the-stack-v2-dedup", name="Python", split="train",
                data_dir=None, files=["a.parquet", "b.parquet"], revision="abc123",
                preprocess_sig="p1", token_sig="t1", fingerprint_version=1)
    k1 = compute_fingerprint(**base)
    for field in ("repo", "name", "split", "revision", "preprocess_sig", "token_sig"):
        d = dict(base)
        d[field] = "CHANGED"
        check(f"{field} change invalidates fingerprint", compute_fingerprint(**d) != k1)
    d = dict(base)
    d["files"] = ["b.parquet", "a.parquet"]
    check("file order does not change fingerprint", compute_fingerprint(**d) == k1)
    d = dict(base)
    d["files"] = ["a.parquet", "c.parquet"]
    check("file list change invalidates fingerprint", compute_fingerprint(**d) != k1)
    d = dict(base)
    d["fingerprint_version"] = 2
    check("schema version change invalidates fingerprint", compute_fingerprint(**d) != k1)


def test_metadata_cache_roundtrip() -> None:
    print("\n--- Metadata Cache Round Trip ---")
    from types import SimpleNamespace
    from src.data.metadata_cache import DatasetMetadataCache

    with tempfile.TemporaryDirectory() as tmpdir:
        cache = DatasetMetadataCache(tmpdir)
        info = SimpleNamespace(path="HuggingFaceFW/fineweb", name=None,
                               split="train", data_dir=None)
        rec = cache.build_record(info, ["data/train-00000.parquet"], "rev42",
                                 "pre1", "tok1", loader="parquet")
        check("record saved", cache.save(rec, info) is True)
        rec_dir = Path(tmpdir) / "HuggingFaceFW___fineweb__default"
        for fname in ("dataset_info.json", "fingerprint.json", "revision.json",
                      "split.json", "cache_location.json", "record.json"):
            check(f"{fname} written", (rec_dir / fname).exists())

        got = cache.get("HuggingFaceFW/fineweb", None, "train")
        check("get returns record", got is not None and got["revision"] == "rev42")
        check("verify ok", cache.verify(info, "pre1", "tok1") is not None)
        check("verify fails on preprocess change", cache.verify(info, "pre2", "tok1") is None)
        check("verify fails on tokenizer change", cache.verify(info, "pre1", "tok2") is None)
        check("verify fails on missing record",
              cache.verify(SimpleNamespace(path="nope/x", name=None, split="train",
                                           data_dir=None), "pre1", "tok1") is None)

        cache.invalidate("HuggingFaceFW/fineweb", None)
        check("invalidate removes record", cache.get("HuggingFaceFW/fineweb") is None)


def test_metadata_cache_fast_stream() -> None:
    print("\n--- Metadata Cache Fast Stream (real parquet) ---")
    import pyarrow as pa
    import pyarrow.parquet as pq
    from types import SimpleNamespace
    from src.data.metadata_cache import DatasetMetadataCache, stream_from_record

    with tempfile.TemporaryDirectory() as tmpdir:
        parq = Path(tmpdir) / "unit.parquet"
        pq.write_table(pa.Table.from_pydict({
            "text": ["row one text content here long enough", "row two text content here as well"],
        }), parq)
        cache = DatasetMetadataCache(tmpdir)
        info = SimpleNamespace(path="fake/repo", name=None, split="train", data_dir=None)
        rec = cache.build_record(info, [str(parq)], None, "pre1", "tok1", loader="parquet")
        cache.save(rec, info)

        rec2 = cache.verify(info, "pre1", "tok1")
        check("record verified", rec2 is not None)
        rows = list(stream_from_record(rec2))
        check("fast stream yields all rows", len(rows) == 2)
        check("row content intact", rows[0]["text"].startswith("row one"))
        rows_limited = list(stream_from_record(rec2, limit=1))
        check("limit honored", len(rows_limited) == 1)


def test_metadata_cache_source_extraction() -> None:
    print("\n--- Metadata Source Extraction & URL Rewrite ---")
    from src.data.metadata_cache import extract_data_sources, rewrite_hf_url

    class FakeEx:
        shard_data_sources = ["data/python-00000-of-00757.parquet", "data/python-00001-of-00757.parquet"]

    out = extract_data_sources(FakeEx())
    check("shard sources extracted", len(out) == 2 and out[0].endswith(".parquet"))
    check("no sources for bare object", extract_data_sources(object()) == [])

    check("https untouched", rewrite_hf_url("https://x/y.parquet") == "https://x/y.parquet")
    u = rewrite_hf_url("hf://datasets/org/repo@abc123/data/train-00000.parquet")
    check("hf url rewritten", u == "https://huggingface.co/datasets/org/repo/resolve/abc123/data/train-00000.parquet")


def test_staged_pretrain_dataset_mode() -> None:
    print("\n--- Staged Pretrain: dataset-granular mode ---")
    from src.data.registry import build_registry
    from src.training.pipeline import TrainingPipeline

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = _staged_cfg(tmpdir, mode="dataset", steps=(300, 400))
        trainer = _FakeTrainer(checkpoint_dir=Path(tmpdir) / "pretrain")
        stage_calls = []
        unit_calls = []
        pipe = TrainingPipeline(cfg, _FakeDist())
        built = _stub_pipeline(pipe, trainer, stage_calls, unit_calls)

        registry = build_registry()
        code_units = [u for u in registry.all_entries() if u.category == "code"]
        math_units = [u for u in registry.all_entries() if u.category == "math"]
        total_units = len(code_units) + len(math_units)

        res = pipe._run_staged_pretrain(cfg.training.pretrain)

        check("every registry dataset becomes a training unit",
              len(unit_calls) == total_units and len(trainer.train_calls) == total_units,
              f"units={len(unit_calls)} train_calls={len(trainer.train_calls)}")
        check("stage-level builder not used in dataset mode", len(stage_calls) == 0)
        check("units ordered by stage then registry order",
              [c[0] for c in unit_calls][: len(code_units)] == ["code"] * len(code_units)
              and [c[0] for c in unit_calls][len(code_units):] == ["math"] * len(math_units))
        check("first unit trains fresh (no checkpoint exists)",
              trainer.train_calls[0] is None)
        check("every later unit resumes from the latest checkpoint",
              all(r is not None for r in trainer.train_calls[1:]))
        check("max_steps = total across stages", built[0]["max_steps"] == 700)
        check("full checkpoint saves enabled", built[0]["save_only_model"] is False)
        check("unit metrics recorded", any(k.startswith("stage_1/unit_") for k in res))
        check("final model saved", trainer.saved is True)
        check("last unit boundary = total steps", trainer.callback.end_step == 700)


def test_staged_pretrain_dataset_mode_resume() -> None:
    print("\n--- Staged Pretrain: dataset mode resume mid-stage ---")
    import json as _json
    from src.data.registry import build_registry
    from src.training.pipeline import TrainingPipeline

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = _staged_cfg(tmpdir, mode="dataset", steps=(300, 400))
        pretrain_dir = Path(tmpdir) / "pretrain"
        pretrain_dir.mkdir(parents=True)
        math_units = [u for u in build_registry().all_entries() if u.category == "math"]
        total_w = sum(u.weight for u in math_units) or len(math_units)
        first_alloc = max(1, int(round(400 * math_units[0].weight / total_w)))
        mid_step = 300 + first_alloc
        (pretrain_dir / "trainer_state.json").write_text(_json.dumps({"global_step": mid_step}))
        ckpt = pretrain_dir / f"checkpoint-{mid_step}"
        ckpt.mkdir()
        (ckpt / "dummy").write_text("x")

        trainer = _FakeTrainer(checkpoint_dir=pretrain_dir)
        stage_calls = []
        unit_calls = []
        pipe = TrainingPipeline(cfg, _FakeDist())
        built = _stub_pipeline(pipe, trainer, stage_calls, unit_calls)

        res = pipe._run_staged_pretrain(cfg.training.pretrain)

        check("completed stage 1 skipped", len(stage_calls) == 0)
        check("completed first math unit skipped",
              [c[1] for c in unit_calls] == list(range(2, len(math_units) + 1)),
              f"unit indices built: {[c[1] for c in unit_calls]}")
        check("remaining units resume from mid-stage checkpoint",
              trainer.train_calls == [str(ckpt)] * (len(math_units) - 1))
        check("no stage_1 metrics recorded", not any(k.startswith("stage_1") for k in res))
        check("unit metrics recorded for stage 2", any(k.startswith("stage_2/unit_") for k in res))
        check("final boundary = total steps", trainer.callback.end_step == 700)


def test_build_pretrain_dataset_unit_cache() -> None:
    print("\n--- Dataset Unit Cache Round Trip ---")
    from src.data.pipeline import DataPipeline
    from src.data.registry import DatasetInfo

    class _UnitPipeline(DataPipeline):
        def __init__(self, cfg, tok):
            super().__init__(cfg, tok)
            self.builder_calls = 0

        def build_pretrain_dataset_from_registry(self, max_samples_per_dataset=None,
                                                 dataset_filter=None, include=None):
            self.builder_calls += 1
            from datasets import Dataset
            ds = Dataset.from_list([{"input_ids": [1, 2, 3], "labels": [1, 2, 3]}] * 10)

            class _Result:
                pass
            r = _Result()
            r._entries = [(ds, 0.5, "x/unit-ds", "code", 0.9)]
            r._dataset_metas = [{"avg_qs": 0.9}]
            return r

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = _staged_cfg(tmpdir)
        cfg.data.use_packed_cache = True
        info = DatasetInfo(path="x/unit-ds", category="code", weight=0.5,
                           quality_score=0.9, name="en", max_samples=50)
        pipe = _UnitPipeline(cfg, _FakeTokenizer())

        ds1, meta1 = pipe.build_pretrain_dataset_unit(info, 1, 1, 3)
        check("cold build resolves dataset", pipe.builder_calls == 1)
        check("unit meta recorded", meta1["packed_count"] == 10 and "key" in meta1)
        check("unit cache files written", (Path(cfg.training.pretrain.staging.stage_cache_dir)
                                           / "stage1" / "u001" / "packed.pt").exists())

        ds2, meta2 = pipe.build_pretrain_dataset_unit(info, 1, 1, 3)
        check("warm build skips resolution entirely", pipe.builder_calls == 1)
        check("warm cache returns dataset", len(ds2) > 0)
        check("same key reused", meta2["key"] == meta1["key"])

        cfg.training.max_seq_length = 2048
        ds3, meta3 = pipe.build_pretrain_dataset_unit(info, 1, 1, 3)
        check("preprocess change invalidates unit cache", pipe.builder_calls == 2)
        check("new key after signature change", meta3["key"] != meta1["key"])


def test_staged_pretrain_prefetch_overlap() -> None:
    print("\n--- Staged Pretrain: CPU prefetch overlaps GPU training ---")
    import threading as _t
    from src.training.pipeline import TrainingPipeline

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = _staged_cfg(tmpdir, mode="dataset", steps=(300, 400))

        ev_build2_started = _t.Event()
        ev_release_build2 = _t.Event()
        ev_train1_started = _t.Event()
        ev_release_train1 = _t.Event()
        overlap = []

        class FakeTok:
            def save_pretrained(self, path):
                pass

        class FakeDs:
            def __init__(self, n):
                self.n = n

            def __len__(self):
                return self.n

        class FakeDataPipeline:
            def build_pretrain_dataset_unit(self, info, i, j, total):
                if j == 2:
                    ev_build2_started.set()
                    ev_release_build2.wait(5)
                return FakeDs(10), {}

            def has_metadata_record(self, info):
                return False

            def warm_metadata_cache(self, info):
                return True

            def unit_cache_packed_count(self, info, i, j):
                return None

            def unit_cache_hit(self, info, i, j):
                return False

        class BlockingTrainer(_FakeTrainer):
            def train(self, resume_from_checkpoint=None):
                self.train_calls.append(resume_from_checkpoint)
                if len(self.train_calls) == 1:
                    ev_train1_started.set()
                    ev_release_train1.wait(5)
                    if ev_build2_started.is_set():
                        overlap.append(True)
                from types import SimpleNamespace
                return SimpleNamespace(metrics={"loss": 0.1})

            def add_callback(self, cb):
                self.callback = cb

            def remove_callback(self, cb):
                pass

        trainer = BlockingTrainer()
        pipe = TrainingPipeline(cfg, _FakeDist())
        pipe.tokenizer = FakeTok()
        pipe.data_pipeline = FakeDataPipeline()

        def fake_build(dataset, stage_name, stage_cfg, **overrides):
            trainer.train_dataset = dataset
            return trainer

        pipe._build_trainer = fake_build
        pipe._save_checkpoint = lambda p: None

        result_holder = {}

        def run():
            result_holder["res"] = pipe._run_staged_pretrain(cfg.training.pretrain)

        t = _t.Thread(target=run)
        t.start()
        check("training of unit 1 started", ev_train1_started.wait(5))
        check("build of unit 2 started before unit 1 finished",
              ev_build2_started.wait(5))
        ev_release_train1.set()
        ev_release_build2.set()
        t.join(30)
        check("run completed", not t.is_alive())
        check("CPU built next dataset while GPU was busy", bool(overlap))
        check("all units trained", len(trainer.train_calls)
              == len([u for u in __import__("src.data.registry", fromlist=["build_registry"]).build_registry().all_entries()
                      if u.category in ("code", "math")]))


def test_resolve_and_cache_warm() -> None:
    print("\n--- Metadata Warm Resolution (real parquet) ---")
    import pyarrow as pa
    import pyarrow.parquet as pq
    from types import SimpleNamespace
    import src.data.streaming as streaming_mod
    from src.data.metadata_cache import DatasetMetadataCache
    from src.data.streaming import resolve_and_cache

    with tempfile.TemporaryDirectory() as tmpdir:
        parq = Path(tmpdir) / "warm.parquet"
        pq.write_table(pa.Table.from_pydict({
            "text": ["warm row one long enough content", "warm row two long enough content"],
        }), parq)
        cache = DatasetMetadataCache(Path(tmpdir) / "meta")
        info = SimpleNamespace(path=str(parq), name=None, split="train", data_dir=None)

        orig = streaming_mod.load_dataset_builder
        calls = []

        def counting(*args, **kwargs):
            calls.append(1)
            return orig(*args, **kwargs)

        streaming_mod.load_dataset_builder = counting
        try:
            ok = resolve_and_cache(path=str(parq), split="train", meta_cache=cache,
                                   preprocess_sig="pre1", token_sig="tok1")
            check("first resolve returns True", ok)
            check("record persisted", cache.verify(info, "pre1", "tok1") is not None)
            check("local files resolved without builder", len(calls) == 0)

            calls.clear()
            ok2 = resolve_and_cache(path=str(parq), split="train", meta_cache=cache,
                                    preprocess_sig="pre1", token_sig="tok1")
            check("second resolve short-circuits on cached record", ok2 and len(calls) == 0)

            cache.invalidate(str(parq), None)
            ok3 = resolve_and_cache(path=str(parq), split="train", meta_cache=cache,
                                    preprocess_sig="pre1", token_sig="tok1")
            check("after invalidate re-resolves from local files", ok3)
            check("record restored after re-resolve",
                  cache.verify(info, "pre1", "tok1") is not None)
        finally:
            streaming_mod.load_dataset_builder = orig


def test_warm_stage_metadata_parallel() -> None:
    print("\n--- Parallel Stage Metadata Warming ---")
    import time as _time
    from src.data.registry import build_registry
    from src.training.pipeline import TrainingPipeline

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = _staged_cfg(tmpdir)
        pipe = TrainingPipeline(cfg, _FakeDist())

        class WarmPipe:
            def __init__(self, n_missing):
                self.n_missing = n_missing
                self.warm_calls = 0

            def has_metadata_record(self, info):
                return not self.n_missing

            def warm_metadata_cache(self, info):
                self.warm_calls += 1
                _time.sleep(0.3)
                return True

        wp = WarmPipe(6)
        pipe.data_pipeline = wp
        units = build_registry().all_entries()[:6]

        t0 = _time.perf_counter()
        pipe._warm_stage_metadata(units, 1)
        elapsed = _time.perf_counter() - t0
        check("all missing datasets warmed", wp.warm_calls == 6)
        check("resolution ran in parallel (serial would be ~1.8s)",
              elapsed < 1.2, f"elapsed={elapsed:.2f}s")

        wp2 = WarmPipe(0)
        pipe.data_pipeline = wp2
        pipe._warm_stage_metadata(units, 2)
        check("fully cached stage skips resolution", wp2.warm_calls == 0)


def test_dynamic_stage_sizing() -> None:
    print("\n--- Dynamic Stage Sizing (packed-count allocation) ---")
    import src.training.pipeline as tp
    from src.data.registry import DatasetRegistry, DatasetInfo
    from src.training.pipeline import TrainingPipeline

    class FakeDs:
        def __init__(self, n):
            self.n = n

        def __len__(self):
            return self.n

    class CountingTrainer(_FakeTrainer):
        def __init__(self, checkpoint_dir=None):
            super().__init__(checkpoint_dir)
            self.boundaries = []

        def add_callback(self, cb):
            self.callback = cb
            self.boundaries.append(cb)

    def make_pipe(tmpdir, counts, weights=(0.5, 0.5)):
        cfg = _staged_cfg(tmpdir, mode="dataset", steps=(400, 0))
        trainer = CountingTrainer()
        reg = DatasetRegistry()
        reg.register(DatasetInfo(path="a/one", category="code", weight=weights[0],
                                 quality_score=0.9, max_samples=10))
        reg.register(DatasetInfo(path="a/two", category="code", weight=weights[1],
                                 quality_score=0.9, max_samples=10))
        orig = tp.build_registry
        tp.build_registry = lambda: reg

        class FakeDataPipeline:
            def build_pretrain_dataset_unit(self, info, i, j, total):
                return FakeDs(10), {}

            def has_metadata_record(self, info):
                return True

            def warm_metadata_cache(self, info):
                return True

            def unit_cache_packed_count(self, info, i, j):
                return counts.get(j) if counts else None

            def unit_cache_hit(self, info, i, j):
                return False

        pipe = TrainingPipeline(cfg, _FakeDist())
        pipe.tokenizer = type("T", (), {"save_pretrained": lambda s, p: None})()
        pipe.data_pipeline = FakeDataPipeline()
        pipe._build_trainer = lambda dataset, sn, sc, **ov: (
            setattr(trainer, "train_dataset", dataset) or trainer)
        pipe._save_checkpoint = lambda p: None
        return pipe, trainer, orig

    with tempfile.TemporaryDirectory() as tmpdir:
        pipe, trainer, orig = make_pipe(tmpdir, {1: 100, 2: 300})
        try:
            res = pipe._run_staged_pretrain(pipe.cfg.training.pretrain)
        finally:
            tp.build_registry = orig
        check("packed basis: both units trained", len(trainer.train_calls) == 2)
        check("packed basis: 100/300 split of 400 steps",
              [c.end_step for c in trainer.boundaries] == [100, 400],
              f"boundaries={[c.end_step for c in trainer.boundaries]}")

    with tempfile.TemporaryDirectory() as tmpdir:
        pipe2, trainer2, orig2 = make_pipe(tmpdir, None)
        try:
            pipe2._run_staged_pretrain(pipe2.cfg.training.pretrain)
        finally:
            tp.build_registry = orig2
        check("weight basis: equal weights split 400 steps evenly",
              [c.end_step for c in trainer2.boundaries] == [200, 400],
              f"boundaries={[c.end_step for c in trainer2.boundaries]}")


def test_tokenizer_manifest_cache() -> None:
    print("\n--- Tokenizer Manifest Cache ---")
    from src.models.factory import ModelFactory

    class _Tok:
        pad_token = None
        padding_side = None
        model_max_length = None

        def add_special_tokens(self, tokens):
            self.pad_token = tokens["pad_token"]

    with tempfile.TemporaryDirectory() as tmpdir:
        tok_dir = Path(tmpdir) / "tokenizer"
        tok_dir.mkdir()
        (tok_dir / "vocab.json").write_text('{"a": 0}')
        (tok_dir / "merges.txt").write_text("a b")

        calls = []
        orig = ModelFactory._try_load_tokenizer
        ModelFactory._try_load_tokenizer = staticmethod(
            lambda path, offline=False: (calls.append(offline), _Tok())[1])
        try:
            t1 = ModelFactory.load_tokenizer(tok_dir)
            check("first load writes manifest", (tok_dir / "tokenizer_version.json").exists())
            check("first load goes through network path", calls == [False])
            check("pad token added", t1.pad_token == "<pad>")
            check("model_max_length defaults", t1.model_max_length == 4096)

            calls.clear()
            ModelFactory.load_tokenizer(tok_dir)
            check("second load verified offline (no network)",
                  calls == [True])

            (tok_dir / "vocab.json").write_text('{"a": 0, "b": 1}')
            calls.clear()
            ModelFactory.load_tokenizer(tok_dir)
            check("file change invalidates manifest", calls == [False])

            (tok_dir / "tokenizer_version.json").unlink()
            calls.clear()
            ModelFactory.load_tokenizer(tok_dir)
            check("missing manifest falls back to network path", calls == [False])
            check("manifest rewritten on fallback",
                  (tok_dir / "tokenizer_version.json").exists())
        finally:
            ModelFactory._try_load_tokenizer = orig


def test_stream_dataset_fast_path_skips_resolution() -> None:
    print("\n--- stream_dataset fast path skips load_dataset ---")
    from types import SimpleNamespace
    import pyarrow as pa
    import pyarrow.parquet as pq
    import src.data.streaming as streaming_mod
    from src.data.metadata_cache import DatasetMetadataCache

    with tempfile.TemporaryDirectory() as tmpdir:
        parq = Path(tmpdir) / "ds.parquet"
        pq.write_table(pa.Table.from_pydict({
            "text": ["hello fast path content row one", "hello fast path content row two"],
        }), parq)
        cache = DatasetMetadataCache(tmpdir)
        info = SimpleNamespace(path="org/ds", name=None, split="train", data_dir=None)
        rec = cache.build_record(info, [str(parq)], None, "pre1", "tok1", loader="parquet")
        cache.save(rec, info)

        orig = streaming_mod.load_dataset
        def _bomb(*a, **k):
            raise AssertionError("load_dataset must not be called on fast path")
        streaming_mod.load_dataset = _bomb
        try:
            rows = list(streaming_mod.stream_dataset(
                "org/ds", split="train", meta_cache=cache,
                preprocess_sig="pre1", token_sig="tok1"))
        finally:
            streaming_mod.load_dataset = orig
        check("fast path streams rows", len(rows) == 2)
        check("text extraction applied", all("text" in r for r in rows))


def test_stream_dataset_captures_metadata() -> None:
    print("\n--- stream_dataset captures metadata on cold path ---")
    from types import SimpleNamespace
    import src.data.streaming as streaming_mod
    from src.data.metadata_cache import DatasetMetadataCache

    class FakeEx:
        shard_data_sources = ["data/x-00000-of-00002.parquet", "data/x-00001-of-00002.parquet"]

    class FakeDS:
        def __init__(self):
            self._ex_iterable = FakeEx()
        def __iter__(self):
            return iter([{"text": "sample document body text enough"}] * 1)

    with tempfile.TemporaryDirectory() as tmpdir:
        cache = DatasetMetadataCache(tmpdir)
        orig = streaming_mod.load_dataset
        streaming_mod.load_dataset = lambda **kw: FakeDS()
        try:
            rows = list(streaming_mod.stream_dataset(
                "bigcode/the-stack-v2-dedup", name="Python", split="train",
                meta_cache=cache, preprocess_sig="pre1", token_sig="tok1"))
        finally:
            streaming_mod.load_dataset = orig
        check("cold path streams rows", len(rows) == 1)
        rec = cache.get("bigcode/the-stack-v2-dedup", "Python", "train")
        check("record captured with shards", rec is not None and rec["num_shards"] == 2)
        check("loader inferred parquet", rec["loader"] == "parquet")


def _write_multi_shard_parquet(tmpdir: Path, shards: int = 3, rows_per: int = 5) -> List[Path]:
    """Multi-shard local parquet; every 3rd row has text=None (gated out).
    Texts are long enough that one row fills one packed example
    (> max_seq_length=8192 chars, tokenizer cap 20000) and each text carries
    a unique block in its middle so every 8192-token random window contains
    row-distinct tokens (the repeated lorem body alone would collide)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    lorem = ("lorem ipsum dolor sit amet consectetur adipiscing elit sed do "
             "eiusmod tempor incididunt ut labore et dolore magna aliqua. ")
    files = []
    for s in range(shards):
        table_rows = []
        for r in range(rows_per):
            if r % 3 == 2:
                table_rows.append({"text": None})
            else:
                body = lorem * 35 + f" UNIQUE {s}/{r} " + lorem * 34
                table_rows.append({"text": f"shard{s} row{r} " + body})
        path = tmpdir / f"part-{s:05d}.parquet"
        pq.write_table(pa.Table.from_pydict(
            {k: [row[k] for row in table_rows] for k in ("text",)}), path)
        files.append(path)
    return files


def _full_gated_reference(files: List[Path]) -> List[str]:
    """Sequential reference: plain datasets streaming over the same files,
    same gate as the streaming layer (detect once, extract, drop no-text)."""
    from datasets import load_dataset
    from src.data.metadata_cache import rewrite_hf_url
    from src.data.registry import detect_text_fields, extract_text

    ds = load_dataset(
        "parquet",
        data_files=[rewrite_hf_url(str(f)) for f in files],
        split="train", streaming=True,
    )
    detected = None
    out = []
    for s in ds:
        if detected is None:
            detected = detect_text_fields(s, None)
        text = extract_text(s, detected)
        if text:
            out.append(text)
    return out


def test_shard_coordinator_identity_and_resume() -> None:
    print("\n--- ShardCoordinator: parallel order identity + exact resume ---")
    from types import SimpleNamespace
    from src.data.metadata_cache import DatasetMetadataCache
    from src.data.streaming import ShardCoordinator, resolve_and_cache

    with tempfile.TemporaryDirectory() as tmpdir:
        files = _write_multi_shard_parquet(Path(tmpdir), shards=3, rows_per=5)
        cache = DatasetMetadataCache(Path(tmpdir) / "meta")
        info = SimpleNamespace(path=str(Path(tmpdir)), name=None, split="train",
                               data_dir=None)
        ok = resolve_and_cache(path=str(Path(tmpdir)), split="train",
                               meta_cache=cache, preprocess_sig="pre1",
                               token_sig="tok1")
        check("local record resolved", ok)
        rec = cache.verify(info, "pre1", "tok1")
        check("record has 3 shards", rec is not None and rec["num_shards"] == 3)

        ref = _full_gated_reference(files)
        check("reference has 12 gated rows (15 raw - 3 no-text)",
              len(ref) == 12, f"got {len(ref)}")

        coord = ShardCoordinator(rec, [(i, 0) for i in range(3)], workers=3)
        out = []
        try:
            for s in coord:
                out.append(s["text"])
        finally:
            coord.close()
        check("parallel stream == sequential order", out == ref)
        check("raw_rows == 15", coord.raw_rows() == 15, f"got {coord.raw_rows()}")
        check("gated == 12", coord.gated_count() == 12, f"got {coord.gated_count()}")
        check("no failed shards", coord.failed_shards() == [])

        # Exact mid-shard resume: run 1 stops after 7 gated samples.
        coord_a = ShardCoordinator(rec, [(i, 0) for i in range(3)], workers=3,
                                   limit=7)
        part1 = []
        try:
            for s in coord_a:
                part1.append(s["text"])
        finally:
            coord_a.close()
        state = coord_a.progress_state()
        # 4 gated per shard -> 7 gated stops mid-shard-1 (raw rows 1,2,4 seen).
        check("stopped mid-shard (raw offset > 0)",
              state[0] == 1 and state[1] > 0, f"state={state}")
        resumed_plan = [(state[0], state[1])]
        resumed_plan += [(i, 0) for i in range(state[0] + 1, 3)]
        coord_b = ShardCoordinator(rec, resumed_plan, workers=3)
        part2 = []
        try:
            for s in coord_b:
                part2.append(s["text"])
        finally:
            coord_b.close()
        check("resume continues without gaps",
              part1 + part2 == ref, f"{len(part1)}+{len(part2)} vs {len(ref)}")
        check("no duplicates across resume",
              len(set(part1 + part2)) == len(part1 + part2))

        # Gated limit mirrors stream_dataset semantics (limit-th sample included).
        coord_c = ShardCoordinator(rec, [(i, 0) for i in range(3)], workers=2,
                                   limit=7)
        limited = []
        try:
            for s in coord_c:
                limited.append(s["text"])
        finally:
            coord_c.close()
        check("limit counts gated samples", limited == ref[:7])


def test_shard_progress_store_roundtrip() -> None:
    print("\n--- ShardProgressStore: persistence + yield ordering ---")
    from src.data.shards import ShardProgressStore, build_shard_plan, shard_progress_key

    with tempfile.TemporaryDirectory() as tmpdir:
        store = ShardProgressStore(tmpdir)
        fp = "abc123"
        rec = store.load("org/ds", None, "train", fp)
        check("fresh record defaults", rec["last_shard"] == -1 and rec["done"] is False)

        store.update_shard_stats(rec, 0, {"streamed": 10, "accepted": 1, "raw": 15})
        store.update_shard_stats(rec, 2, {"streamed": 10, "accepted": 8, "raw": 20})
        rec["resume"] = {"shard": 1, "offset": 5}
        rec["last_shard"] = 0
        store.save(rec)

        rec2 = store.load("org/ds", None, "train", fp)
        check("stats persisted", rec2["stats"]["0"]["accepted"] == 1)
        check("stats persisted shard 2", rec2["stats"]["2"]["accepted"] == 8)
        check("resume position persisted", rec2["resume"] == {"shard": 1, "offset": 5})

        plan = build_shard_plan(3, rec2, "sequential")
        check("resume plan starts at exact position",
              plan == [(1, 5), (2, 0)], f"got {plan}")

        # completed shards are skipped (complete flags + legacy last_shard)
        rec2["last_shard"] = 2
        store.save(rec2)
        rec3 = store.load("org/ds", None, "train", fp)
        plan = build_shard_plan(3, rec3, "sequential")
        check("completed shards skipped", plan == [], f"got {plan}")

        # yield order: shard 2 (80%) before shard 0 (10%); untouched shard 1 last
        rec4 = store.load("org/ds", None, "train", fp)
        rec4["last_shard"] = -1
        plan = build_shard_plan(3, rec4, "yield")
        order = [i for i, _ in plan]
        check("yield order by acceptance", order == [2, 0, 1], f"got {order}")

        store.delete("org/ds", None, "train", fp)
        rec5 = store.load("org/ds", None, "train", fp)
        check("delete removes record", rec5["last_shard"] == -1 and not rec5["stats"])
        check("key is stable", shard_progress_key("org/ds", None, "train", fp)
              == shard_progress_key("org/ds", None, "train", fp))


def test_policy_matching_and_diagnostics() -> None:
    print("\n--- Dataset policies + bottleneck/low-acceptance diagnostics ---")
    import io
    from types import SimpleNamespace
    from src.config.schema import Config, DatasetPolicyConfig
    from src.data.pipeline import DataPipeline
    from src.data.registry import DatasetInfo

    cfg = Config()
    cfg.data.dataset_policies = [
        DatasetPolicyConfig(path="bigcode/*", accepted_target=500),
        DatasetPolicyConfig(path="openai/*", accepted_target=200, text_fields=["text"]),
    ]
    pipe = DataPipeline(cfg, _FakeTokenizer())

    info = DatasetInfo("bigcode/the-stack-v2-dedup", "code", 1.0, 0.8)
    p = pipe._dataset_policy(info)
    check("glob match bigcode/*", p.accepted_target == 500)

    info2 = DatasetInfo("openai/gsm8k", "math", 1.0, 0.8)
    p2 = pipe._dataset_policy(info2)
    check("glob match openai/*", p2.accepted_target == 200 and p2.text_fields == ["text"])

    info3 = DatasetInfo("allenai/c4", "web_text", 1.0, 0.8)
    p3 = pipe._dataset_policy(info3)
    check("no policy -> defaults", p3.accepted_target == 0
          and p3.shard_order == "sequential")

    class FakeStreamer:
        stats = SimpleNamespace(
            timings={"metadata_resolve": 0.5, "shard_select": 0.2,
                     "arrow_open_sec": 0.1, "arrow_open_max_sec": 0.05,
                     "network_wait_sec": 31.0, "extraction_sec": 0.3,
                     "stream_sec": 32.5},
            raw_rows=100,
            shard_stats={0: {"raw": 60}, 1: {"raw": 40}},
        )

    cfg.data.bottleneck_threshold_sec = 30.0
    cfg.data.acceptance_investigation_threshold = 0.10
    ledger = {"accepted": 1, "rejected_boilerplate": 0, "rejected_short": 5,
              "rejected_quality": 80, "rejected_dedup": 3, "rejected_ast": 0,
              "rejected_empty": 11}
    plogger = logging.getLogger("src.data.pipeline")
    buffer = io.StringIO()
    handler = logging.StreamHandler(buffer)
    handler.setLevel(logging.INFO)
    plogger.addHandler(handler)
    old_level = plogger.level
    plogger.setLevel(logging.INFO)
    try:
        pipe._report_dataset_diagnostics("fake/ds", FakeStreamer(),
                                         {"quality_scoring": 1.0}, ledger, 2.0)
        out = buffer.getvalue()
    finally:
        plogger.removeHandler(handler)
        plogger.setLevel(old_level)
    check("BOTTLENECK DETECTED logged",
          "BOTTLENECK DETECTED" in out, out[:200])
    check("LOW ACCEPTANCE investigation logged",
          "LOW ACCEPTANCE" in out and "rejected by low quality" in out,
          out[:400])
    check("bottleneck stage named", "network_wait" in out)


class _FakeTok:
    eos_token_id = 0
    name_or_path = "fake-tokenizer"
    vocab_size = 32000

    def __call__(self, text, truncation=False, add_special_tokens=False):
        return {"input_ids": [ord(c) % 977 + 1 for c in text[:20000]]}


def _pipe_with_local_registry(tmpdir: str, target: int, rows_per: int = 5):
    """DataPipeline wired to a local 3-shard parquet registry entry with
    deterministic acceptance (quality scorer stubbed to 0.9 -> all gated rows
    accepted; no AST/function stages for web_text category)."""
    from types import SimpleNamespace
    from src.config.schema import Config, DatasetPolicyConfig
    from src.data.pipeline import DataPipeline
    from src.data.registry import DatasetInfo, DatasetRegistry
    from src.data.streaming import resolve_and_cache

    root = Path(tmpdir)
    data_dir = root / "data"
    data_dir.mkdir(exist_ok=True)
    _write_multi_shard_parquet(data_dir, shards=3, rows_per=rows_per)

    cfg = Config()
    cfg.data.metadata_cache.enabled = True
    cfg.data.metadata_cache.dir = str(root / "meta")
    cfg.data.shard_progress_dir = str(root / "shards")
    cfg.data.resume_shards = True
    cfg.data.use_packed_cache = False
    cfg.data.sanity_checks.enabled = False
    cfg.data.health_reporting.enabled = False
    # Tiny lorem texts are near-duplicates for simhash banding; exact dedup
    # keeps the acceptance path deterministic (no two rows are identical).
    cfg.data.quality.deduplication.method = "exact"
    # Deterministic random_window_sample windows across the three runs.
    import random
    random.seed(42)
    cfg.data.dataset_policies = [
        DatasetPolicyConfig(path="*", accepted_target=target)
    ]

    pipe = DataPipeline(cfg, _FakeTok())
    pipe._get_cleanup_pool = lambda: None

    import src.data.pipeline as pipeline_mod
    pipeline_mod._pool_quality_score = lambda text, category, language: 0.9

    registry = DatasetRegistry()
    info = DatasetInfo(
        path=str(data_dir), category="web_text", weight=1.0, quality_score=0.8,
        text_fields=["text"], max_samples=None,
    )
    registry.register(info)
    pipeline_mod.build_registry = lambda: registry

    ppsig = pipe.processing_signature()
    toksig = pipe.tokenizer_signature
    ok = resolve_and_cache(path=str(data_dir), split="train",
                           meta_cache=pipe.meta_cache, preprocess_sig=ppsig,
                           token_sig=toksig)
    assert ok, "local record resolution failed"
    rec = pipe.meta_cache.verify(
        SimpleNamespace(path=str(data_dir), name=None, split="train",
                        data_dir=None), ppsig, toksig)
    assert rec is not None and rec["num_shards"] == 3
    return pipe, info


def test_registry_build_accepted_target_and_resume() -> None:
    print("\n--- Registry build: accepted-target stop + shard resume ---")
    # 3 shards x 1200 raw rows (800 gated each = 2400 total). A gated limit of
    # 1024 streams exactly one chunk; target=3 stops right after it, mid-shard-1.
    ROWS = 1200
    CHUNK = 1024
    GATED = 3 * (ROWS * 2 // 3)  # 2400

    with tempfile.TemporaryDirectory() as tmpdir:
        pipe, _ = _pipe_with_local_registry(tmpdir, target=3, rows_per=ROWS)
        result = pipe.build_pretrain_dataset_from_registry(
            max_samples_per_dataset=CHUNK)
        entries = result._entries
        check("one dataset built", len(entries) == 1)
        packed = entries[0][0]
        check("stopped after first chunk (1024 >= target 3)",
              len(packed) == CHUNK, f"got {len(packed)}")
        check("packed ids tokenized",
              all(len(p["input_ids"]) > 0 for p in packed))
        run1_ids = {tuple(p["input_ids"]) for p in packed}

        prog_dir = Path(pipe.cfg.data.shard_progress_dir)
        prog_files = list(prog_dir.glob("*.json"))
        check("early stop persisted shard progress",
              len(prog_files) == 1, f"files: {prog_files}")
        if prog_files:
            import json
            prec = json.loads(prog_files[0].read_text(encoding="utf-8"))
            resume = prec.get("resume", {})
            check("resume point is mid-shard",
                  resume.get("offset", 0) > 0 and resume.get("shard", 0) < 2,
                  f"resume={resume}")
            check("per-shard stats recorded",
                  any(st.get("accepted", 0) > 0
                      for st in prec.get("stats", {}).values()))

        # Second run resumes from the exact position and stops at the same
        # chunk boundary — disjoint, gap-free continuation.
        pipe2, _ = _pipe_with_local_registry(tmpdir, target=3, rows_per=ROWS)
        result2 = pipe2.build_pretrain_dataset_from_registry(
            max_samples_per_dataset=CHUNK)
        packed2 = result2._entries[0][0]
        check("resumed build produced the next chunk",
              len(packed2) == CHUNK, f"got {len(packed2)}")
        run2_ids = {tuple(p["input_ids"]) for p in packed2}
        check("no duplicate samples across runs",
              not (run1_ids & run2_ids))
        check("progress still persisted (not complete yet)",
              len(list(Path(pipe2.cfg.data.shard_progress_dir).glob("*.json"))) == 1)

        # Third run: no target -> drains the remaining 352 gated rows;
        # the progress record is removed once every shard is complete.
        pipe3, _ = _pipe_with_local_registry(tmpdir, target=0, rows_per=ROWS)
        result3 = pipe3.build_pretrain_dataset_from_registry(
            max_samples_per_dataset=GATED)
        packed3 = result3._entries[0][0]
        check("no target -> remaining rows drained",
              len(packed3) == GATED - 2 * CHUNK, f"got {len(packed3)}")
        run3_ids = {tuple(p["input_ids"]) for p in packed3}
        check("progress cleared after full completion",
              len(list(Path(pipe3.cfg.data.shard_progress_dir).glob("*.json"))) == 0)
        full_ids = run1_ids | run2_ids | run3_ids
        check("runs are pairwise disjoint and cover the full stream",
              len(full_ids) == GATED
              and not (run1_ids & run2_ids) and not (run1_ids & run3_ids)
              and not (run2_ids & run3_ids))


def test_registry_build_shard_order_yield() -> None:
    print("\n--- Registry build: yield ordering reuses measured shards first ---")
    import hashlib
    from src.config.schema import DatasetPolicyConfig
    from src.data.shards import ShardProgressStore

    with tempfile.TemporaryDirectory() as tmpdir:
        pipe, info = _pipe_with_local_registry(tmpdir, target=3)
        ppsig = pipe.processing_signature()
        toksig = pipe.tokenizer_signature
        fp = hashlib.sha256(
            f"{info.path}||train|{ppsig}|{toksig}".encode()).hexdigest()
        store = ShardProgressStore(pipe.cfg.data.shard_progress_dir)
        rec = store.load(info.path, None, "train", fp)
        # shard 2 yielded 100%, shard 0 yielded 33% — yield order starts with 2
        store.update_shard_stats(rec, 2, {"streamed": 4, "accepted": 4})
        store.update_shard_stats(rec, 0, {"streamed": 6, "accepted": 2})
        rec["last_shard"] = -1
        store.save(rec)

        pipe.cfg.data.dataset_policies = [
            DatasetPolicyConfig(path="*", accepted_target=3,
                                shard_order="yield")
        ]
        policy = pipe._dataset_policy(info)
        check("yield policy matched", policy.shard_order == "yield")

        import src.data.pipeline as pipeline_mod
        streamer = pipe._stream_sharded(info, pipeline_mod.build_registry(),
                                        limit=1000, policy=policy)
        seen_shards = []
        texts = []
        try:
            for s in streamer:
                seen_shards.append(s["_shard"])
                texts.append(s["text"])
                if len(texts) >= 2:
                    break
        finally:
            streamer.stats.shard_streamed = {}
            streamer.stats.shard_accepted = {}
            streamer.close()
        check("yield order starts with measured-best shard",
              seen_shards[:2] == [2, 2], f"got {seen_shards[:4]}")


def test_rewrite_hf_url_legacy_ids() -> None:
    print("\n--- URL rewrite: legacy single-component repo ids ---")
    from src.data.metadata_cache import rewrite_hf_url

    u = rewrite_hf_url("hf://datasets/code_search_net@abc123/train-00000.json.gz")
    check("single-component repo + revision",
          u == "https://huggingface.co/datasets/code_search_net/resolve/abc123/train-00000.json.gz",
          u)
    u = rewrite_hf_url("hf://datasets/code_search_net/train-00000.json.gz")
    check("single-component repo, no revision",
          u == "https://huggingface.co/datasets/code_search_net/resolve/train-00000.json.gz",
          u)
    u = rewrite_hf_url("hf://datasets/org/repo@rev/data/x.parquet")
    check("two-component repo + revision preserved",
          u == "https://huggingface.co/datasets/org/repo/resolve/rev/data/x.parquet", u)
    u = rewrite_hf_url("hf://datasets/org/repo/data/x.parquet")
    check("two-component repo preserved",
          u == "https://huggingface.co/datasets/org/repo/resolve/data/x.parquet", u)
    u = rewrite_hf_url("hf://datasets/org/repo/resolve/rev/x.parquet")
    check("resolve-form preserved",
          u == "https://huggingface.co/datasets/org/repo/resolve/rev/x.parquet", u)
    check("https untouched",
          rewrite_hf_url("https://x/y.parquet") == "https://x/y.parquet")


def test_builder_cache_roundtrip() -> None:
    print("\n--- Builder cache: record + resume state ---")
    from types import SimpleNamespace
    from src.data.drivers import BuilderCache

    with tempfile.TemporaryDirectory() as tmpdir:
        bc = BuilderCache(Path(tmpdir) / "builders")
        info = SimpleNamespace(path="bigcode/the-stack-v2-dedup", name="Python",
                               split="train", data_dir=None)
        rec = bc.build_record(info, revision="main", script_revision="s1",
                              builder_class="FakeScriptBuilder",
                              preprocess_sig="pre1", token_sig="tok1")
        check("record schema", rec["schema_version"] == 2
              and rec["driver"] == "script")
        check("record saved", bc.save(rec, info))
        check("verify roundtrip", bc.verify(info, "pre1", "tok1") is not None)
        check("fingerprint changes with preprocess sig",
              bc.verify(info, "pre2", "tok1") is None)
        check("fingerprint changes with tokenizer sig",
              bc.verify(info, "pre1", "tok2") is None)
        bc.save_resume(rec, 1234)
        rec2 = bc.verify(info, "pre1", "tok1")
        check("resume persisted", rec2 is not None
              and rec2["resume"]["offset"] == 1234)
        bc.reset_resume(rec2)
        rec3 = bc.verify(info, "pre1", "tok1")
        check("resume reset", rec3 is not None and rec3["resume"]["offset"] == 0)
        bc.invalidate("bigcode/the-stack-v2-dedup", "Python")
        check("invalidated", bc.verify(info, "pre1", "tok1") is None)


def test_detect_driver_families() -> None:
    print("\n--- Driver detection: local / file / script / streaming ---")
    import src.data.drivers as drivers_mod
    from types import SimpleNamespace
    from src.data.drivers import (
        BuilderCache,
        DRIVER_KIND_FILE,
        DRIVER_KIND_LOCAL,
        DRIVER_KIND_SCRIPT,
        LocalDatasetDriver,
        detect_driver,
    )
    from src.data.metadata_cache import DatasetMetadataCache

    class FakeScriptBuilder:
        def __init__(self, sources):
            self._sources = sources
            self.revision = "script-sha"

        def as_streaming_dataset(self, split):
            ex = (SimpleNamespace(shard_data_sources=self._sources)
                  if self._sources else None)
            return SimpleNamespace(_ex_iterable=ex)

    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir) / "data"
        data_dir.mkdir()
        _write_multi_shard_parquet(data_dir, shards=2, rows_per=4)
        mc = DatasetMetadataCache(Path(tmpdir) / "meta")
        bc = BuilderCache(Path(tmpdir) / "builders")

        local_info = SimpleNamespace(path=str(data_dir), name=None,
                                     split="train", data_dir=None)
        drv, diag = detect_driver(local_info, mc, bc, "pre1", "tok1")
        check("local dir -> local driver",
              diag["driver_kind"] == DRIVER_KIND_LOCAL
              and isinstance(drv, LocalDatasetDriver))
        check("local resolution skipped", diag["repo_resolution_skipped"])

        orig = drivers_mod.load_dataset_builder
        try:
            hub_info = SimpleNamespace(path="fake/repo", name=None,
                                       split="train", data_dir=None)
            drivers_mod.load_dataset_builder = (
                lambda *a, **k: FakeScriptBuilder(
                    ["https://x/part-00000.parquet"]))
            drv, diag = detect_driver(hub_info, mc, bc, "pre1", "tok1")
            check("file sources -> file driver",
                  diag["driver_kind"] == DRIVER_KIND_FILE
                  and drv.record is not None)
            check("file record cached",
                  mc.verify(hub_info, "pre1", "tok1") is not None)
            check("first resolution flagged", diag["first_resolution"])

            script_info = SimpleNamespace(path="bigcode/the-stack-v2-dedup",
                                          name="Python", split="train",
                                          data_dir=None)
            drivers_mod.load_dataset_builder = (
                lambda *a, **k: FakeScriptBuilder(None))
            drv, diag = detect_driver(script_info, mc, bc, "pre1", "tok1")
            check("no sources -> script driver",
                  diag["driver_kind"] == DRIVER_KIND_SCRIPT)
            check("builder record cached",
                  bc.verify(script_info, "pre1", "tok1") is not None)
            check("no file-list record for script",
                  mc.get(script_info.path, script_info.name, "train") is None)

            calls = []
            drivers_mod.load_dataset_builder = (
                lambda *a, **k: (calls.append(1), FakeScriptBuilder(None))[1])
            drv2, diag2 = detect_driver(script_info, mc, bc, "pre1", "tok1")
            check("script warm start skips resolution",
                  diag2["builder_hit"] and len(calls) == 0,
                  f"calls={len(calls)}")
        finally:
            drivers_mod.load_dataset_builder = orig


def test_auto_parquet_builder_detection() -> None:
    print("\n--- Auto-parquet builder (config.data_files) -> file family ---")
    import src.data.drivers as drivers_mod
    from types import SimpleNamespace
    from src.data.drivers import (
        BuilderCache,
        DRIVER_KIND_FILE,
        detect_driver,
        inspect_builder,
    )
    from src.data.metadata_cache import (
        DatasetMetadataCache,
        extract_data_sources,
    )

    class ArrowLike:
        def __init__(self, files):
            self._files = files

        def shard_data_sources(self):
            return self._files

    class FakeParquetBuilder:
        def __init__(self, data_files):
            self.config = SimpleNamespace(data_files=data_files)
            self.revision = "parquet-sha"

    files = extract_data_sources(ArrowLike(["https://x/a.parquet"]))
    check("callable shard_data_sources invoked",
          files == ["https://x/a.parquet"], f"got {files}")

    df = {"train": ["hf://datasets/org/repo@abc123/train-00000.parquet",
                    "hf://datasets/org/repo@abc123/train-00001.parquet"],
          "test": ["hf://datasets/org/repo@abc123/test-00000.parquet"]}
    inf_files, loader = inspect_builder(FakeParquetBuilder(df), "train")
    check("config.data_files split-filtered",
          len(inf_files) == 2
          and all("train-" in f for f in inf_files)
          and all("test-" not in f for f in inf_files),
          f"got {inf_files}")
    check("loader detected for parquet list", loader == "parquet",
          f"got {loader}")

    with tempfile.TemporaryDirectory() as tmpdir:
        mc = DatasetMetadataCache(Path(tmpdir) / "meta")
        bc = BuilderCache(Path(tmpdir) / "builders")
        hub_info = SimpleNamespace(path="bigcode/the-stack-v2-dedup",
                                   name="Python", split="train", data_dir=None)
        orig = drivers_mod.load_dataset_builder
        try:
            drivers_mod.load_dataset_builder = (
                lambda *a, **k: FakeParquetBuilder(df))
            drv, diag = detect_driver(hub_info, mc, bc, "pre1", "tok1")
            check("auto-parquet -> file driver",
                  diag["driver_kind"] == DRIVER_KIND_FILE
                  and diag["first_resolution"])
            check("file record cached",
                  mc.verify(hub_info, "pre1", "tok1") is not None)
            check("no builder record for file family",
                  bc.verify(hub_info, "pre1", "tok1") is None)

            rec = mc.get(hub_info.path, hub_info.name, "train")
            check("record holds 2 train files",
                  rec is not None and len(rec.get("files", [])) == 2,
                  f"got {rec and len(rec.get('files', []))}")
            check("record excludes test split",
                  rec is not None and not any("test-" in f for f in rec["files"]))

            calls = []
            drivers_mod.load_dataset_builder = (
                lambda *a, **k: (calls.append(1), FakeParquetBuilder(df))[1])
            drv2, diag2 = detect_driver(hub_info, mc, bc, "pre1", "tok1")
            check("warm start metadata hit, no resolution",
                  diag2["metadata_hit"] and len(calls) == 0,
                  f"calls={len(calls)}")
        finally:
            drivers_mod.load_dataset_builder = orig


def test_script_first_row_timeout() -> None:
    print("\n--- Script driver: bounded first row ---")
    import src.data.drivers as drivers_mod
    from types import SimpleNamespace
    from src.data.drivers import BuilderCache, ScriptDatasetDriver, _bounded_next
    import time

    fast = iter([{"text": "row1"}])
    check("bounded_next fast path returns row",
          _bounded_next(fast, 5.0, "fake") == {"text": "row1"})
    check("bounded_next disabled timeout",
          _bounded_next(iter([{"text": "row2"}]), 0, "fake") == {"text": "row2"})

    def stalled():
        time.sleep(30)

    t0 = time.perf_counter()
    try:
        _bounded_next(iter(stalled, None), 1.0, "fake/repo")
        check("bounded_next times out", False, "no TimeoutError raised")
    except TimeoutError as e:
        check("bounded_next times out", time.perf_counter() - t0 < 5.0)
        check("timeout message has guidance",
              "HF_HUB_DISABLE_XET" in str(e))

    rows = [{"text": f"row{i}"} for i in range(4)]

    class FakeScriptBuilder:
        revision = "s1"

        def as_streaming_dataset(self, split):
            return _script_iterable(rows)

    with tempfile.TemporaryDirectory() as tmpdir:
        bc = BuilderCache(Path(tmpdir) / "builders")
        info = SimpleNamespace(path="fake/script", name=None,
                               split="train", data_dir=None)
        rec = bc.build_record(info, revision=None, script_revision="s1",
                              builder_class="FakeScriptBuilder",
                              preprocess_sig="pre1", token_sig="tok1")
        bc.save(rec, info)
        orig = drivers_mod.load_dataset_builder
        orig_timeout = drivers_mod._FIRST_ROW_TIMEOUT
        drivers_mod._FIRST_ROW_TIMEOUT = 5.0
        drivers_mod.load_dataset_builder = lambda *a, **k: FakeScriptBuilder()
        try:
            drv = ScriptDatasetDriver(rec, builder_cache=bc)
            got = list(drv.stream(limit=4, text_fields=["text"]))
            check("no first row lost with guard",
                  [s["text"] for s in got] == ["row0", "row1", "row2", "row3"]
                  and [s["_raw_seq"] for s in got] == [1, 2, 3, 4])
        finally:
            drivers_mod._FIRST_ROW_TIMEOUT = orig_timeout
            drivers_mod.load_dataset_builder = orig


def _script_iterable(rows: list):
    """datasets.IterableDataset over a python list (datasets 4.x removed
    from_list; from_generator is the supported path)."""
    from datasets import IterableDataset
    return IterableDataset.from_generator(lambda: iter(rows))


def test_script_driver_resume() -> None:
    print("\n--- Script driver: iterator resume ---")
    import src.data.drivers as drivers_mod
    from types import SimpleNamespace
    from src.data.drivers import BuilderCache, ScriptDatasetDriver

    rows = [{"text": f"row {i} " + "x" * 60, "idx": i} for i in range(10)]
    iterable = _script_iterable(rows)

    class FakeScriptBuilder:
        revision = "s1"

        def as_streaming_dataset(self, split):
            return iterable

    with tempfile.TemporaryDirectory() as tmpdir:
        bc = BuilderCache(Path(tmpdir) / "builders")
        info = SimpleNamespace(path="fake/script", name=None,
                               split="train", data_dir=None)
        rec = bc.build_record(info, revision=None, script_revision="s1",
                              builder_class="FakeScriptBuilder",
                              preprocess_sig="pre1", token_sig="tok1")
        check("record saved", bc.save(rec, info))

        orig = drivers_mod.load_dataset_builder
        drivers_mod.load_dataset_builder = lambda *a, **k: FakeScriptBuilder()
        try:
            drv = ScriptDatasetDriver(rec, builder_cache=bc)
            part1 = list(drv.stream(limit=3, text_fields=["text"]))
            check("streams 3 gated",
                  len(part1) == 3 and part1[0]["text"] == rows[0]["text"])
            check("raw_seq starts at 1",
                  [s["_raw_seq"] for s in part1] == [1, 2, 3])
            check("no natural end at limit", not drv.natural_end())
            drv.save_resume()
            check("resume offset = 3 raw rows", rec["resume"]["offset"] == 3)

            rec2 = bc.verify(info, "pre1", "tok1")
            drv2 = ScriptDatasetDriver(rec2, builder_cache=bc)
            check("resumed flag", drv2.resumed)
            part2 = list(drv2.stream(limit=3, text_fields=["text"]))
            check("resumed stream continues",
                  [s["text"] for s in part2] == [rows[3]["text"],
                                                 rows[4]["text"],
                                                 rows[5]["text"]])
            check("raw_seq continues after resume",
                  [s["_raw_seq"] for s in part2] == [4, 5, 6])
            drv2.save_resume()
            check("resume advanced", rec2["resume"]["offset"] == 6)

            # Natural exhaustion resets the iterator state.
            drv3 = ScriptDatasetDriver(rec2, builder_cache=bc)
            rest = list(drv3.stream(text_fields=["text"]))
            check("drained the rest", len(rest) == 4)
            check("natural end after drain", drv3.natural_end())
            drv3.reset_resume()
            rec3 = bc.verify(info, "pre1", "tok1")
            check("resume reset after natural end",
                  rec3 is not None and rec3["resume"]["offset"] == 0)
        finally:
            drivers_mod.load_dataset_builder = orig


def test_detect_driver_warm_no_resolution() -> None:
    print("\n--- Detect driver: warm record skips all HF resolution ---")
    from types import SimpleNamespace
    from src.data.drivers import (
        BuilderCache,
        DRIVER_KIND_FILE,
        detect_driver,
    )
    from src.data.metadata_cache import DatasetMetadataCache

    with tempfile.TemporaryDirectory() as tmpdir:
        mc = DatasetMetadataCache(Path(tmpdir) / "meta")
        bc = BuilderCache(Path(tmpdir) / "builders")
        hub_info = SimpleNamespace(path="fake/repo", name=None,
                                   split="train", data_dir=None)
        rec = mc.build_record(
            hub_info,
            ["https://x/part-00000.parquet", "https://x/part-00001.parquet"],
            None, "pre1", "tok1", "parquet")
        check("record saved", mc.save(rec, hub_info))
        check("record verifies warm", mc.verify(hub_info, "pre1", "tok1") is not None)

        def _must_not_load_builder(*a, **k):
            raise AssertionError("load_dataset_builder called on a warm cache hit")

        drv, diag = detect_driver(
            hub_info, mc, bc, "pre1", "tok1", load_builder=_must_not_load_builder)
        check("file driver on warm hit",
              diag["driver_kind"] == DRIVER_KIND_FILE and drv.record is not None)
        check("metadata_hit flagged", diag["metadata_hit"])
        check("repo resolution skipped", diag["repo_resolution_skipped"])
        check("not first resolution", not diag["first_resolution"])
        check("two shards in record", len(drv.record["files"]) == 2)


def test_coordinator_first_row_timeout() -> None:
    print("\n--- ShardCoordinator: bounded first row (no silent hang) ---")
    import os
    import time as _time
    from types import SimpleNamespace
    from src.data.streaming import ShardCoordinator

    rec = {
        "driver": "file",
        "repo": "fake/repo",
        "loader": "parquet",
        "files": ["https://localhost:65535/nope/part-00000.parquet"],
    }
    env_key = "DATA_FILE_FIRST_ROW_TIMEOUT"
    old = os.environ.get(env_key)
    os.environ[env_key] = "0.6"
    try:
        coord = ShardCoordinator(rec, [(0, 0)], workers=1)

        def _blocked_shard(run):
            _time.sleep(30)
            return
            yield  # pragma: no cover

        coord._stream_shard = _blocked_shard
        t0 = _time.perf_counter()
        raised = False
        try:
            next(coord)
        except TimeoutError as e:
            raised = True
            msg = str(e)
            check("timeout message explains gated datasets", "gated" in msg.lower())
            check("timeout message names the env override",
                  "DATA_FILE_FIRST_ROW_TIMEOUT" in msg)
        finally:
            coord.close()
        check("coordinator times out, no hang", raised and
              (_time.perf_counter() - t0) < 12)
    finally:
        if old is None:
            os.environ.pop(env_key, None)
        else:
            os.environ[env_key] = old


def test_script_family_pipeline_stream() -> None:
    print("\n--- Pipeline: script family streams via builder, warms cache ---")
    import src.data.drivers as drivers_mod
    from src.config.schema import Config, DatasetPolicyConfig
    from src.data.pipeline import DataPipeline
    from src.data.registry import DatasetInfo, DatasetRegistry

    rows = [{"text": f"row {i} " + "y" * 60, "idx": i} for i in range(12)]
    iterable = _script_iterable(rows)

    class FakeScriptBuilder:
        revision = "s1"

        def as_streaming_dataset(self, split):
            return iterable

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = Config()
        cfg.data.metadata_cache.enabled = True
        cfg.data.metadata_cache.dir = str(Path(tmpdir) / "meta")
        cfg.data.shard_progress_dir = str(Path(tmpdir) / "shards")
        cfg.data.resume_shards = True
        cfg.data.use_packed_cache = False
        cfg.data.sanity_checks.enabled = False
        cfg.data.health_reporting.enabled = False
        cfg.data.quality.deduplication.method = "exact"
        import random
        random.seed(42)
        cfg.data.dataset_policies = [
            DatasetPolicyConfig(path="*", accepted_target=0)
        ]

        pipe = DataPipeline(cfg, _FakeTok())
        pipe._get_cleanup_pool = lambda: None
        import src.data.pipeline as pipeline_mod
        pipeline_mod._pool_quality_score = lambda text, category, language: 0.9

        registry = DatasetRegistry()
        info = DatasetInfo(path="fake/script", category="web_text", weight=1.0,
                           quality_score=0.8, text_fields=["text"],
                           max_samples=None)
        registry.register(info)
        pipeline_mod.build_registry = lambda: registry

        orig = drivers_mod.load_dataset_builder
        drivers_mod.load_dataset_builder = lambda *a, **k: FakeScriptBuilder()
        policy = pipe._dataset_policy(info)
        try:
            streamer = pipe._stream_sharded(info, registry, limit=5, policy=policy)
            out = list(streamer)
            streamer.close()
            check("script family streamed 5 gated", len(out) == 5)
            check("driver kind = script",
                  streamer.stats.driver_kind == "script")
            check("first resolution flagged", streamer.stats.first_resolution)
            check("script samples tagged",
                  all(s["_shard"] == 0 for s in out))
            check("builder record persisted",
                  (Path(tmpdir) / "meta" / "builders" / "fake___script__default"
                   / "builder_record.json").exists())
            check("raw_rows tracked", streamer.stats.raw_rows == 5)

            # Warm second run: detection uses the builder cache (zero builder
            # resolution); streaming re-instantiates the builder from the
            # local script module cache (the one unavoidable load_dataset_builder
            # call) and resumes the iterator at the persisted raw offset.
            calls = []
            drivers_mod.load_dataset_builder = (
                lambda *a, **k: (calls.append(1), FakeScriptBuilder())[1])
            pipe2 = DataPipeline(cfg, _FakeTok())
            pipe2._get_cleanup_pool = lambda: None
            ppsig2 = pipe2.processing_signature()
            toksig2 = pipe2.tokenizer_signature
            drv2, diag2 = pipe2._driver_for(info, ppsig2, toksig2)
            check("warm detection skips builder resolution",
                  len(calls) == 0 and diag2["builder_hit"]
                  and diag2["repo_resolution_skipped"],
                  f"calls={len(calls)}")
            streamer2 = pipe2._stream_sharded(info, registry, limit=5,
                                              policy=policy)
            out2 = list(streamer2)
            streamer2.close()
            check("stream re-instantiates builder once (module cache)",
                  len(calls) == 1, f"calls={len(calls)}")
            check("warm run reuses builder",
                  streamer2.stats.builder_hit
                  and not streamer2.stats.first_resolution)
            check("warm run resumes iterator",
                  streamer2.stats.script_resumed)
            check("warm run continues from offset",
                  [s["_raw_seq"] for s in out2] == [6, 7, 8, 9, 10],
                  str([s["_raw_seq"] for s in out2]))
        finally:
            drivers_mod.load_dataset_builder = orig


def main() -> None:

    tests = [
        test_missing_directory,
        test_missing_jsonl_file,
        test_empty_jsonl,
        test_malformed_jsonl_high_ratio,
        test_malformed_jsonl_medium_ratio,
        test_malformed_jsonl_low_ratio,
        test_invalid_utf8,
        test_warning_schema,
        test_valid_jsonl,
        test_duplicate_accounting,
        test_category_weight_accounting,
        test_warning_local_weight,
        test_doc_builder_zero_pages_report,
        test_report_skipped_dup_in_ok_column,
        test_stage_boundary_callback_behavior,
        test_stage_cache_key_sensitivity,
        test_stage_dataset_cache_roundtrip,
        test_stage_dataset_passes_filter,
        test_staged_pretrain_flow,
        test_staged_pretrain_skip_and_resume,
        test_metadata_cache_fingerprint,
        test_metadata_cache_roundtrip,
        test_metadata_cache_fast_stream,
        test_metadata_cache_source_extraction,
        test_stream_dataset_fast_path_skips_resolution,
        test_stream_dataset_captures_metadata,
        test_staged_pretrain_dataset_mode,
        test_staged_pretrain_dataset_mode_resume,
        test_build_pretrain_dataset_unit_cache,
        test_staged_pretrain_prefetch_overlap,
        test_tokenizer_manifest_cache,
        test_resolve_and_cache_warm,
        test_warm_stage_metadata_parallel,
        test_dynamic_stage_sizing,
        test_shard_coordinator_identity_and_resume,
        test_shard_progress_store_roundtrip,
        test_policy_matching_and_diagnostics,
        test_registry_build_accepted_target_and_resume,
        test_registry_build_shard_order_yield,
        test_rewrite_hf_url_legacy_ids,
        test_builder_cache_roundtrip,
        test_detect_driver_families,
        test_auto_parquet_builder_detection,
        test_script_driver_resume,
        test_script_first_row_timeout,
        test_detect_driver_warm_no_resolution,
        test_coordinator_first_row_timeout,
        test_script_family_pipeline_stream,
    ]
    for t in tests:
        try:
            t()
        except Exception as e:
            global FAIL
            FAIL += 1
            print(f"  EXCEPTION {t.__name__}: {e}")

    total = PASS + FAIL
    print(f"\n{'=' * 60}")
    print(f"  RESULTS: {PASS}/{total} passed, {FAIL} failed")
    print(f"{'=' * 60}")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
