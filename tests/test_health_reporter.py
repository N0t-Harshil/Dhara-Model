from __future__ import annotations

"""DatasetHealthReport regression tests (mandate Phases 4-7).

Root cause of the live foundation-training crash
(``TypeError: unsupported format string passed to NoneType.__format__`` at
``health_reporter.summary_text`` line 223):

* ``build_pretrain_dataset_from_registry``'s packed-cache branch builds
  ``lang_dist={cached_lang: meta["accepted"]}`` where
  ``cached_lang = meta.get("lang_d") or info.language`` is ``None`` for
  datasets without a single detected language (the-stack style, OpenCoder).
* ``add_dataset_stats`` stored the ``None`` label as a dict key, then
  ``summary_text`` applied the ``:<15`` format spec to it — ``None`` only
  accepts an empty format spec, hence the TypeError.

These tests pin: None labels -> "unknown"; None counts -> "-" with "N/A";
zero totals -> "N/A"; empty/missing optional stats -> no crash; normal
distributions render correct percentages; and the exact cached-load call
shape the async worker runs (Phase 6) never raises.
"""

import json

import pytest

from src.data.health_reporter import DatasetHealthReport

OPENCODER = dict(
    path="OpenCoder-LLM/opc-fineweb-code-corpus/default", category="code",
    weight=0.045, raw_count=100000, packed_count=5739, total_tokens=11753472,
    duplicate_removed=28023,
)

STACK_CPP = dict(
    path="bigcode/the-stack-v2-dedup/C++", category="code",
    weight=0.013, raw_count=100000, packed_count=197, total_tokens=403456,
    duplicate_removed=7630,
)


def _add(r, *, accepted, lang_dist=None, domain_dist=None, quality=None, **kw):
    """Mimic add_dataset_stats callers (both fresh and packed-cache paths)."""
    r.add_dataset_stats(
        after_boilerplate=accepted, after_quality=accepted, after_dedup=accepted,
        rejection_reasons={"simhash_dedup": kw.get("dup"), "no_parse": 2},
        quality_scores=quality if quality is not None else [0.9] * accepted,
        lang_dist=lang_dist, domain_dist=domain_dist,
        token_lengths=[512] * 1000, max_seq_length=2048,
        **kw,
    )


##############################
# 1. normal distribution
##############################

def test_normal_language_distribution_percentages():
    r = DatasetHealthReport("normal")
    _add(r, accepted=100,
         lang_dist={"python": 80, "javascript": 20},
         domain_dist={"app": 100}, **OPENCODER)
    text = r.summary_text()
    assert "python" in text and "80.0%" in text
    assert "javascript" in text and "20.0%" in text
    assert "app" in text and "100.0%" in text


##############################
# 2. empty distributions
##############################

def test_empty_distribution_no_crash():
    r = DatasetHealthReport("empty")
    r.add_dataset_stats(
        path="p", category="code", weight=0.5, raw_count=10,
        after_boilerplate=5, after_quality=5, after_dedup=5,
        packed_count=5, total_tokens=100, duplicate_removed=0,
    )
    r.compute_global_stats()
    assert r.languages == {}
    assert r.domains == {}
    text = r.summary_text()  # must not raise
    assert "Language Distribution" not in text
    assert "Domain Distribution" not in text


##############################
# 3. zero total
##############################

def test_zero_total_renders_na():
    r = DatasetHealthReport("zero")
    r.languages = {"code": 0}
    r.domains = {}
    r.datasets = []
    r.categories = {}
    r.global_stats = {}
    text = r.summary_text()  # must not raise
    assert "N/A" in text


##############################
# 4. count explicitly None -> N/A, not 0.0
##############################

def test_none_count_renders_na():
    r = DatasetHealthReport("nonecount")
    r.languages = {"python": 80, "javascript": None}
    r.domains = {}
    r.datasets = []
    r.categories = {}
    r.global_stats = {}
    text = r.summary_text()  # must not raise
    assert "python" in text and "100.0%" in text
    assert "javascript" in text and "N/A" in text
    assert "-" in text  # count rendered as unavailable


##############################
# 5. missing optional statistics
##############################

def test_missing_optional_stats_no_crash():
    r = DatasetHealthReport("optional")
    r.add_dataset_stats(
        path="p", category="code", weight=0.5, raw_count=10,
        after_boilerplate=5, after_quality=5, after_dedup=5,
        packed_count=5, total_tokens=100, duplicate_removed=0,
    )  # no rejection_reasons / quality_scores / lang_dist / domain_dist / token_lengths
    r.compute_global_stats()
    text = r.summary_text()  # must not raise
    assert r.languages == {}
    assert "DATASET HEALTH REPORT" in text


##############################
# 6. THE box scenario: cached path with None lang label
##############################

def test_cached_load_none_language_label():
    """configure the exact packed-cache branch:
    cached_lang = meta.get("lang_d") or info.language  -> None
    domain     = {None: accepted} if cd else {}
    """
    r = DatasetHealthReport("cached")
    cached_lang = None
    _add(r, accepted=44548,
         lang_dist={cached_lang: 44548},
         domain_dist={"algorithms": 44548}, **OPENCODER)
    r.compute_global_stats()
    assert None not in r.languages
    assert "unknown" in r.languages and r.languages["unknown"] == 44548
    text = r.summary_text()  # must not raise
    assert "unknown" in text and "100.0%" in text


def test_cached_load_none_domain_label():
    r = DatasetHealthReport("cacheddomain")
    _add(r, accepted=10389,
         lang_dist={"cpp": 10389},
         domain_dist={None: 10389}, **STACK_CPP)
    r.compute_global_stats()
    assert None not in r.domains
    assert "unknown" in r.domains
    r.summary_text()  # must not raise


##############################
# 7.-8. real dataset styles
##############################

def test_opencoder_style_report():
    r = DatasetHealthReport("opencoder")
    _add(r, accepted=44548,
         lang_dist={"other": 44510, "javascript": 38},
         domain_dist={"algorithms": 44548},
         quality=[0.92] * 44548, **OPENCODER)
    r.compute_global_stats()
    text = r.summary_text()
    assert "opc-fineweb" in text
    assert "other" in text and "javascript" in text
    assert "44548" in text and "5739" in text


def test_stack_cpp_style_report():
    r = DatasetHealthReport("stackcpp")
    _add(r, accepted=10389, lang_dist={"cpp": 10389},
         domain_dist={"backend": 10389}, quality=[0.88] * 10389, **STACK_CPP)
    r.compute_global_stats()
    text = r.summary_text()
    assert "cpp" in text and "10389" in text
    assert "backend" in text


##############################
# 9. async-worker call order (pipeline.py:2342)
##############################

def test_async_worker_call_order(tmp_path):
    """The async prefetch worker runs build_pretrain_dataset_from_registry,
    which ends with health_report.compute_global_stats() + save() +
    summary_text(). Replay that exact sequence on the crashed object."""
    r = DatasetHealthReport("worker")
    _add(r, accepted=44548,
         lang_dist={None: 44548}, domain_dist={"algorithms": 44548}, **OPENCODER)
    r.compute_global_stats()
    out = r.save(str(tmp_path))
    assert out.exists()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert "unknown" in payload["languages"]
    text = r.summary_text()  # the exact call that crashed on the box
    assert "DATASET HEALTH REPORT" in text


##############################
# 10. two datasets aggregated on one report (stage window)
##############################

def test_two_datasets_aggregated_concurrently():
    r = DatasetHealthReport("stage")
    _add(r, accepted=44548,
         lang_dist={None: 44548}, domain_dist={"algorithms": 44548}, **OPENCODER)
    _add(r, accepted=10389,
         lang_dist={"cpp": 10389}, domain_dist={"backend": 10389}, **STACK_CPP)
    r.compute_global_stats()
    assert r.languages["unknown"] == 44548 and r.languages["cpp"] == 10389
    text = r.summary_text()  # must not raise
    assert "unknown" in text and "cpp" in text
    assert "81.1%" in text and "18.9%" in text


def test_two_report_instances_do_not_share_state():
    a = DatasetHealthReport("a")
    b = DatasetHealthReport("b")
    _add(a, accepted=44548, lang_dist={None: 44548}, **OPENCODER)
    _add(b, accepted=10389, lang_dist={"cpp": 10389}, **STACK_CPP)
    assert set(a.languages) == {"unknown"}
    assert set(b.languages) == {"cpp"}


##############################
# Phase 8: distinct vs labeled-sample counts
##############################

def test_global_stats_distinct_and_labeled_counts():
    """Global stats must report BOTH the labeled-sample count (legacy keys)
    and the distinct-label count; the report text uses unambiguous names."""
    r = DatasetHealthReport("distinct")
    _add(r, accepted=530, lang_dist={"cpp": 500, "javascript": 30},
         domain_dist={"backend": 530}, **STACK_CPP)
    _add(r, accepted=100, lang_dist={"other": 100},
         domain_dist={"general": 100}, **OPENCODER)
    r.compute_global_stats()
    g = r.global_stats
    assert g["languages_detected"] == 630          # legacy = labeled-sample count
    assert g["language_labeled_samples"] == 630
    assert g["domains_detected"] == 630
    assert g["domain_labeled_samples"] == 630
    assert g["distinct_languages"] == 3
    assert g["distinct_domains"] == 2
    text = r.summary_text()
    assert "Distinct Languages" in text and "Distinct Domains" in text
    assert "Language Labeled Samples" in text


##############################
# Phase 6: cached meta preserves per-text language distribution
##############################

def test_health_stats_from_meta_preserves_per_text_lang_dist():
    """The packed-cache replay must carry the per-text lang_dist into the
    health report, not collapse it to a single static-label key (the box-run
    'unknown 44548 (100.0%)' regression)."""
    from src.data.pipeline import DataPipeline
    from src.data.registry import DatasetInfo

    pipe = DataPipeline.__new__(DataPipeline)
    info = DatasetInfo(path="bigcode/the-stack-v2-dedup", name="C++",
                       category="code", weight=0.013, quality_score=0.88)
    meta = {
        "loaded": 100000, "accepted": 10389, "packed_count": 197,
        "total_tokens": 403456, "rejected_dedup": 7630,
        "quality_scores": [0.88] * 197, "token_lengths": [512] * 100,
        "lang_dist": {"other": 10380, "javascript": 9},
        "domain_dist": {"backend": 10389},
        "ledger": {"loaded": 100000, "accepted": 10389},
        "lang_d": None, "domain_d": "backend",
    }
    kwargs = pipe._health_stats_from_meta(meta, info, 2048)
    assert kwargs["lang_dist"] == {"other": 10380, "javascript": 9}
    assert kwargs["domain_dist"] == {"backend": 10389}
    assert kwargs["path"] == info.path
    assert kwargs["packed_count"] == 197

    # No distribution AND no static label -> honest empty dist (no fabricated
    # 'unknown' bucket and no collapse to a single label).
    meta_nodist = dict(meta)
    meta_nodist["lang_dist"] = {}
    kwargs = pipe._health_stats_from_meta(meta_nodist, info, 2048)
    assert kwargs["lang_dist"] == {}

    # Legacy collapse ONLY when a static label exists and no dist is stored.
    meta_static = dict(meta)
    meta_static["lang_dist"] = {}
    info_lang = DatasetInfo(path="codeparrot/codeparrot-clean",
                            category="code", weight=0.01, quality_score=0.84,
                            language="python")
    kwargs = pipe._health_stats_from_meta(meta_static, info_lang, 2048)
    assert kwargs["lang_dist"] == {"python": 10389}


##############################
# Audit: per-dataset dist contract + fresh timestamp
##############################

def test_per_dataset_lang_domain_distribution_sums_without_inflation():
    """add_dataset_stats must accumulate the SUM of each dataset's OWN
    distribution — never a cumulative run-global dict passed per dataset
    (which would double-count every label once per dataset, inflating
    languages_detected/domains_detected on multi-dataset builds)."""
    r = DatasetHealthReport("nocumulative")
    _add(r, accepted=500, lang_dist={"cpp": 300, "javascript": 200},
         domain_dist={"backend": 500}, **STACK_CPP)
    # Second dataset gets ONLY its own labels — the pipeline must pass the
    # per-dataset dict here, not the accumulated run-wide one.
    _add(r, accepted=250, lang_dist={"python": 250},
         domain_dist={"web": 250}, **OPENCODER)
    r.compute_global_stats()
    assert r.languages == {"cpp": 300, "javascript": 200, "python": 250}
    assert r.domains == {"backend": 500, "web": 250}
    assert sum(r.languages.values()) == 750
    assert sum(r.domains.values()) == 750
    assert r.global_stats["languages_detected"] == 750
    assert r.global_stats["domains_detected"] == 750


def test_compute_global_stats_refreshes_timestamp():
    """'Generated:' must reflect the aggregation time, not the pipeline
    construction time (a warm-cache run computes + re-saves the report hours
    after DataPipeline __init__)."""
    import time as _t

    r = DatasetHealthReport("fresh-ts")
    ts0 = r.timestamp
    _add(r, accepted=100, lang_dist={"cpp": 100}, domain_dist={"backend": 100},
         **STACK_CPP)
    _t.sleep(0.01)  # ensure the refreshed isoformat differs
    r.compute_global_stats()
    assert r.timestamp != ts0
    assert r.timestamp > ts0