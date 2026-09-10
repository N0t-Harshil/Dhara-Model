"""Phase 10 (mandate §10): tokenizer/model special-token alignment.

Packing pads sequences with ``eos_token_id`` (src/data/pipeline.py
``pack_sequences``), so the active tokenizer's ``pad_token_id == eos_token_id``
aliasing (deepseek-style) must be surfaced explicitly, not silently relied on.
``ModelFactory.special_tokens_report`` is the single audit surface.
"""
from types import SimpleNamespace

from src.models.factory import ModelFactory


def _fake_tokenizer(pad=None, eos=1):
    return SimpleNamespace(
        bos_token_id=0, eos_token_id=eos, unk_token_id=2,
        pad_token_id=pad, mask_token_id=None,
    )


def test_report_resolves_table_and_alias_flag():
    tok = _fake_tokenizer(pad=1, eos=1)  # deepseek-style PAD == EOS
    report = ModelFactory.special_tokens_report(tok)
    assert report["eos_token_id"] == 1
    assert report["pad_token_id"] == 1
    assert report["pad_aliases_eos"] is True


def test_distinct_pad_not_flagged():
    tok = _fake_tokenizer(pad=3, eos=1)
    report = ModelFactory.special_tokens_report(tok)
    assert report["pad_token_id"] == 3
    assert report["pad_aliases_eos"] is False


def test_missing_pad_is_not_aliased():
    tok = _fake_tokenizer(pad=None, eos=1)
    report = ModelFactory.special_tokens_report(tok)
    assert report["pad_token_id"] is None
    assert report["pad_aliases_eos"] is False


def test_load_tokenizer_warns_on_pad_eos_alias(caplog, tmp_path, monkeypatch):
    import logging
    from src.models import factory as factory_mod

    tok = _fake_tokenizer(pad=1, eos=1)
    tok.pad_token = "</s>"
    tok.padding_side = "right"
    tok.model_max_length = 8192
    (tmp_path / "tokenizer.json").write_text("{}", encoding="utf-8")
    (tmp_path / "vocab.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(factory_mod.ModelFactory, "_verify_tokenizer_cache",
                        lambda p: {"version": 1})
    monkeypatch.setattr(factory_mod.ModelFactory, "_try_load_tokenizer",
                        lambda p, offline=True: tok)
    with caplog.at_level(logging.WARNING, logger="src.models.factory"):
        out = ModelFactory.load_tokenizer(tmp_path)
    assert out is tok
    assert any("PAD token id == EOS token id (1)" in r.message
               for r in caplog.records)


def test_load_tokenizer_no_warning_when_distinct(caplog, tmp_path, monkeypatch):
    import logging
    from src.models import factory as factory_mod

    tok = _fake_tokenizer(pad=3, eos=1)
    tok.pad_token = "<pad>"
    tok.padding_side = "right"
    tok.model_max_length = 8192
    (tmp_path / "tokenizer.json").write_text("{}", encoding="utf-8")
    (tmp_path / "vocab.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(factory_mod.ModelFactory, "_verify_tokenizer_cache",
                        lambda p: {"version": 1})
    monkeypatch.setattr(factory_mod.ModelFactory, "_try_load_tokenizer",
                        lambda p, offline=True: tok)
    with caplog.at_level(logging.WARNING, logger="src.models.factory"):
        ModelFactory.load_tokenizer(tmp_path)
    assert not any("PAD token id == EOS token id" in r.message
                   for r in caplog.records)