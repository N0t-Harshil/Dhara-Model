"""CLI parsing regression tests — the --gpu flag used argparse.SUPPRESS,
which made `args.gpu` absent from the namespace and crashed
reserved-training/full-training with AttributeError when --gpu was omitted."""

from __future__ import annotations


def test_reserved_training_parses_without_gpu_flag():
    from main import build_parser

    parser = build_parser()
    args = parser.parse_args(["reserved-training", "--fresh-start"])
    assert args.command == "reserved-training"
    assert args.gpu is None


def test_reserved_training_parses_with_gpu_flag():
    from main import build_parser

    parser = build_parser()
    args = parser.parse_args(["reserved-training", "--gpu", "3"])
    assert args.gpu == "3"


def test_full_training_parses_without_gpu_flag():
    from main import build_parser

    parser = build_parser()
    args = parser.parse_args(["full-training"])
    assert args.command == "full-training"
    assert args.gpu is None


def test_generate_parses_without_gpu_flag():
    from main import build_parser

    parser = build_parser()
    args = parser.parse_args(["generate", "--prompt", "hello"])
    assert args.gpu is None


def test_resolve_gpu_explicit_flag_overrides_preexisting_env(monkeypatch):
    """Container images pre-set CUDA_VISIBLE_DEVICES to every GPU, which used to
    make an explicit --gpu silently no-op (full-training ran 8-way DataParallel
    on all A100s while the step accounting assumed world=1)."""
    import sys

    from main import _resolve_gpu

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
    monkeypatch.setattr(sys, "argv", ["main.py", "full-training", "--gpu", "3"])
    assert _resolve_gpu() == "3"


def test_resolve_gpu_keeps_preexisting_env_without_flag(monkeypatch):
    import sys

    from main import _resolve_gpu

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    monkeypatch.setattr(sys, "argv", ["main.py", "full-training"])
    assert _resolve_gpu() is None


def test_resolve_gpu_ignores_flag_under_distributed(monkeypatch):
    """Under torchrun/DDP every rank must keep its own device; never mask when
    LOCAL_RANK is set, even if --gpu was passed."""
    import sys

    from main import _resolve_gpu

    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(sys, "argv", ["main.py", "full-training", "--gpu", "3"])
    assert _resolve_gpu() is None
