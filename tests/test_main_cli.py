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
