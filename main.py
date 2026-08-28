from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from src.utils.shutdown import ShutdownCoordinator

SHUTDOWN = ShutdownCoordinator()

def _resolve_checkpoint(cfg) -> str:
    model_dir = Path(cfg.output.model_dir)
    if (model_dir / "config.json").exists() or (model_dir / "pytorch_model.bin").exists() or (model_dir / "model.safetensors").exists():
        return str(model_dir)
    candidates = sorted(model_dir.rglob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]), reverse=True)
    if candidates:
        return str(candidates[0])
    return str(model_dir)

# ── GPU selection: must run before import torch ──────────────────────

os.environ["TOKENIZERS_PARALLELISM"] = "false"

_top_cmd = sys.argv[1] if len(sys.argv) > 1 else ""
_skip_gpu_init = _top_cmd == "reserved-training"


def _resolve_gpu() -> str | None:
    """Parse --gpu from CLI, or auto-detect best GPU, return device index string."""
    existing = os.environ.get("CUDA_VISIBLE_DEVICES")
    if existing:
        return None
    if os.environ.get("LOCAL_RANK"):
        return None
    early = argparse.ArgumentParser(add_help=False)
    early.add_argument("--gpu", type=str, default=None)
    known, _ = early.parse_known_args()
    if known.gpu is not None:
        return str(known.gpu)
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            encoding="utf-8",
        )
        free_mems = [int(x) for x in out.strip().split("\n")]
        best_idx = free_mems.index(max(free_mems))
        return str(best_idx)
    except Exception as e:
        print(f"[!] GPU auto-detection failed: {e}")
        return None


if not _skip_gpu_init:
    _gpu_selected = _resolve_gpu()
    if _gpu_selected is not None:
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if not cvd:
            os.environ["CUDA_VISIBLE_DEVICES"] = _gpu_selected
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", f"--id={_gpu_selected}", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                    encoding="utf-8",
                )
                free_mb = out.strip()
                print(f"[*] Single-GPU: physical GPU {_gpu_selected} ({free_mb}MB free)")
            except Exception:
                print(f"[*] Single-GPU: CUDA_VISIBLE_DEVICES={_gpu_selected}")

import yaml
import torch

if torch.cuda.is_available() and not _skip_gpu_init:
    try:
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            free, total = torch.cuda.mem_get_info(i)
            print(f"[*] Visible GPU {i}: {props.name} ({free/1e9:.1f}/{total/1e9:.1f} GB free)")
    except RuntimeError:
        pass

os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ["HF_HOME"] = os.path.join(_PROJECT_DIR, "hf_cache")
os.environ["HF_DATASETS_CACHE"] = os.path.join(_PROJECT_DIR, "hf_cache", "datasets")
os.environ["HF_HUB_CACHE"] = os.path.join(_PROJECT_DIR, "hf_cache", "hub")

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("datasets").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

try:
    from src.utils.logging import setup_logging  # idempotent, respects OutputConfig.log_dir
    # Defer full setup until config is loaded (log_dir comes from OutputConfig);
    # this early call just ensures console formatting.
    setup_logging(level=logging.INFO)
except Exception:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s - %(message)s",
        datefmt="%H:%M:%S",
    )
logger = logging.getLogger("main")


# ===================================================================
# COMMANDS
# ===================================================================

def _apply_hf_token_env(cfg) -> None:
    """Set HF_TOKEN from config before any hub/datasets import so downloads
    are authenticated (tokenizer download runs before DataPipeline exists)."""
    token = getattr(getattr(cfg, "data", None), "hf_token", None)
    if token:
        os.environ.setdefault("HF_TOKEN", token)


def cmd_full_training(args: argparse.Namespace) -> None:
    """Run the complete training sequence (pretrain -> SFT -> instruction tuning)."""
    from src.config.schema import load_config
    from src.training.pipeline import TrainingPipeline
    from src.infrastructure.distributed import DistributedSetup
    from src.tokenizer_trainer import ensure_tokenizer

    config_path = os.path.join(_PROJECT_DIR, args.config)
    cfg = load_config(config_path)
    _apply_hf_token_env(cfg)
    from src.utils.hf_auth import report_hf_auth
    report_hf_auth(cfg)
    dist = DistributedSetup(cfg)

    if dist.is_main_process():
        ensure_tokenizer(config_path=args.config, force=args.fresh_start)
    if dist.is_distributed:
        import torch.distributed as dist_pkg
        dist_pkg.barrier()

    pipeline = TrainingPipeline(cfg, dist_setup=dist)
    try:
        pipeline.initialize(fresh_start=args.fresh_start)
        results = pipeline.full_training_sequence()
        pipeline.save_model()
    finally:
        if SHUTDOWN.requested():
            logger.info("[SHUTDOWN] exiting after signal %s — final cleanup", SHUTDOWN.reason())
        pipeline.cleanup()
    logger.info("Training complete!")


def cmd_config_validate(args: argparse.Namespace) -> None:
    """Validate the configuration file."""
    from src.config.schema import Config, load_config
    from src.models.factory import ModelFactory
    from pydantic import ValidationError
    try:
        config_path = os.path.join(_PROJECT_DIR, args.config)
        cfg = load_config(config_path)
        print("Configuration is valid!")
        print(f"  Model: {cfg.model.name}")
        arch = cfg.model.architecture
        print(f"  Architecture: {arch.model_type} {arch.hidden_size}")
        print(f"  Max context: {arch.max_position_embeddings}")
        print(f"  Training stages: pretrain={cfg.training.pretrain.enabled}, sft={cfg.training.sft.enabled}")
        print(f"  Distributed: {cfg.distributed.strategy}")
        print(f"  Datasets: {len(cfg.data.datasets)}")
        estimate = ModelFactory.estimate_model_size(arch)
        print(f"  Estimated params: {estimate['total_params_b']}B")
    except (yaml.YAMLError, ValidationError, FileNotFoundError) as e:
        print(f"Configuration INVALID: {e}")
        sys.exit(1)


def cmd_info(args: argparse.Namespace) -> None:
    """Print system and model information."""
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB)")
    print(f"Python: {sys.version}")


def cmd_generate(args: argparse.Namespace) -> None:
    """Generate code from a prompt using a trained checkpoint."""
    from src.config.schema import load_config
    from src.models.factory import ModelFactory
    from transformers import AutoTokenizer

    config_path = os.path.join(_PROJECT_DIR, args.config)
    cfg = load_config(config_path)
    checkpoint = args.checkpoint or _resolve_checkpoint(cfg)
    tokenizer = ModelFactory.load_tokenizer(args.tokenizer, cfg)
    model, _ = ModelFactory.load_model(checkpoint, cfg, tokenizer, strict=True)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    prompt = args.prompt or sys.stdin.read().strip()
    if not prompt:
        print("Error: no prompt provided. Use --prompt or pipe text to stdin.")
        sys.exit(1)

    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    output_ids = model.generate(
        inputs["input_ids"],
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        eos_token_id=tokenizer.eos_token_id,
    )
    generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(generated)


def cmd_download_tokenizer(args: argparse.Namespace) -> None:
    """Download a tokenizer from HuggingFace Hub."""
    from src.config.schema import load_config
    from src.tokenizer_trainer import download_tokenizer
    config_path = os.path.join(_PROJECT_DIR, args.config)
    model_id = args.model_id
    if not model_id:
        try:
            cfg = load_config(config_path)
            model_id = cfg.tokenizer.huggingface_model
            print(f"[*] Using model from config: {model_id}")
        except Exception as e:
            model_id = "Xenova/claude-tokenizer"
            print(f"[!] Config load failed ({e}), falling back to {model_id}")
    else:
        print(f"[*] Using explicit model: {model_id}")
    output_dir = os.path.join(_PROJECT_DIR, args.output)
    print(f"[*] Saving tokenizer to: {output_dir}")
    download_tokenizer(
        output_dir=output_dir,
        model_id=model_id,
        force=args.force,
    )
    print(f"[*] Tokenizer download complete: {output_dir}")


def cmd_test(args: argparse.Namespace) -> None:
    """Run the test suite."""
    import subprocess
    import sys
    cmd = [sys.executable, "-m", "pytest", "tests/", "-v"]
    if args.filter:
        cmd.extend(["-k", args.filter])
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    sys.exit(result.returncode)


def cmd_benchmark(args: argparse.Namespace) -> None:
    """Run coding benchmarks against a trained checkpoint."""
    from src.config.schema import load_config
    from src.evaluation.benchmarks import BenchmarkRunner
    from src.models.factory import ModelFactory
    config_path = os.path.join(_PROJECT_DIR, args.config)
    cfg = load_config(config_path)
    checkpoint = args.checkpoint or _resolve_checkpoint(cfg)
    tokenizer = ModelFactory.load_tokenizer(args.tokenizer, cfg)
    model, _ = ModelFactory.load_model(checkpoint, cfg, tokenizer, strict=True)

    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    benchmarks = args.benchmarks.split(",") if args.benchmarks else ["human_eval", "mbpp"]
    runner = BenchmarkRunner(model, tokenizer)
    results = runner.run_benchmarks(benchmarks)
    for r in results:
        print(f"{r.name}: {r.score:.2%}")


# ── GPU Wait + Reserve System ─────────────────────────────────────────


def _gpu_free_mb(gpu_idx: str) -> int:
    """Query free memory (MB) for a specific GPU via nvidia-smi."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", f"--id={gpu_idx}", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            encoding="utf-8",
        )
        return int(out.strip())
    except (subprocess.CalledProcessError, ValueError, OSError) as e:
        raise RuntimeError(f"Failed to query GPU memory for {gpu_idx}: {e}")


def _wait_for_gpu(
    min_free_gb: float,
    poll_interval: int = 30,
    timeout: Optional[int] = None,
) -> str:
    """Poll nvidia-smi until a GPU has >= min_free_gb free, return its index."""
    start = time.time()
    while True:
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
                encoding="utf-8",
            )
            for line in out.strip().split("\n"):
                idx, free_mb = line.split(", ")
                free_gb = int(free_mb) / 1024
                if free_gb >= min_free_gb:
                    return idx
        except Exception as e:
            print(f"[!] GPU query failed: {e}")

        elapsed = time.time() - start
        if timeout is not None and elapsed > timeout:
            raise TimeoutError(
                f"Waited {elapsed:.0f}s but no GPU with >= {min_free_gb} GB free became available."
            )

        mins, secs = divmod(int(elapsed), 60)
        print(
            f"[*] Waiting for GPU with >= {min_free_gb} GB free ... "
            f"({mins}m {secs}s elapsed, checking every {poll_interval}s)"
        )
        time.sleep(poll_interval)


def cmd_reserved_training(args: argparse.Namespace) -> None:
    """
    Wait until a GPU has enough free memory, lock it via a reservation
    tensor, then run full training.
    """
    # 1 ── Wait for a GPU with enough free memory ──────────────────────
    min_free = args.min_free_gb
    if min_free < 0 or args.reserve_gb < 0:
        print("[!] Invalid arguments: min-free-gb and reserve-gb must be non-negative")
        sys.exit(1)
    if args.gpu is not None:
        gpu_idx = str(args.gpu)
        free_gb = _gpu_free_mb(gpu_idx) / 1024
        if free_gb < min_free:
            print(f"[*] GPU {gpu_idx} has {free_gb:.1f} GB free (need {min_free}). Waiting...")
            gpu_idx = _wait_for_gpu(min_free, args.poll_interval, args.timeout)
        else:
            print(f"[*] Using GPU {gpu_idx} ({free_gb:.1f} GB free)")
    else:
        gpu_idx = _wait_for_gpu(min_free, args.poll_interval, args.timeout)

    # 2 ── Set env var and lock the GPU ────────────────────────────────
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_idx
    reservation = None
    if args.reserve_gb > 0:
        num_elements = int(args.reserve_gb * 1024 ** 3 / 4)  # fp32 = 4 bytes
        for attempt in range(5):
            try:
                torch.cuda.empty_cache()
                reservation = torch.empty(num_elements, device="cuda:0", dtype=torch.float32)
                break
            except RuntimeError:
                if attempt < 4:
                    print(f"[!] Reservation attempt {attempt+1} failed, retrying...")
                    time.sleep(2)
                else:
                    print(f"[!] Reservation failed after 5 attempts — training without lock")
        if reservation is not None:
            free_after = _gpu_free_mb(gpu_idx) / 1024
            print(
                f"[*] Reserved GPU {gpu_idx} ({free_after:.1f} GB free after "
                f"{args.reserve_gb:.1f} GB reservation)"
            )

    # 3 ── Run training ────────────────────────────────────────────────
    try:
        cmd_full_training(args)
    except (TimeoutError, RuntimeError) as e:
        print(f"[!] Training failed: {e}")
        sys.exit(1)
    finally:
        if reservation is not None:
            del reservation
            torch.cuda.empty_cache()
        print(f"[*] Released GPU {gpu_idx} reservation")


def build_parser() -> argparse.ArgumentParser:
    main_parent_parser = argparse.ArgumentParser(add_help=False)
    main_parent_parser.add_argument("--config", default="config.yaml", help="Path to config file (default: config.yaml)")
    main_parent_parser.add_argument("--gpu", type=str, default=None, help="GPU index to use (e.g. '0', '3'). Overrides auto-detection.")

    sub_parent_parser = argparse.ArgumentParser(add_help=False)
    sub_parent_parser.add_argument("--config", default=argparse.SUPPRESS, help="Path to config file")
    sub_parent_parser.add_argument("--gpu", type=str, default=None, help="GPU index to use (e.g. '0', '3'). Overrides auto-detection.")

    parser = argparse.ArgumentParser(prog="main.py", description="Methos Class Model - Neural-State Liquid Transformer", parents=[main_parent_parser])

    sub = parser.add_subparsers(dest="command", required=True)

    p_full = sub.add_parser("full-training", help="Run full training (pretrain -> SFT -> instruction tuning)", parents=[sub_parent_parser])
    p_full.add_argument("--fresh-start", action="store_true", help="Ignore existing checkpoints")

    p_res = sub.add_parser("reserved-training", help="Wait for a free GPU, lock it, then train", parents=[sub_parent_parser])
    p_res.add_argument("--fresh-start", action="store_true", help="Ignore existing checkpoints")
    p_res.add_argument("--min-free-gb", type=float, default=50.0, help="Minimum free GPU memory to wait for (GB)")
    p_res.add_argument("--reserve-gb", type=float, default=1.0, help="GPU memory to hold as reservation (GB)")
    p_res.add_argument("--poll-interval", type=int, default=30, help="Seconds between GPU availability checks")
    p_res.add_argument("--timeout", type=int, default=None, help="Max seconds to wait (default: forever)")

    p_config = sub.add_parser("config-validate", help="Validate configuration file", parents=[sub_parent_parser])
    sub.add_parser("info", help="Print system information", parents=[sub_parent_parser])

    p_gen = sub.add_parser("generate", help="Generate code from a prompt", parents=[sub_parent_parser])
    p_gen.add_argument("--prompt", type=str, help="Prompt text (or pipe to stdin)")
    p_gen.add_argument("--checkpoint", type=str, default=None, help="Model checkpoint path (default: from config)")
    p_gen.add_argument("--tokenizer", type=str, default="models/tokenizer", help="Tokenizer path")
    p_gen.add_argument("--max-new-tokens", type=int, default=1024)
    p_gen.add_argument("--temperature", type=float, default=0.7)
    p_gen.add_argument("--top-k", type=int, default=40)
    p_gen.add_argument("--top-p", type=float, default=0.9)

    p_test = sub.add_parser("test", help="Run the test suite", parents=[sub_parent_parser])
    p_test.add_argument("--filter", type=str, help="Filter tests by keyword (-k)")

    p_dl = sub.add_parser("download-tokenizer", help="Download a tokenizer from HuggingFace Hub", parents=[sub_parent_parser])
    p_dl.add_argument("--model-id", type=str, default="", help="HuggingFace model ID (default: from config)")
    p_dl.add_argument("--output", type=str, default="models/tokenizer", help="Output directory")
    p_dl.add_argument("--force", action="store_true", help="Overwrite existing tokenizer")

    p_bench = sub.add_parser("benchmark", help="Run coding benchmarks", parents=[sub_parent_parser])
    p_bench.add_argument("--checkpoint", type=str, default=None, help="Model checkpoint path (default: from config)")
    p_bench.add_argument("--tokenizer", type=str, default="models/tokenizer", help="Tokenizer path")
    p_bench.add_argument("--benchmarks", type=str, default="human_eval,mbpp", help="Comma-separated benchmark names")

    return parser


def _setup_signal_handlers() -> None:
    """Install cooperative SIGINT/SIGTERM shutdown.

    First signal requests a graceful shutdown (flag + SystemExit so normal
    finally-cleanup still drains pools/checkpoints); a second signal or the
    grace watchdog forces an immediate exit so a hung teardown can never
    block an operator kill. Only main() installs handlers — worker modules
    poll SHUTDOWN.requested().
    """
    SHUTDOWN.install()


def main() -> None:
    _setup_signal_handlers()
    cmd_map = {
        "full-training": cmd_full_training,
        "reserved-training": cmd_reserved_training,
        "config-validate": cmd_config_validate,
        "info": cmd_info,
        "generate": cmd_generate,
        "test": cmd_test,
        "download-tokenizer": cmd_download_tokenizer,
        "benchmark": cmd_benchmark,
    }

    is_distributed = os.environ.get("LOCAL_RANK") is not None
    world_size = int(os.environ.get("WORLD_SIZE", "0"))

    if is_distributed and world_size > 1:
        rank = int(os.environ.get("LOCAL_RANK", "0"))
        if rank == 0:
            print(f"[*] Distributed mode with FSDP ({world_size} GPUs)")

    parser = build_parser()
    args = parser.parse_args()

    handler = cmd_map.get(args.command)
    if handler:
        handler(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
