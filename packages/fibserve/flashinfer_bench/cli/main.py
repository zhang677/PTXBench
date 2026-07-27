"""Command-line entry point for the FIBServe HTTP service."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)
pkg_name = __name__.split(".")[0]


def cli_config_logging(args: argparse.Namespace) -> None:
    """Configure package-level logging from CLI arguments."""
    pkg_logger = logging.getLogger(pkg_name)
    pkg_logger.setLevel(args.log_level)
    if not pkg_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                fmt="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        pkg_logger.addHandler(handler)
    pkg_logger.propagate = False


def serve(args: argparse.Namespace) -> None:
    """Start the FIBServe HTTP server."""
    try:
        import uvicorn
    except ImportError as error:
        raise RuntimeError(
            "uvicorn is required for FIBServe. Install with: pip install 'fibserve[serve]'"
        ) from error

    from flashinfer_bench.bench import BenchmarkConfig
    from flashinfer_bench.data import TraceSet
    from flashinfer_bench.serve.app import init_app
    from flashinfer_bench.serve.scheduler import Scheduler

    trace_set = TraceSet.from_paths(args.local)
    devices = args.devices.split(",") if args.devices else None
    if devices is None:
        from flashinfer_bench.utils import list_cuda_devices

        devices = list_cuda_devices()
    if not devices:
        raise RuntimeError("No CUDA devices available")

    raw_overrides = {
        "warmup_runs": args.warmup_runs,
        "iterations": args.iterations,
        "num_trials": args.num_trials,
        "rtol": args.rtol,
        "atol": args.atol,
        "required_matched_ratio": args.required_matched_ratio,
        "timeout_seconds": args.timeout,
    }
    overrides = {key: value for key, value in raw_overrides.items() if value is not None}
    if args.config:
        config = BenchmarkConfig.from_yaml(args.config, **overrides)
    else:
        config = BenchmarkConfig.default(**overrides)

    scheduler = Scheduler(trace_set=trace_set, config=config, devices=devices)
    app = init_app(scheduler)
    logger.info("Starting FIBServe on %s:%d with devices %s", args.host, args.port, devices)
    uvicorn.run(app, host=args.host, port=args.port)


def cli() -> None:
    parser = argparse.ArgumentParser(description="FIBServe GPU evaluation service")
    subparsers = parser.add_subparsers(dest="command", required=True)
    serve_parser = subparsers.add_parser("serve", help="Start the HTTP evaluation service")
    serve_parser.add_argument(
        "--local",
        type=Path,
        nargs="+",
        required=True,
        help="Path(s) to FlashInfer trace-set datasets",
    )
    serve_parser.add_argument(
        "--devices",
        default=None,
        help="Comma-separated CUDA devices (for example, cuda:0,cuda:1)",
    )
    serve_parser.add_argument("--host", default="0.0.0.0")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument("--warmup-runs", type=int, default=None)
    serve_parser.add_argument("--iterations", type=int, default=None)
    serve_parser.add_argument("--num-trials", type=int, default=None)
    serve_parser.add_argument("--rtol", type=float, default=None)
    serve_parser.add_argument("--atol", type=float, default=None)
    serve_parser.add_argument("--required-matched-ratio", type=float, default=None)
    serve_parser.add_argument("--timeout", type=int, default=None)
    serve_parser.add_argument("--config", type=str, default=None)
    serve_parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    serve_parser.set_defaults(func=serve)

    args = parser.parse_args()
    cli_config_logging(args)
    args.func(args)
