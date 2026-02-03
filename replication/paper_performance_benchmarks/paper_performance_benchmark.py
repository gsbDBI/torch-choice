"""Master CLI for paper performance benchmarks.

Commands:
  - generate: synthetic data generation
  - torch-choice: torch-choice benchmarking
  - visualize: build figures (v2)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from steps import (
    step01_generate_synthetic_data as _step01_generate_synthetic_data,
    step02_torch_choice_benchmark as _step02_torch_choice_benchmark,
    step03_performance_visualization_v2 as _step03_performance_visualization_v2,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Paper performance benchmarks master CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    gen_parent = _step01_generate_synthetic_data.build_arg_parser(add_help=False)
    subparsers.add_parser("generate", parents=[gen_parent], help="Generate synthetic datasets.")

    torch_parent = _step02_torch_choice_benchmark.build_arg_parser(add_help=False)
    subparsers.add_parser("torch-choice", parents=[torch_parent], help="Run torch-choice benchmarks.")

    vis_parent = _step03_performance_visualization_v2.build_arg_parser(add_help=False)
    subparsers.add_parser("visualize", parents=[vis_parent], help="Generate benchmark figures (v2).")

    return parser


def main(argv: List[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "generate":
        _step01_generate_synthetic_data.run(
            output_path=args.output_path,
            experiments=args.experiments,
            num_records=args.num_records,
            skip_plots=args.skip_plots,
            reference_path=args.reference_path,
            smoke_test=getattr(args, "smoke_test", False),
        )
    elif args.command == "torch-choice":
        _step02_torch_choice_benchmark._set_device(args.device)
        # Resolve batch size: None means auto-detect
        if args.batch_size is None:
            args.batch_size = _step02_torch_choice_benchmark._auto_batch_size()
        _step02_torch_choice_benchmark.run_all(args)
    elif args.command == "visualize":
        _step03_performance_visualization_v2.visualize(
            torch_results=args.torch_results,
            r_results=args.r_results,
            output_path=args.output_path,
        )
    else:
        parser.error("Unknown command.")


if __name__ == "__main__":
    main()

