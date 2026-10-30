"""Command-line entry points: demo and benchmark."""

from __future__ import annotations

import argparse
import json

from .benchmark import run_grid, run_stream
from .controller import PortalConfig
from .data import StreamConfig, generate_stream
from .experiment import SCENARIOS, SEEDS, aggregate, write_results
from .policies import PolicySpec, default_policies


def demo(n_batches: int = 6, seed: int = 0) -> None:
    config = StreamConfig(
        scenario="mixed_arrivals", seed=seed, n_batches=n_batches
    )
    batches = generate_stream(config)
    spec = PolicySpec(
        name="portal",
        kind="portal",
        portal_config=PortalConfig(max_labels=16, seed=0),
    )
    result = run_stream(spec, config, batches=batches)
    print("batch  arrival  domain   choice       accuracy  labels")
    for i, batch in enumerate(batches):
        print(
            f"{i:>5}  {batch.arrival:<7}  {batch.domain:<8}  "
            f"{result.choices[i]:<12}  {result.batch_accuracies[i]:.3f}     "
            f"{result.labels_used[i]}"
        )
    summary = result.summary()
    print(
        f"mean accuracy {summary['accuracy']:.3f}, "
        f"mean harm {summary['mean_harm']:.3f}, "
        f"labels/batch {summary['labels_per_batch']:.1f}"
    )


def benchmark(experiment_path: str, measured_path: str) -> None:
    rows = run_grid(SCENARIOS, default_policies(), SEEDS)
    measured = aggregate(rows)
    write_results(rows, measured, experiment_path, measured_path)
    print(json.dumps(measured["headline"], indent=2, sort_keys=True))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="portal-tta")
    sub = parser.add_subparsers(dest="command", required=True)
    demo_p = sub.add_parser("demo", help="run one mixed-arrivals stream")
    demo_p.add_argument("--n-batches", type=int, default=6)
    demo_p.add_argument("--seed", type=int, default=0)
    bench_p = sub.add_parser("benchmark", help="run the full grid")
    bench_p.add_argument("--experiment-out", default="results/experiment.json")
    bench_p.add_argument("--measured-out", default="results/measured.json")
    args = parser.parse_args(argv)
    if args.command == "demo":
        demo(n_batches=args.n_batches, seed=args.seed)
    elif args.command == "benchmark":
        benchmark(args.experiment_out, args.measured_out)


if __name__ == "__main__":
    main()
