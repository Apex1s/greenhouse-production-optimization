"""Solve the greenhouse plan from the command line.

Examples (from the repository root)::

    python src/solve.py
    python src/solve.py --max-avg-risk 5
    python src/solve.py --max-peak-labour 500 --max-avg-risk 5 --csv plan.csv

Labour-constrained solves default to a 1% gap and a 120 s time limit, the
settings used in the notebooks. The exit code is 0 for a usable plan and 1
otherwise.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import enumerate_options, load_data, solve_plan, validate_data  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", default=os.path.join(ROOT, "data"),
                        help="folder holding the four CSV files (default: data/)")
    parser.add_argument("--production-cap", type=float, metavar="KG",
                        help="ceiling on total season output")
    parser.add_argument("--max-avg-risk", type=float, metavar="SCORE",
                        help="ceiling on the mean risk score per greenhouse (out of 10)")
    parser.add_argument("--max-peak-labour", type=float, metavar="PICKERS",
                        help="ceiling on pickers needed in the busiest harvest week")
    parser.add_argument("--sector-uniform", action="store_true",
                        help="every greenhouse in a sector takes the same scenario")
    parser.add_argument("--gap", type=float,
                        help="relative MIP gap (default: 0.01 with a labour cap, exact otherwise)")
    parser.add_argument("--time-limit", type=int, metavar="SECONDS",
                        help="solver time limit (default: 120 with a labour cap, none otherwise)")
    parser.add_argument("--csv", metavar="PATH", help="also write the plan to this CSV file")
    parser.add_argument("--quiet", action="store_true", help="skip data-quality warnings")
    args = parser.parse_args(argv)
    if args.max_peak_labour is not None:
        args.gap = 0.01 if args.gap is None else args.gap
        args.time_limit = 120 if args.time_limit is None else args.time_limit
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    data = load_data(args.data)
    if not args.quiet:
        for warning in validate_data(data):
            print(f"warning: {warning}", file=sys.stderr)

    result = solve_plan(
        enumerate_options(data),
        data,
        production_cap=args.production_cap,
        max_avg_risk=args.max_avg_risk,
        max_peak_labour=args.max_peak_labour,
        sector_uniform=args.sector_uniform,
        gap=args.gap,
        time_limit=args.time_limit,
    )

    print(f"status        {result.status}")
    if not result.feasible:
        print("no plan satisfies these constraints")
        return 1

    print(f"profit        € {result.profit:,.0f}")
    print(f"production    {result.production_kg:,.0f} kg")
    print(f"mean risk     {result.avg_risk:.2f} / 10")
    print(f"peak crew     {result.peak_pickers:,.0f} pickers")
    print()
    summary = (result.plan
               .groupby(["scenario", "crop", "variety", "start_week"])
               .agg(greenhouses=("greenhouse", lambda g: ", ".join(map(str, g))),
                    kg=("production_kg", "sum"), profit=("profit", "sum"))
               .reset_index())
    print(summary.to_string(index=False, float_format=lambda v: f"{v:,.0f}"))

    if args.csv:
        result.plan.to_csv(args.csv, index=False)
        print(f"\nplan written to {args.csv}")
    if not result.proven_optimal:
        print(f"\nnote: {result.status.lower()}; the profit is not proven optimal", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
