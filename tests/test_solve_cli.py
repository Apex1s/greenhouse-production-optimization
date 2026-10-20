import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from solve import main, parse_args  # noqa: E402


def test_cli_solves_and_writes_the_plan(tmp_path, capsys):
    out = tmp_path / "plan.csv"
    assert main(["--max-avg-risk", "5", "--csv", str(out), "--quiet"]) == 0

    printed = capsys.readouterr().out
    assert "status        Optimal" in printed
    assert "€ 11,197,575" in printed
    plan = pd.read_csv(out)
    assert len(plan) == 24 and set(plan["scenario"]) == {15}


def test_cli_reports_infeasible_constraints(capsys):
    assert main(["--production-cap", "1000", "--quiet"]) == 1
    assert "no plan satisfies" in capsys.readouterr().out


def test_labour_cap_gets_notebook_solver_defaults():
    args = parse_args(["--max-peak-labour", "500"])
    assert args.gap == 0.01 and args.time_limit == 120
    exact = parse_args([])
    assert exact.gap is None and exact.time_limit is None
