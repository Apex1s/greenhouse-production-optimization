import os
import sys

import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from model import (PROVEN_OPTIMAL, SECTORS, enumerate_options, load_data,  # noqa: E402
                   solve_plan, validate_data, weekly_labour_profile)

DATA_DIR = os.path.join(ROOT, "data")


@pytest.fixture(scope="module")
def data():
    return load_data(DATA_DIR)


@pytest.fixture(scope="module")
def options(data):
    return enumerate_options(data)


@pytest.fixture(scope="module")
def base(data, options):
    return solve_plan(options, data)


def test_base_plan_is_proven_optimal(base):
    assert base.status == PROVEN_OPTIMAL
    assert base.proven_optimal and base.feasible
    assert base.profit == pytest.approx(11_206_575.0)


def test_every_greenhouse_gets_exactly_one_option(data, base):
    assert sorted(base.plan.greenhouse) == sorted(data.greenhouses["Serre"].astype(int))


def test_options_carry_the_farm_scenario_number(data, options):
    assert {o.scenario for o in options} <= set(data.production["Scénario"])
    for o in options[:50]:
        assert data.scenario_number(o.scenario_row) == o.scenario


def test_charges_are_joined_by_scenario_not_position(tmp_path, data):
    for name in ("Production.csv", "Prices.csv", "Simulation.csv"):
        (tmp_path / name).write_bytes(open(os.path.join(DATA_DIR, name), "rb").read())
    shuffled = pd.read_csv(os.path.join(DATA_DIR, "Charges_var.csv")).iloc[::-1]
    shuffled.to_csv(tmp_path / "Charges_var.csv", index=False)

    reloaded = load_data(str(tmp_path))
    pd.testing.assert_frame_equal(reloaded.charges, data.charges)


def test_infeasible_constraint_returns_no_plan(data, options):
    result = solve_plan(options, data, production_cap=1_000)
    assert not result.feasible
    assert result.profit is None and result.plan is None
    with pytest.raises(ValueError):
        weekly_labour_profile(result, options)


def test_time_limited_solve_is_not_reported_as_proven(data, options):
    result = solve_plan(options, data, max_peak_labour=200, gap=0.0, time_limit=1)
    assert not result.proven_optimal


def test_sector_uniform_plan_is_uniform(data, options):
    result = solve_plan(options, data, sector_uniform=True)
    assert result.feasible
    for members in SECTORS.values():
        scenarios = result.plan.loc[result.plan.greenhouse.isin(members), "scenario"]
        assert scenarios.nunique() == 1


def test_validate_data_flags_known_defects(data):
    warnings = "\n".join(validate_data(data))
    assert "Scenario 14 carries a cost" in warnings
    assert "Scenario 3, W3" in warnings
