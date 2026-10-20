"""Mixed-integer model for assigning production scenarios to greenhouses.

The farm has 24 greenhouses in 6 sectors. Each must be assigned exactly one
*option*: a (scenario, planting week) pair. A scenario fixes the crop, variety,
planting method and planting month; the week fixes when inside that month the
planting starts, which shifts the whole harvest profile against the weekly
price curve.

The objective is total profit. Four optional constraints can be switched on
independently, which is what makes the model useful for trade-off analysis
rather than a single answer:

``production_cap``      total season output ceiling, in kg
``max_avg_risk``        ceiling on the mean agronomic risk score per greenhouse
``max_peak_labour``     ceiling on pickers needed in the busiest harvest week
``sector_uniform``      every greenhouse in a sector takes the same scenario

Usage::

    data = load_data("data")
    options = enumerate_options(data)
    result = solve_plan(options, data, max_avg_risk=5.0)
    print(result.profit, result.plan)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import pandas as pd
import pulp

SCENARIO_COLUMN = "Scénario"
LABOUR_COST_COLUMN = "Coûts de Main d'œuvre par kg de production"

# A weekly yield below half of both neighbouring weeks is flagged by
# validate_data() as a probable data-entry error.
YIELD_DIP_RATIO = 0.5

# Planting can start in any week of the scenario's nominated month.
PLANTING_WEEKS = {
    "Avril": [14, 15, 16, 17, 18],
    "Mai": [18, 19, 20, 21, 22],
    "Juin": [22, 23, 24, 25, 26],
    "Juillet": [27, 28, 29, 30, 31],
    "Aout": [31, 32, 33, 34, 35],
    "Octobre": [40, 41, 42, 43, 44],
    "Novembre": [44, 45, 46, 47, 48],
}

SECTORS = {1: [1, 2, 3, 4, 5, 6], 2: [7, 8, 9, 10, 11], 3: [12, 13, 14, 15, 16, 17],
           4: [18, 19], 5: [20, 21], 6: [22, 23, 24]}

SECTOR_6_ONLY = {4, 5}
SECTOR_5_ONLY = {12, 13, 14, 18}
SECTOR_5_MAX_SURFACE = 2.87          # see validate_data(): never binds on this dataset
PRODUCTION_WEEKS = range(1, 38)
HARVEST_DAYS_PER_WEEK = 6


@dataclass
class FarmData:
    production: pd.DataFrame
    prices: pd.DataFrame
    greenhouses: pd.DataFrame
    charges: pd.DataFrame

    @property
    def n_greenhouses(self) -> int:
        return len(self.greenhouses)

    @property
    def n_scenarios(self) -> int:
        return len(self.production)

    def scenario_number(self, row: int) -> int:
        """The farm's own (non-contiguous) number for a Production.csv row."""
        return int(self.production[SCENARIO_COLUMN].iloc[row])


@dataclass
class Option:
    """One assignable (greenhouse, scenario, planting week) triple."""
    greenhouse: int                   # 1-based, as the farm numbers them
    scenario_row: int                 # 0-based row in Production.csv
    scenario: int                     # the farm's scenario number
    start_week: int
    profit: float
    production_kg: float
    risk: float
    pickers_by_week: dict = field(default_factory=dict)

    @property
    def key(self) -> tuple:
        return (self.greenhouse, self.scenario_row, self.start_week)


# Statuses for which a usable plan exists. CBC stopped by a time limit or a
# relative gap returns an integer-feasible plan that is not proven optimal.
PROVEN_OPTIMAL = "Optimal"
OPTIMAL_WITHIN_GAP = "Optimal within gap"
FEASIBLE_NOT_PROVEN = "Feasible (not proven optimal)"


@dataclass
class PlanResult:
    status: str
    profit: float | None
    plan: pd.DataFrame | None
    production_kg: float | None = None
    avg_risk: float | None = None
    peak_pickers: float | None = None

    @property
    def feasible(self) -> bool:
        return self.status in (PROVEN_OPTIMAL, OPTIMAL_WITHIN_GAP, FEASIBLE_NOT_PROVEN)

    @property
    def proven_optimal(self) -> bool:
        return self.status == PROVEN_OPTIMAL


def _read_csv(data_dir: str, name: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(data_dir, name))


def load_data(data_dir: str = "data") -> FarmData:
    """Read the four CSVs and return them as one object.

    Charges_var.csv is re-ordered to follow Production.csv by scenario number,
    so every later lookup by row position refers to the same scenario in both
    tables whatever order the files are saved in.
    """
    production = _read_csv(data_dir, "Production.csv")
    charges = _read_csv(data_dir, "Charges_var.csv")

    scenarios = production[SCENARIO_COLUMN]
    if scenarios.duplicated().any():
        raise ValueError("Production.csv lists the same scenario twice")
    missing = set(scenarios) - set(charges[SCENARIO_COLUMN])
    if missing:
        raise ValueError(f"Charges_var.csv has no row for scenario(s) {sorted(missing)}")
    charges = charges.set_index(SCENARIO_COLUMN).loc[scenarios].reset_index()

    return FarmData(
        production=production,
        prices=_read_csv(data_dir, "Prices.csv"),
        greenhouses=_read_csv(data_dir, "Simulation.csv"),
        charges=charges,
    )


def validate_data(data: FarmData) -> list[str]:
    """Return a list of data-quality warnings rather than raising.

    These are real defects in the source files that silently change the
    model's behaviour, so they are surfaced instead of being absorbed.
    """
    warnings = []

    if (data.greenhouses["Surface"] < SECTOR_5_MAX_SURFACE).all():
        warnings.append(
            f"The {SECTOR_5_MAX_SURFACE} ha ceiling on sector-5 scenarios never binds: "
            f"the largest greenhouse is {data.greenhouses['Surface'].max()} ha."
        )

    costs = data.charges["Cout"]
    outliers = costs[costs > 10 * costs.median()]
    for row in outliers.index:
        warnings.append(
            f"Scenario {data.scenario_number(row)} carries a cost of {costs[row]:,.0f} "
            f"against a median of {costs.median():,.0f} — a sentinel value that "
            f"makes the scenario unselectable rather than a real price."
        )

    weeks = [f"W{w}" for w in PRODUCTION_WEEKS]
    for row in range(data.n_scenarios):
        yields = data.production[weeks].iloc[row].to_numpy(dtype=float)
        for i in range(1, len(yields) - 1):
            neighbours = min(yields[i - 1], yields[i + 1])
            if neighbours > 0 and yields[i] < YIELD_DIP_RATIO * neighbours:
                warnings.append(
                    f"Scenario {data.scenario_number(row)}, {weeks[i]}: {yields[i]:,.0f} kg/ha "
                    f"between {yields[i - 1]:,.0f} and {yields[i + 1]:,.0f} — probable "
                    f"data-entry error, check against the source."
                )

    for crop in ("Framboise", "Mure"):
        zeros = (data.prices[crop] == 0).sum()
        if zeros:
            warnings.append(f"{crop}: {zeros} week(s) priced at zero in Prices.csv.")

    missing = set(data.production["Mois"]) - set(PLANTING_WEEKS)
    if missing:
        warnings.append(f"No planting weeks defined for month(s): {sorted(missing)}")

    return warnings


def _is_allowed(scenario: int, sector: int, surface: float) -> bool:
    if scenario in SECTOR_6_ONLY and sector != 6:
        return False
    if scenario in SECTOR_5_ONLY and (sector != 5 or surface >= SECTOR_5_MAX_SURFACE):
        return False
    return True


def _price(data: FarmData, crop: str, calendar_week: int) -> float:
    """Market price in a given calendar week; zero outside the priced horizon."""
    if 1 <= calendar_week <= len(data.prices):
        return float(data.prices[crop].iloc[calendar_week - 1])
    return 0.0


def enumerate_options(data: FarmData) -> list[Option]:
    """Build every feasible (greenhouse, scenario, week) option with its economics.

    Harvest in production week ``w`` lands in calendar week
    ``start_week + w + delay``, which is what aligns yield against the price
    curve and against every other greenhouse's harvest for the labour
    constraint.
    """
    options: list[Option] = []

    for _, gh in data.greenhouses.iterrows():
        greenhouse, sector, surface = int(gh["Serre"]), int(gh["Secteur"]), float(gh["Surface"])

        for row in range(data.n_scenarios):
            scenario = data.scenario_number(row)
            if not _is_allowed(scenario, sector, surface):
                continue

            crop = data.production["Culture"][row]
            delay = int(data.production["Delai"][row])
            fixed_cost = surface * float(data.charges["Cout"][row])
            labour_cost_per_kg = float(data.charges[LABOUR_COST_COLUMN][row])
            risk = float(data.charges["Risque/10"][row])
            speed = float(data.charges["Vitesse de main d'œuvre kg/personne/jour"][row])

            for start_week in PLANTING_WEEKS[data.production["Mois"][row]]:
                revenue = 0.0
                total_kg = 0.0
                pickers: dict[int, float] = {}

                for w in PRODUCTION_WEEKS:
                    yield_kg = float(data.production[f"W{w}"][row]) * surface
                    if yield_kg == 0:
                        continue
                    calendar_week = start_week + w + delay
                    revenue += yield_kg * _price(data, crop, calendar_week)
                    total_kg += yield_kg
                    pickers[calendar_week] = yield_kg / (speed * HARVEST_DAYS_PER_WEEK)

                cost = fixed_cost + labour_cost_per_kg * total_kg
                options.append(Option(
                    greenhouse=greenhouse, scenario_row=row, scenario=scenario,
                    start_week=start_week, profit=revenue - cost, production_kg=total_kg,
                    risk=risk, pickers_by_week=pickers,
                ))

    return options


def solve_plan(
    options: list[Option],
    data: FarmData,
    production_cap: float | None = None,
    max_avg_risk: float | None = None,
    max_peak_labour: float | None = None,
    sector_uniform: bool = False,
    gap: float | None = None,
    time_limit: int | None = None,
    verbose: bool = False,
) -> PlanResult:
    """Solve the assignment under whichever constraints are supplied.

    ``gap`` accepts a relative MIP gap (0.005 = stop within 0.5% of proven
    optimal). The labour constraint couples every greenhouse through shared
    harvest weeks and makes branch-and-bound far slower than the unconstrained
    problem; a small gap keeps a sweep tractable, and 0.5% is well inside the
    precision of the underlying yield forecasts. Leave it None for an exact
    solve.
    """
    model = pulp.LpProblem("greenhouse_plan", pulp.LpMaximize)
    x = {o.key: pulp.LpVariable(f"x_{o.greenhouse}_{o.scenario_row}_{o.start_week}",
                                cat="Binary") for o in options}

    model += pulp.lpSum(o.profit * x[o.key] for o in options)

    # Every greenhouse gets exactly one scenario and one start week.
    for greenhouse in data.greenhouses["Serre"].astype(int):
        model += pulp.lpSum(x[o.key] for o in options if o.greenhouse == greenhouse) == 1

    if production_cap is not None:
        model += pulp.lpSum(o.production_kg * x[o.key] for o in options) <= production_cap

    if max_avg_risk is not None:
        model += (pulp.lpSum(o.risk * x[o.key] for o in options)
                  <= max_avg_risk * data.n_greenhouses)

    if max_peak_labour is not None:
        # One constraint per calendar week: the crew needed that week, summed
        # across every greenhouse harvesting simultaneously, stays under the cap.
        all_weeks = sorted({w for o in options for w in o.pickers_by_week})
        for week in all_weeks:
            model += pulp.lpSum(o.pickers_by_week.get(week, 0.0) * x[o.key]
                                for o in options) <= max_peak_labour

    if sector_uniform:
        for members in SECTORS.values():
            first, *rest = members
            rows = sorted({o.scenario_row for o in options if o.greenhouse == first})
            for row in rows:
                lead = pulp.lpSum(x[o.key] for o in options
                                  if o.greenhouse == first and o.scenario_row == row)
                for other in rest:
                    model += pulp.lpSum(x[o.key] for o in options
                                        if o.greenhouse == other
                                        and o.scenario_row == row) == lead
            # A scenario the leader cannot take is not shared by the sector, so
            # no other member may take it either.
            for o in options:
                if o.greenhouse in rest and o.scenario_row not in rows:
                    model += x[o.key] == 0

    solver_args = {"msg": verbose}
    if gap is not None:
        solver_args["gapRel"] = gap
    if time_limit is not None:
        solver_args["timeLimit"] = time_limit

    model.solve(pulp.PULP_CBC_CMD(**solver_args))
    # model.status reads "Optimal" whenever CBC returns an integer solution,
    # including when it stopped on the time limit or the gap; sol_status tells
    # a proven optimum apart from a merely feasible plan.
    if model.sol_status == pulp.LpSolutionOptimal:
        status = PROVEN_OPTIMAL if gap is None else OPTIMAL_WITHIN_GAP
    elif model.sol_status == pulp.LpSolutionIntegerFeasible:
        status = FEASIBLE_NOT_PROVEN
    else:
        return PlanResult(status=pulp.LpStatus[model.status], profit=None, plan=None)

    chosen = [o for o in options if x[o.key].varValue and x[o.key].varValue > 0.5]
    chosen.sort(key=lambda o: o.greenhouse)

    plan = pd.DataFrame([{
        "greenhouse": o.greenhouse,
        "sector": int(data.greenhouses.loc[data.greenhouses["Serre"] == o.greenhouse, "Secteur"].iloc[0]),
        "scenario": o.scenario,
        "crop": data.production["Culture"][o.scenario_row],
        "variety": data.production["variété 23-24"][o.scenario_row],
        "start_week": o.start_week,
        "production_kg": round(o.production_kg, 1),
        "profit": round(o.profit, 2),
        "risk": o.risk,
    } for o in chosen])

    weekly = {}
    for o in chosen:
        for week, pickers in o.pickers_by_week.items():
            weekly[week] = weekly.get(week, 0.0) + pickers

    return PlanResult(
        status=status,
        profit=pulp.value(model.objective),
        plan=plan,
        production_kg=sum(o.production_kg for o in chosen),
        avg_risk=sum(o.risk for o in chosen) / len(chosen),
        peak_pickers=max(weekly.values()) if weekly else 0.0,
    )


def weekly_labour_profile(result: PlanResult, options: list[Option]) -> pd.Series:
    """Pickers required in each calendar week under a solved plan."""
    if result.plan is None:
        raise ValueError(f"no plan to profile: solver status is {result.status!r}")
    chosen_keys = {(r.greenhouse, r.scenario, r.start_week)
                   for r in result.plan.itertuples()}
    weekly: dict[int, float] = {}
    for o in options:
        if (o.greenhouse, o.scenario, o.start_week) in chosen_keys:
            for week, pickers in o.pickers_by_week.items():
                weekly[week] = weekly.get(week, 0.0) + pickers
    return pd.Series(weekly).sort_index()
