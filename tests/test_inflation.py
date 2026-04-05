"""Tests for compounding inflation system."""

from cropRL.config import EnvConfig
from cropRL.models import CroprlAction
from cropRL.server.cropRL_environment import CroprlEnvironment
from cropRL.enums import ActionType


class TestInflation:
    """Verify that inflation compounds correctly each year."""

    def test_no_inflation_when_rate_zero(self):
        """Easy mode has 0% inflation — costs should stay constant."""
        cfg = EnvConfig(inflation_rate=0.0)
        e = CroprlEnvironment(config=cfg)
        e.reset(seed=42)

        initial_seed_1 = e._internal["inflated_seed_costs"][1]

        # Wait 12 months to complete a year
        for _ in range(12):
            e.step(CroprlAction(action_id=ActionType.WAIT))

        assert e._internal["inflated_seed_costs"][1] == initial_seed_1

    def test_inflation_applies_on_year_boundary(self):
        """Costs should increase at the start of each new year (Jan)."""
        cfg = EnvConfig(inflation_rate=0.03)
        e = CroprlEnvironment(config=cfg)
        e.reset(seed=42)

        initial_seed_1 = e._internal["inflated_seed_costs"][1]

        # Wait 12 months (Jan→Jan), inflation fires at month=1
        for _ in range(12):
            e.step(CroprlAction(action_id=ActionType.WAIT))

        inflated = e._internal["inflated_seed_costs"][1]
        expected = initial_seed_1 * 1.03
        assert abs(inflated - expected) < 0.1

    def test_inflation_compounds_over_years(self):
        """After 2 years at 3%, costs should be base × 1.03²."""
        cfg = EnvConfig(inflation_rate=0.03)
        e = CroprlEnvironment(config=cfg)
        e.reset(seed=42)

        initial = e._internal["inflated_seed_costs"][1]

        # Wait 24 months for 2 years
        for _ in range(24):
            obs = e.step(CroprlAction(action_id=ActionType.WAIT))
            if obs.done:
                break

        expected = initial * (1.03 ** 2)
        actual = e._internal["inflated_seed_costs"][1]
        assert abs(actual - expected) < 1.0

    def test_inflation_affects_all_costs(self):
        """All inflatable values should increase together."""
        cfg = EnvConfig(inflation_rate=0.05)
        e = CroprlEnvironment(config=cfg)
        e.reset(seed=42)

        # Record initial values
        init_irrigate = e._internal["inflated_cost_irrigate"]
        init_fert = e._internal["inflated_cost_fertilize"]
        init_fixed = e._internal["inflated_monthly_fixed_cost"]

        # Wait 12 months
        for _ in range(12):
            e.step(CroprlAction(action_id=ActionType.WAIT))

        factor = 1.05
        assert abs(e._internal["inflated_cost_irrigate"] - init_irrigate * factor) < 0.1
        assert abs(e._internal["inflated_cost_fertilize"] - init_fert * factor) < 0.1
        assert abs(e._internal["inflated_monthly_fixed_cost"] - init_fixed * factor) < 0.1

    def test_land_price_inflates(self):
        """Base land price should inflate, affecting observation."""
        cfg = EnvConfig(inflation_rate=0.10)
        e = CroprlEnvironment(config=cfg)
        e.reset(seed=42)

        init_land_base = e._internal["inflated_base_land_price"]

        for _ in range(12):
            e.step(CroprlAction(action_id=ActionType.WAIT))

        expected = init_land_base * 1.10
        assert abs(e._internal["inflated_base_land_price"] - expected) < 1.0
