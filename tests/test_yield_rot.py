import pytest
from cropRL.config import EnvConfig
from cropRL.dynamics import calculate_expected_yield_potential, calculate_yield

def test_yield_decays_after_maturity(config):
    """Ensure that the yield potential decays after crossing growth_months."""
    crop_type = 1  # Corn, 4-month maturity
    maturity = config.growth_months[crop_type]

    # Use Monsoon (month 7) for optimal season, good water and soil
    yield_at_peak = calculate_expected_yield_potential(
        crop_type=crop_type, crop_age=maturity,
        soil_nitrogen=0.6, current_water_level=0.6,
        current_month=7, config=config,
    )
    yield_late_1 = calculate_expected_yield_potential(
        crop_type=crop_type, crop_age=maturity + 1,
        soil_nitrogen=0.6, current_water_level=0.6,
        current_month=7, config=config,
    )
    yield_late_2 = calculate_expected_yield_potential(
        crop_type=crop_type, crop_age=maturity + 2,
        soil_nitrogen=0.6, current_water_level=0.6,
        current_month=7, config=config,
    )

    assert yield_at_peak > 0.0, "Peak yield should be positive"
    assert yield_late_1 < yield_at_peak, "Yield must decay 1 month after maturity"
    assert yield_late_2 < yield_late_1, "Yield must decay further 2 months after"
    assert yield_late_2 == 0.0, "Should be fully rotted 2 months past peak"

def test_yield_grows_before_maturity(config):
    crop_type = 1
    # Use Monsoon for optimal season
    y_2 = calculate_expected_yield_potential(crop_type, 2, 0.6, 0.6, 7, config)
    y_3 = calculate_expected_yield_potential(crop_type, 3, 0.6, 0.6, 7, config)
    y_4 = calculate_expected_yield_potential(crop_type, 4, 0.6, 0.6, 7, config)

    assert 0 < y_2 < y_3 < y_4, "Yield should climb before reaching maturity."
