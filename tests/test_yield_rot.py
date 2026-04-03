import pytest
from cropRL.config import EnvConfig
from cropRL.dynamics import calculate_expected_yield_potential, calculate_yield

def test_yield_decays_after_maturity(config):
    """Ensure that the yield potential decays mathematically after crossing growth_months."""
    # We use crop_type=1 (Corn) which has a 4-month maturity in default config.
    crop_type = 1
    maturity = config.growth_months[crop_type]
    
    # Store yield potentials at varying ages
    yield_at_peak = calculate_expected_yield_potential(
        crop_type=crop_type,
        crop_age=maturity,
        soil_nitrogen=0.6,
        expected_rainfall=0.6,
        config=config,
    )
    
    yield_late_1 = calculate_expected_yield_potential(
        crop_type=crop_type,
        crop_age=maturity + 1,
        soil_nitrogen=0.6,
        expected_rainfall=0.6,
        config=config,
    )
    
    yield_late_2 = calculate_expected_yield_potential(
        crop_type=crop_type,
        crop_age=maturity + 2,
        soil_nitrogen=0.6,
        expected_rainfall=0.6,
        config=config,
    )
    
    # Assertions
    assert yield_at_peak > 0.0, "Peak yield should be substantially positive"
    assert yield_late_1 < yield_at_peak, "Yield must decay 1 month after maturity"
    assert yield_late_2 < yield_late_1, "Yield must continue to decay 2 months after maturity"
    assert yield_late_2 == 0.0, "Given the 50% decay per month, it should be fully rotted by month 2 past peak"

def test_yield_grows_before_maturity(config):
    crop_type = 1
    maturity = config.growth_months[crop_type]

    y_2 = calculate_expected_yield_potential(crop_type, 2, 0.6, 0.6, config)
    y_3 = calculate_expected_yield_potential(crop_type, 3, 0.6, 0.6, config)
    y_4 = calculate_expected_yield_potential(crop_type, 4, 0.6, 0.6, config)
    
    assert 0 < y_2 < y_3 < y_4, "Yield should climb before reaching maturity."
