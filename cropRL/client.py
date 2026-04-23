"""Croprl Environment Client."""

from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult

from .models import CroprlAction, CroprlObservation, CroprlState


class CroprlEnv(EnvClient[CroprlAction, CroprlObservation, CroprlState]):
    """Typed client for the Croprl environment."""

    def _step_payload(self, action: CroprlAction) -> dict:
        """Serialize a CroprlAction to the wire format."""
        return {"action_id": action.action_id}

    def _parse_result(self, payload: dict) -> StepResult:
        """Deserialize a step response into a StepResult."""
        obs_data = payload.get("observation", {})
        return StepResult(
            observation=CroprlObservation(
                done=payload.get("done", False),
                reward=payload.get("reward", 0.0),
                current_month=obs_data.get("current_month", 1),
                current_step=obs_data.get("current_step", 0),
                expected_rainfall=obs_data.get("expected_rainfall", 0.0),
                active_crop_type=obs_data.get("active_crop_type", 0),
                crop_age_months=obs_data.get("crop_age_months", 0),
                expected_yield_potential=obs_data.get("expected_yield_potential", 0.0),
                soil_nitrogen=obs_data.get("soil_nitrogen", 0.0),
                current_water_level=obs_data.get("current_water_level", 0.0),
                cash_balance=obs_data.get("cash_balance", 0.0),
                current_debt=obs_data.get("current_debt", 0.0),
                current_interest_rate=obs_data.get("current_interest_rate", 0.0),
                market_price_crop_1=obs_data.get("market_price_crop_1", 0.0),
                market_price_crop_2=obs_data.get("market_price_crop_2", 0.0),
                market_price_crop_3=obs_data.get("market_price_crop_3", 0.0),
                cost_seed_1=obs_data.get("cost_seed_1", 0.0),
                cost_seed_2=obs_data.get("cost_seed_2", 0.0),
                cost_seed_3=obs_data.get("cost_seed_3", 0.0),
                cost_irrigate=obs_data.get("cost_irrigate", 0.0),
                cost_fertilize=obs_data.get("cost_fertilize", 0.0),
                stored_crop_type=obs_data.get("stored_crop_type", 0),
                stored_amount=obs_data.get("stored_amount", 0.0),
                stored_age_months=obs_data.get("stored_age_months", 0),
                message=obs_data.get("message", ""),
                text_summary=obs_data.get("text_summary", ""),
            ),
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict) -> CroprlState:
        """Deserialize a state response into a CroprlState."""
        return CroprlState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            irrigated_this_month=payload.get("irrigated_this_month", False),
            fertilized_this_month=payload.get("fertilized_this_month", False),
            previous_cash=payload.get("previous_cash", 0.0),
            has_active_loan=payload.get("has_active_loan", False),
            task_id=payload.get("task_id", "default"),
        )
