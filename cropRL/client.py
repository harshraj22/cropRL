"""Croprl Environment Client."""
from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult
from .models import CroprlAction, CroprlObservation, CroprlState

class CroprlEnv(EnvClient[CroprlAction, CroprlObservation, CroprlState]):
    def _step_payload(self, action: CroprlAction) -> dict:
        p = {"action_id": action.action_id, "farmer_id": action.farmer_id}
        if action.forum_message: p["forum_message"] = action.forum_message
        return p

    def _parse_result(self, payload: dict) -> StepResult:
        od = payload.get("observation", {})
        return StepResult(
            observation=CroprlObservation(
                done=payload.get("done", False), reward=payload.get("reward", 0.0),
                farmer_id=od.get("farmer_id", 0), current_month=od.get("current_month", 1),
                current_step=od.get("current_step", 0), month_step=od.get("month_step", 0),
                expected_rainfall=od.get("expected_rainfall", 0.0),
                active_crop_type=od.get("active_crop_type", 0),
                crop_age_months=od.get("crop_age_months", 0),
                expected_yield_potential=od.get("expected_yield_potential", 0.0),
                soil_nitrogen=od.get("soil_nitrogen", 0.0),
                current_water_level=od.get("current_water_level", 0.0),
                cash_balance=od.get("cash_balance", 0.0), current_debt=od.get("current_debt", 0.0),
                current_interest_rate=od.get("current_interest_rate", 0.0),
                current_land_price=od.get("current_land_price", 0.0),
                market_price_crop_1=od.get("market_price_crop_1", 0.0),
                market_price_crop_2=od.get("market_price_crop_2", 0.0),
                market_price_crop_3=od.get("market_price_crop_3", 0.0),
                cost_seed_1=od.get("cost_seed_1", 0.0), cost_seed_2=od.get("cost_seed_2", 0.0),
                cost_seed_3=od.get("cost_seed_3", 0.0), cost_irrigate=od.get("cost_irrigate", 0.0),
                cost_fertilize=od.get("cost_fertilize", 0.0),
                stored_crop_type=od.get("stored_crop_type", 0),
                stored_amount=od.get("stored_amount", 0.0),
                stored_age_months=od.get("stored_age_months", 0),
                other_farmers_crops=od.get("other_farmers_crops", []),
                forum_messages=od.get("forum_messages", []),
                message=od.get("message", ""), text_summary=od.get("text_summary", "")),
            reward=payload.get("reward", 0.0), done=payload.get("done", False))

    def _parse_state(self, payload: dict) -> CroprlState:
        return CroprlState(
            episode_id=payload.get("episode_id"), step_count=payload.get("step_count", 0),
            irrigated_this_month=payload.get("irrigated_this_month", False),
            fertilized_this_month=payload.get("fertilized_this_month", False),
            previous_cash=payload.get("previous_cash", 0.0),
            has_active_loan=payload.get("has_active_loan", False),
            task_id=payload.get("task_id", "default"))
