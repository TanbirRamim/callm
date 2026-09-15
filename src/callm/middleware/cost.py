"""Cost guard: per-call limits and shared budgets, enforced before the request is sent."""

from __future__ import annotations

import logging

from callm.budgets import Reservation, active_budgets
from callm.config import get_settings
from callm.errors import BudgetExceeded
from callm.pipeline import CallState, Handler, Step
from callm.pricing import estimate_cost
from callm.types import LLMResponse

logger = logging.getLogger("callm")
_warned_unpriced: set[str] = set()


class CostGuardMiddleware:
    name = "cost_guard"

    def handle(self, state: CallState, call_next: Handler) -> Step[LLMResponse]:
        config = state.config
        target = state.target or state.request.target
        request = (
            state.request if target == state.request.target else state.request.retarget(target)
        )
        assumed_output = get_settings().assumed_output_tokens

        estimate, _, _ = estimate_cost(request, assumed_output)
        if estimate is not None:
            previous = state.record.estimated_cost_usd
            state.record.estimated_cost_usd = (
                estimate if previous is None else max(previous, estimate)
            )

        budgets = list(dict.fromkeys((*config.budgets, *active_budgets())))
        if config.max_cost is None and not budgets:
            return (yield from call_next(state))

        if estimate is None:
            if target.label not in _warned_unpriced:
                _warned_unpriced.add(target.label)
                logger.warning(
                    "callm: cannot enforce cost limits for %s because its price is unknown; "
                    "register one with callm.set_price()",
                    target,
                )
        elif config.max_cost is not None and estimate > config.max_cost:
            hint = (
                ""
                if request.params.get("max_tokens")
                else f" (assuming {assumed_output} output tokens; set max_tokens to tighten)"
            )
            raise BudgetExceeded(
                f"Estimated cost ${estimate:.6f} for {target} exceeds max_cost "
                f"${config.max_cost:.6f}{hint}",
                scope="call",
                limit=config.max_cost,
                estimated=estimate,
            )

        reservations: list[Reservation] = []
        try:
            for budget in budgets:
                reservations.append(budget.reserve(estimate or 0.0))
        except BudgetExceeded:
            for reservation in reservations:
                reservation.release()
            raise

        try:
            response = yield from call_next(state)
        except BaseException:
            for reservation in reservations:
                reservation.release()
            raise

        actual = response.cost if response.cost is not None else estimate
        for reservation in reservations:
            reservation.commit(actual)
        if (
            config.max_cost is not None
            and response.cost is not None
            and response.cost > config.max_cost
        ):
            logger.warning(
                "callm: actual cost $%.6f for %s exceeded max_cost $%.6f (estimate $%.6f)",
                response.cost,
                target,
                config.max_cost,
                estimate or 0.0,
            )
        return response


__all__ = ["CostGuardMiddleware"]
