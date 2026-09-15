"""Spending limits shared across calls.

A :class:`Budget` is a thread-safe USD counter with a hard limit. Before each request the
cost guard *reserves* the estimated cost; after the response arrives the reservation is
replaced by the actual cost. Concurrent calls therefore cannot overshoot the limit by
racing each other.

Budgets can be attached in three ways::

    @callm(budget=Budget(10.00))            # per function
    with callm.budget(0.50):                 # per session / request scope
        ...
    user_budget = callm.budget_for(f"user:{user_id}", limit=5.00)   # per user
"""

from __future__ import annotations

import threading
from contextvars import ContextVar
from types import TracebackType

from callm.errors import BudgetExceeded, ConfigurationError

_ACTIVE: ContextVar[tuple[Budget, ...]] = ContextVar("callm_active_budgets", default=())


class Reservation:
    """A pending charge against a budget. Call :meth:`commit` or :meth:`release` exactly once."""

    __slots__ = ("_budget", "_done", "amount")

    def __init__(self, budget: Budget, amount: float) -> None:
        self._budget = budget
        self.amount = amount
        self._done = False

    def commit(self, actual: float | None) -> None:
        if self._done:
            return
        self._done = True
        self._budget._settle(self.amount, actual if actual is not None else 0.0)

    def release(self) -> None:
        if self._done:
            return
        self._done = True
        self._budget._settle(self.amount, 0.0)


def _check_limit(limit: float) -> float:
    if isinstance(limit, bool) or not isinstance(limit, (int, float)) or limit < 0:
        raise ConfigurationError("Budget limit must be a non-negative number of US dollars")
    return float(limit)


class Budget:
    """A hard USD spending limit."""

    def __init__(self, limit: float, name: str | None = None) -> None:
        self._limit = _check_limit(limit)
        self.name = name
        self._spent = 0.0
        self._reserved = 0.0
        self._lock = threading.Lock()

    # ----------------------------------------------------------------- accounting

    @property
    def limit(self) -> float:
        return self._limit

    @limit.setter
    def limit(self, value: float) -> None:
        self._limit = _check_limit(value)

    @property
    def spent(self) -> float:
        return self._spent

    @property
    def reserved(self) -> float:
        return self._reserved

    @property
    def remaining(self) -> float:
        return max(self._limit - self._spent - self._reserved, 0.0)

    def reserve(self, estimate: float) -> Reservation:
        """Reserve ``estimate`` dollars or raise :class:`BudgetExceeded`."""
        estimate = max(float(estimate), 0.0)
        with self._lock:
            committed = self._spent + self._reserved
            if committed + estimate > self._limit + 1e-12:
                label = f" '{self.name}'" if self.name else ""
                raise BudgetExceeded(
                    f"Budget{label} exceeded: estimated ${estimate:.6f} with "
                    f"${committed:.6f} already committed of ${self._limit:.6f}",
                    scope=f"budget:{self.name}" if self.name else "budget",
                    limit=self._limit,
                    estimated=estimate,
                    spent=committed,
                )
            self._reserved += estimate
        return Reservation(self, estimate)

    def add(self, cost: float) -> None:
        """Record spend that happened outside callm's cost guard."""
        with self._lock:
            self._spent += max(float(cost), 0.0)

    def reset(self) -> None:
        with self._lock:
            self._spent = 0.0
            self._reserved = 0.0

    def _settle(self, reserved: float, actual: float) -> None:
        with self._lock:
            self._reserved = max(self._reserved - reserved, 0.0)
            self._spent += max(actual, 0.0)

    # ----------------------------------------------------------------- scoping

    def __enter__(self) -> Budget:
        _ACTIVE.set((*_ACTIVE.get(), self))
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        active = list(_ACTIVE.get())
        for index in range(len(active) - 1, -1, -1):
            if active[index] is self:
                del active[index]
                break
        _ACTIVE.set(tuple(active))

    async def __aenter__(self) -> Budget:
        return self.__enter__()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.__exit__(exc_type, exc, tb)

    def __repr__(self) -> str:
        label = f"{self.name!r}, " if self.name else ""
        return f"Budget({label}limit={self._limit:.4f}, spent={self._spent:.6f})"


def budget(limit: float, name: str | None = None) -> Budget:
    """Create a budget to use as a context manager: ``with callm.budget(1.00): ...``."""
    return Budget(limit, name=name)


def active_budgets() -> tuple[Budget, ...]:
    """Budgets entered with ``with`` in the current context."""
    return _ACTIVE.get()


_registry: dict[str, Budget] = {}
_registry_lock = threading.Lock()


def budget_for(key: str, limit: float | None = None) -> Budget:
    """Return the process-wide budget registered under ``key``, creating it if needed.

    Use it for per-user or per-tenant limits::

        with callm.budget_for(f"user:{user.id}", limit=2.00):
            answer = ask(question)

    ``limit`` is required the first time a key is used; passing a different limit later
    updates the existing budget.
    """
    with _registry_lock:
        existing = _registry.get(key)
        if existing is None:
            if limit is None:
                raise ConfigurationError(f"budget_for({key!r}) needs a limit the first time")
            existing = _registry[key] = Budget(limit, name=key)
        elif limit is not None:
            existing.limit = limit
        return existing


def clear_budgets() -> None:
    """Forget all budgets created with :func:`budget_for`."""
    with _registry_lock:
        _registry.clear()


__all__ = ["Budget", "Reservation", "active_budgets", "budget", "budget_for", "clear_budgets"]
