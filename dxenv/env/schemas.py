"""Pydantic models for everything that crosses a boundary.

The observation model is the load-bearing one: `Observation` has no field capable of
holding a condition label, so I1 ("ground truth never enters an observation") is a
property of the type, not of the care taken by the code that builds it.
"""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

Probability = Annotated[float, Field(ge=0.0, le=1.0)]


class Strict(BaseModel):
    """Base: reject unknown fields everywhere. The allowlist fails closed [I2]."""

    model_config = ConfigDict(extra="forbid", frozen=True)


# ------------------------------------------------------------------ observations ----


class Demographics(Strict):
    age_years: Annotated[int, Field(ge=0, le=120)]
    sex: Literal["female", "male", "other"]


class AnalyteResult(Strict):
    """One revealed result. Deliberately carries no interpretation.

    `display` is the analyte's own name, taken from the catalog -- never a string lifted
    out of the patient record, because a lab display reading "HbA1c - diabetes
    monitoring" is the label in disguise.
    """

    analyte: str
    display: str
    unit: str = ""
    value_number: float | None = None
    value_code: str | None = None
    ref_low: float | None = None
    ref_high: float | None = None

    @model_validator(mode="after")
    def exactly_one_value(self) -> Self:
        if (self.value_number is None) == (self.value_code is None):
            raise ValueError(
                "exactly one of value_number / value_code must be set; a result that is "
                "neither is the 'unavailable' case that I4 forbids"
            )
        return self


class Observation(Strict):
    """What the agent sees. Structurally incapable of carrying the label [I1].

    There is no free-text field, no `condition`, no `reason`, no `note`. Adding one is
    not a small change -- it reopens the channel the whole environment exists to close.
    """

    patient_ref: str
    turn: int
    demographics: Demographics
    presenting_complaint: str
    vitals: tuple[AnalyteResult, ...]
    family_history: tuple[str, ...]
    allergies: tuple[str, ...]
    revealed_results: tuple[AnalyteResult, ...]
    remaining_budget: float
    turns_remaining: int
    menu_fingerprint: str


# ----------------------------------------------------------------------- actions ----


class OrderTest(Strict):
    kind: Literal["order_test"] = "order_test"
    action_id: str
    test_key: str
    prediction: Literal["low", "normal", "high", "abnormal_categorical", "normal_categorical"]
    """Predict-then-verify commitment. MANDATORY -- a test order without a prediction is
    rejected here, at the schema, so `test_commit_is_mandatory` cannot be satisfied by a
    runtime check someone later forgets to call."""


class Prescribe(Strict):
    kind: Literal["prescribe"] = "prescribe"
    action_id: str
    treatment_key: str


class Diagnose(Strict):
    kind: Literal["diagnose"] = "diagnose"
    action_id: str
    distribution: dict[str, Probability]

    @model_validator(mode="after")
    def sums_to_one(self) -> Self:
        total = sum(self.distribution.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"diagnosis distribution sums to {total}, not 1")
        return self


class Abstain(Strict):
    kind: Literal["abstain"] = "abstain"
    action_id: str


Action = OrderTest | Prescribe | Diagnose | Abstain


# ------------------------------------------------------------------ trajectories ----


class Step(Strict):
    """One turn, as stored. Enough to rescore offline without replaying the env [I8]."""

    turn: int
    action: Action
    revealed: tuple[AnalyteResult, ...] = ()
    was_duplicate: bool = False
    cost_charged: float = 0.0


class Trajectory(Strict):
    """The persisted episode. Reward is a pure function of this plus ground truth."""

    patient_id: str
    seed: int
    config_hash: str
    menu_fingerprint: str
    budget: float
    steps: tuple[Step, ...]
    terminated: bool
    termination_reason: Literal["diagnose", "abstain", "max_turns", "budget_exhausted"]
