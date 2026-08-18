"""The global action space [I3].

The menu is built from the catalog alone. `build_menu()` takes no patient argument and
there is no code path that could give it one -- if the menu were derived per patient,
the menu would *be* the diagnosis.

Action ids are content-hashed, not positional (`test_action_ids_stable`), so inserting a
test into the catalog does not renumber the others and invalidate stored trajectories.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
from typing import Final

from dxenv.env.catalog import Catalog, load_catalog

ID_LENGTH: Final = 16


class ActionKind(StrEnum):
    ORDER_TEST = "order_test"
    PRESCRIBE = "prescribe"
    DIAGNOSE = "diagnose"
    ABSTAIN = "abstain"


TERMINAL_KINDS: Final = frozenset({ActionKind.DIAGNOSE, ActionKind.ABSTAIN})


class ActionError(ValueError):
    """Unknown action id or malformed action. Never caught inside `dxenv.env`."""


def action_id(kind: ActionKind, key: str) -> str:
    """Stable content-addressed id: `<kind>:<16 hex of sha256(kind|key)>`.

    Positional ids are the trap here: renumbering on catalog edits silently remaps every
    stored trajectory, and nothing fails loudly when it happens.
    """
    digest = hashlib.sha256(f"{kind.value}|{key}".encode()).hexdigest()[:ID_LENGTH]
    return f"{kind.value}:{digest}"


@dataclass(frozen=True, slots=True)
class Action:
    action_id: str
    kind: ActionKind
    key: str
    display: str


@dataclass(frozen=True, slots=True)
class ActionMenu:
    """The complete, patient-independent action menu."""

    actions: tuple[Action, ...]

    def __post_init__(self) -> None:
        ids = [a.action_id for a in self.actions]
        if len(set(ids)) != len(ids):
            raise ActionError("action id collision in the menu")

    @property
    def ids(self) -> frozenset[str]:
        return frozenset(a.action_id for a in self.actions)

    def by_id(self, aid: str) -> Action:
        for a in self.actions:  # menu is ~150 entries; a dict view is built below
            if a.action_id == aid:
                return a
        raise ActionError(f"action id not on the menu: {aid!r}")

    @property
    def index(self) -> dict[str, Action]:
        return {a.action_id: a for a in self.actions}

    def test_actions(self) -> tuple[Action, ...]:
        return tuple(a for a in self.actions if a.kind is ActionKind.ORDER_TEST)

    def treatment_actions(self) -> tuple[Action, ...]:
        return tuple(a for a in self.actions if a.kind is ActionKind.PRESCRIBE)

    def test_key_for(self, aid: str) -> str:
        a = self.by_id(aid)
        if a.kind is not ActionKind.ORDER_TEST:
            raise ActionError(f"{aid!r} is not a test order")
        return a.key

    def id_for_test(self, test_key: str) -> str:
        return action_id(ActionKind.ORDER_TEST, test_key)

    def id_for_treatment(self, treatment_key: str) -> str:
        return action_id(ActionKind.PRESCRIBE, treatment_key)

    def fingerprint(self) -> str:
        """Hash of the menu; pinned into every run config and stored trajectory."""
        blob = "|".join(sorted(a.action_id for a in self.actions))
        return hashlib.sha256(blob.encode()).hexdigest()


def _build(catalog: Catalog) -> ActionMenu:
    actions: list[Action] = []
    for key in catalog.test_keys:
        spec = catalog.test(key)
        actions.append(
            Action(action_id(ActionKind.ORDER_TEST, key), ActionKind.ORDER_TEST, key, spec.display)
        )
    for key in catalog.treatment_keys:
        t = catalog.treatment(key)
        actions.append(
            Action(action_id(ActionKind.PRESCRIBE, key), ActionKind.PRESCRIBE, key, t.display)
        )
    actions.append(
        Action(
            action_id(ActionKind.DIAGNOSE, "diagnose"),
            ActionKind.DIAGNOSE,
            "diagnose",
            "Declare a probability distribution over conditions",
        )
    )
    actions.append(
        Action(
            action_id(ActionKind.ABSTAIN, "abstain"),
            ActionKind.ABSTAIN,
            "abstain",
            "Abstain from diagnosis",
        )
    )
    return ActionMenu(tuple(actions))


@lru_cache(maxsize=1)
def build_menu() -> ActionMenu:
    """The one menu. Deliberately takes no patient and no config."""
    return _build(load_catalog())
