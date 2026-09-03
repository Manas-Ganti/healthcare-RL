"""Append-only trajectory store: one JSON line per episode, under a pinned config hash.

CLAUDE.md 4: *persist every trajectory ever generated*. Reward is a pure function of
(trajectory, ground_truth, reward_config) [I8], so rescoring a stored corpus under new
weights is free -- regenerating rollouts is not, and on a 7B policy it is the dominant
cost of the whole project. You WILL change the reward weights repeatedly. This module is
what makes that cheap.

Ground truth is stored ON THE LINE, under its own top-level `ground_truth` key, so a
stored run is self-contained and rescoring needs no corpus alongside it. That is a
deliberate trade and it has one sharp edge: **an episodes.jsonl file is not safe to feed
to a model.** The rule is structural rather than advisory -- `stored_trajectory()`
returns the trajectory sub-object alone, and it is the only accessor the training and
rollout paths use. `test_stored_trajectory_excludes_ground_truth` asserts the separation
over a real run.

This module imports nothing from `dxenv` on purpose. `dxenv.data.taxonomy` is imported by
`dxenv.reward`, so a store that reached back into `reward` for scoring would close an
import cycle; the rescoring entry point takes the scoring function as an injected
callable instead, the same way `env/bayes.py` takes the scoring rule.
"""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Callable, Iterator, Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Final, Self

EPISODES_FILE: Final = "episodes.jsonl"
META_FILE: Final = "meta.json"
DEFAULT_ROOT: Final = Path("runs")


class StoreError(ValueError):
    """Malformed run metadata, or a hash mismatch against an existing run. Never caught."""


def git_sha() -> str:
    """Short HEAD sha, or "unknown" outside a checkout.

    Recorded per run because "which weights produced this number" and "which code
    produced this number" are different questions and both get asked.
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown"
    return out.stdout.strip() or "unknown"


@dataclass(frozen=True, slots=True)
class RunMeta:
    """Everything needed to interpret a run's episodes, and to refuse to mix two runs."""

    run_id: str
    env_config_hash: str
    reward_config_hash: str
    menu_fingerprint: str
    taxonomy_hash: str
    phase: str = "unspecified"
    policy: str = "unspecified"
    git_sha: str = field(default_factory=git_sha)
    created_utc: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    notes: dict[str, Any] = field(default_factory=dict)

    # The fields that must match for two writes to belong to the same run. `created_utc`,
    # `notes` and `git_sha` deliberately are not: resuming a run after a docstring fix is
    # normal, resuming it after the cost table changed is not.
    IDENTITY: Final = (
        "run_id", "env_config_hash", "reward_config_hash", "menu_fingerprint",
        "taxonomy_hash",
    )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def assert_compatible(self, other: Mapping[str, Any]) -> None:
        drift = {
            k: (other.get(k), getattr(self, k))
            for k in self.IDENTITY
            if other.get(k) != getattr(self, k)
        }
        if drift:
            raise StoreError(
                f"run {self.run_id!r} already exists with different pinned hashes: "
                f"{drift}. Appending would mix episodes generated under two different "
                "configurations into one file, and nothing downstream could tell them "
                "apart. Start a new run_id."
            )


@dataclass(frozen=True, slots=True)
class StoredEpisode:
    """One line. `trajectory` is model-safe; `ground_truth` is emphatically not."""

    trajectory: dict[str, Any]
    ground_truth: dict[str, Any]
    tags: dict[str, Any] = field(default_factory=dict)

    def as_line(self) -> str:
        payload = {
            "trajectory": self.trajectory,
            "ground_truth": self.ground_truth,
            "tags": self.tags,
        }
        # sort_keys so a stored corpus is byte-stable across runs and diffable [I10].
        return json.dumps(payload, separators=(",", ":"), sort_keys=True, default=_encode)


def _encode(obj: object) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "as_dict"):
        return obj.as_dict()  # type: ignore[no-any-return]
    raise TypeError(f"{type(obj)!r} is not JSON-serialisable; convert it at the call site")


class TrajectoryStore:
    """Append-only writer for `runs/{run_id}/episodes.jsonl`.

    Flushes every line. A training run that dies at step 900 must not lose 900 steps of
    rollouts -- they are the expensive artifact, and buffering them to save syscalls
    trades the cheap resource for the expensive one.
    """

    def __init__(self, meta: RunMeta, root: Path | None = None) -> None:
        self.meta = meta
        self.root = (root or DEFAULT_ROOT) / meta.run_id
        self.path = self.root / EPISODES_FILE
        self._fh: Any = None
        self._n = 0

    def __enter__(self) -> Self:
        return self.open()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def open(self) -> Self:
        self.root.mkdir(parents=True, exist_ok=True)
        meta_path = self.root / META_FILE
        if meta_path.exists():
            self.meta.assert_compatible(json.loads(meta_path.read_text()))
        else:
            meta_path.write_text(json.dumps(self.meta.as_dict(), indent=2, default=_encode) + "\n")
        self._fh = self.path.open("a", encoding="utf-8")
        return self

    def append(
        self,
        trajectory: Mapping[str, Any],
        ground_truth: Mapping[str, Any],
        **tags: Any,
    ) -> None:
        if self._fh is None:
            raise StoreError("store is not open; use `with TrajectoryStore(meta) as s:`")
        traj = dict(trajectory)
        if traj.get("config_hash") != self.meta.env_config_hash:
            raise StoreError(
                f"episode was generated under env config {traj.get('config_hash')!r} but "
                f"this run pins {self.meta.env_config_hash!r}. A stored corpus whose "
                "lines were produced under different configs cannot be rescored as one."
            )
        line = StoredEpisode(traj, dict(ground_truth), dict(tags)).as_line()
        self._fh.write(line + "\n")
        self._fh.flush()
        os.fsync(self._fh.fileno())
        self._n += 1

    @property
    def n_written(self) -> int:
        return self._n

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


def read_episodes(path: Path) -> Iterator[StoredEpisode]:
    """Stream a stored run. Raises on a malformed line rather than skipping it.

    Skipping is the tempting behaviour and the wrong one: a run that silently drops 3%
    of its episodes produces a mean that is 3% selected-on-something, and nothing says so.
    """
    with path.open(encoding="utf-8") as fh:
        for i, raw in enumerate(fh, start=1):
            if not raw.strip():
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise StoreError(f"{path}:{i} is not valid JSON: {exc}") from exc
            missing = {"trajectory", "ground_truth"} - set(obj)
            if missing:
                raise StoreError(f"{path}:{i} is missing {sorted(missing)}")
            yield StoredEpisode(obj["trajectory"], obj["ground_truth"], obj.get("tags", {}))


def stored_trajectory(episode: StoredEpisode) -> dict[str, Any]:
    """The model-safe projection of a stored line. The ONLY accessor rollout code uses."""
    return dict(episode.trajectory)


def read_meta(root: Path) -> dict[str, Any]:
    p = root / META_FILE
    if not p.exists():
        raise StoreError(f"{p} does not exist; a run without pinned hashes is uninterpretable")
    return dict(json.loads(p.read_text()))


def rescore(
    path: Path,
    score_fn: Callable[[dict[str, Any], dict[str, Any]], float],
) -> list[float]:
    """Rescore a stored run under a new reward config.

    `score_fn` is injected -- `dxenv.reward` imports `dxenv.data`, so importing the
    reward engine here would close a cycle. Callers pass a closure over
    `reward.engine.score_trajectory` and the config they want; see `scripts/rescore.py`.
    """
    return [score_fn(ep.trajectory, ep.ground_truth) for ep in read_episodes(path)]
