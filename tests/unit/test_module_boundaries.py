"""The import rules from CLAUDE.md 3, enforced rather than documented.

  reward/ must not import policy/ or train/
  env/    must not import reward/

The second is what makes offline rescoring of a stored trajectory corpus free: the
environment produces trajectories, the reward engine scores them, and `env.step()`
deliberately returns no reward. If `env` could reach `reward`, every stored rollout would
be quietly coupled to the config that generated it.

Checked by parsing the source rather than by importing, so a violation is caught even if
the offending import is inside a function.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PACKAGE = Path("dxenv")

FORBIDDEN = {
    "reward": ("dxenv.policy", "dxenv.train"),
    "env": ("dxenv.reward",),
    # data/ underpins everything and must not depend on anything above it. This is what
    # keeps `data.store` free of an import cycle with `reward.engine`, which is why the
    # rescoring entry point takes its scoring function as an injected callable.
    "data": ("dxenv.reward", "dxenv.policy", "dxenv.train", "dxenv.eval"),
}


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            out.add(node.module)
    return out


@pytest.mark.parametrize("subpackage", sorted(FORBIDDEN))
def test_subpackage_respects_its_import_rules(subpackage: str) -> None:
    forbidden = FORBIDDEN[subpackage]
    for path in sorted((PACKAGE / subpackage).rglob("*.py")):
        for module in _imported_modules(path):
            for bad in forbidden:
                assert not (module == bad or module.startswith(bad + ".")), (
                    f"{path} imports {module}; dxenv/{subpackage} must not depend on {bad} "
                    "(CLAUDE.md 3)"
                )


def test_env_step_returns_no_reward() -> None:
    """Structural: the environment produces trajectories, it does not score them."""
    import inspect

    from dxenv.env.episode import DiagnosticEpisode

    sig = inspect.signature(DiagnosticEpisode.step)
    assert "reward" not in str(sig.return_annotation).lower()


def test_boundary_checker_would_catch_a_violation(tmp_path) -> None:
    """Test the detector. A parse-based check that matched nothing would look identical."""
    bad = tmp_path / "bad.py"
    bad.write_text("def f():\n    from dxenv.policy.sft import SFTDataset\n    return SFTDataset\n")
    assert "dxenv.policy.sft" in _imported_modules(bad)


def test_no_module_reads_config_at_import_time() -> None:
    """CLAUDE.md 3: anything that reads config takes it as an argument.

    A module-level config read makes the config invisible in the call signature and
    un-overridable in a test, and it fixes the value at import time -- which is how a run
    ends up scored under a config nobody passed.
    """
    loaders = {"load_reward_config", "load_episode_config", "load_catalog",
               "load_taxonomy", "load_severity", "load_curriculum", "load_cost_table"}
    for path in sorted(PACKAGE.rglob("*.py")):
        tree = ast.parse(path.read_text())
        for node in tree.body:  # module level only
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            for call in ast.walk(node):
                if isinstance(call, ast.Call) and isinstance(call.func, ast.Name):
                    assert call.func.id not in loaders, (
                        f"{path} calls {call.func.id}() at module level"
                    )
