"""Drive a real model through the environment once. The cheapest possible first contact.

Everything this touches has never run: the vLLM engine, guided decoding against the action
grammar, the parse back into a typed `Action`, and the episode loop driven by a model
rather than a heuristic. Each is a place where an API has plausibly moved since the code
was written, and each would otherwise first execute inside a multi-hour Gate B job.

Deliberately tiny. A 0.5B model cannot diagnose anything, and the output is expected to be
poor -- what is being tested is that the plumbing works at all, and that every generation
comes back schema-valid. If this passes, the Gate B sweep will run; whether it scores well
is a different question and the gate is what answers it.
"""

from __future__ import annotations

import argparse
import json

from dxenv.data.corpus import generate_corpus
from dxenv.env.episode import load_episode_config
from dxenv.policy.decoding import action_json_schema, parse_action, schema_fingerprint
from dxenv.policy.llm import LLMPolicy, VLLMBackend
from dxenv.policy.rollout import RolloutContext, rollout_once


def probe_vllm_api() -> None:
    """Report the shape of vLLM's structured-output API before using it.

    This code was written against vLLM ~0.6 and the cluster has 0.28 -- roughly twenty
    releases, over which the structured-output parameter has been renamed at least once
    (`guided_decoding` -> `structured_outputs`, `GuidedDecodingParams` ->
    `StructuredOutputsParams`). On a batch scheduler every fix costs a queue wait, so this
    prints what the INSTALLED version actually offers rather than only what failed. One
    run is then enough to correct the call sites.

    Never raises: a probe that aborts the job it is diagnosing is worse than no probe.
    """
    import inspect

    print("--- vLLM API probe ---")
    import os

    print(f"VLLM_USE_FLASHINFER_SAMPLER={os.environ.get('VLLM_USE_FLASHINFER_SAMPLER', '<unset>')} "
          f"CUDA_HOME={os.environ.get('CUDA_HOME', '<unset>')}")
    try:
        import vllm

        print(f"vllm {vllm.__version__}")
    except Exception as exc:
        print(f"could not import vllm: {type(exc).__name__}: {exc}")
        return

    for mod, names in [
        ("vllm.sampling_params",
         ["GuidedDecodingParams", "StructuredOutputsParams"]),
        ("vllm", ["LLM", "SamplingParams"]),
    ]:
        try:
            m = __import__(mod, fromlist=["_"])
            present = [n for n in names if hasattr(m, n)]
            missing = [n for n in names if not hasattr(m, n)]
            print(f"{mod}: present={present} missing={missing}")
        except Exception as exc:
            print(f"{mod}: import failed ({type(exc).__name__})")

    try:
        from vllm import LLM, SamplingParams

        params = set(inspect.signature(SamplingParams).parameters)
        interesting = sorted(
            p for p in params
            if any(k in p for k in ("guided", "structured", "json", "grammar", "seed", "n"))
        )
        print(f"SamplingParams structured-output params: {interesting}")
        print(f"LLM.__init__ params: {sorted(inspect.signature(LLM.__init__).parameters)}")
        if hasattr(LLM, "chat"):
            print(f"LLM.chat params: {sorted(inspect.signature(LLM.chat).parameters)}")
        else:
            print("LLM has NO .chat method")
    except Exception as exc:
        print(f"signature inspection failed: {type(exc).__name__}: {exc}")
    print("--- end probe ---\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--n", type=int, default=2, help="patients to run")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.60)
    args = ap.parse_args()

    probe_vllm_api()

    schema = action_json_schema()
    print(f"grammar fingerprint {schema_fingerprint(schema)}, "
          f"{len(schema['oneOf'])} action kinds")

    backend = VLLMBackend(
        model=args.model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    ctx = RolloutContext(episode_config=load_episode_config())
    records = generate_corpus(args.n, seed=1234)

    n_gen = 0
    for i, rec in enumerate(records):
        policy = LLMPolicy(backend=backend, temperature=args.temperature, seed=i)
        # Default-arg binding, not a bare closure: the same policy OBJECT has to come
        # back so its recorded generations can be checked afterwards.
        rollout = rollout_once(rec, lambda _s, p=policy: p, seed=i, ctx=ctx, budget=100.0)
        for gen in rollout.generations:
            # Under guided decoding this cannot fail. Asserting it anyway is the point:
            # "cannot fail" is a claim about the backend, and this is where it is checked
            # rather than assumed.
            parse_action(str(gen["completion"]))
            n_gen += 1
        print(
            f"  patient {i}: {rollout.breakdown.n_turns} turns, "
            f"{rollout.n_tests} tests, terminated on "
            f"{rollout.breakdown.termination_reason}, reward {rollout.reward:+.3f} "
            f"(hard ceiling {rollout.hard_ceiling:+.3f})"
        )
        assert rollout.reward <= rollout.hard_ceiling + 1e-6, "I9 violated on first contact"

    print(f"\n{n_gen} generations, all schema-valid, all episodes below the hard ceiling.")
    print("first completion:")
    print("  " + json.dumps(json.loads(str(records and rollout.generations[0]['completion'])),
                            indent=2).replace("\n", "\n  ")[:600])


if __name__ == "__main__":
    main()
