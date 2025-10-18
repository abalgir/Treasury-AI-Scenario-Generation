#!/usr/bin/env python3
"""
orchestrator_phase1.py - minimalist no-args orchestrator

Behavior:
 - Deterministically calls main() in a fixed list of modules (in order).
 - For each module it tries these import paths in order:
     1) with_whom.<module>
     2) <module>
 - If the module is found, it attempts to call module.main() (no args).
 - If module.main() raises a TypeError because it expects arguments,
   it will try calling main(None) and main({}) as fallbacks.
 - Prints clear status for each step and keeps running through the list.
 - Stops after attempting the last module in the ordered list.

Note: this intentionally keeps logic simple and imperative as requested.
"""

from __future__ import annotations

import importlib
import sys
import traceback
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent
WITH_WHOM = ROOT / "with_whom"
DATA_DIR = WITH_WHOM / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def _try_import(module_name: str):
    """Try import with_whom.<module_name> then <module_name>. Return module or None."""
    candidates = [f"with_whom.{module_name}", module_name]
    for cand in candidates:
        try:
            mod = importlib.import_module(cand)
            print(f"[IMPORT] Imported {cand}")
            return mod
        except Exception as e:
            # keep trying
            print(f"[IMPORT] Could not import {cand}: {e}")
    return None


def _call_main(mod) -> bool:
    """
    Try to call mod.main() with a few sensible fallbacks.
    Returns True if a call succeeded (didn't raise), False otherwise.
    """
    if mod is None:
        return False

    main_fn = getattr(mod, "main", None)
    if main_fn is None or not callable(main_fn):
        print(f"[CALL] Module {getattr(mod, '__name__', '<unknown>')} has no callable main().")
        return False

    # Try calling main() with multiple fallbacks
    attempts = [
        ((), {}),
        ((None,), {}),
        (({},), {}),
    ]
    for args, kwargs in attempts:
        try:
            print(f"[CALL] Invoking {mod.__name__}.main{args} {kwargs} ...")
            main_fn(*args, **kwargs)
            print(f"[CALL] {mod.__name__}.main succeeded with args={args}.")
            return True
        except TypeError as te:
            # likely wrong signature; try next fallback
            print(f"[CALL] TypeError calling {mod.__name__}.main with args={args}: {te}")
            continue
        except Exception as e:
            # Unexpected exception from module; show traceback but keep orchestrator running.
            print(f"[ERROR] Exception while running {mod.__name__}.main(): {e}", file=sys.stderr)
            traceback.print_exc()
            return False

    print(f"[CALL] All invocation fallbacks for {mod.__name__}.main() failed (signature mismatch).")
    return False


def run_sequence():
    """
    Ordered list of module basenames to call main() on, in the requested sequence.
    Keep this list minimal and exact as requested.
    """
    sequence = [
        "portfolio_aggregator",   # -> with_whom.portfolio_aggregator.main()
        "state_builder",          # -> with_whom.state_builder.main()
        "counterpart_aggregator", # -> with_whom.counterpart_aggregator.main()
        "metrics",  # -> with_whom.metrics.main()
        "macro_market_data",      # -> with_whom.macro_market_data.main()
        "news_ai",                # -> with_whom.news_ai.main()
        "macro_ai",               # -> with_whom.macro_ai.main()
        "scenario_builder",       # -> with_whom.scenario_builder.main()
    ]

    print("=== Orchestrator Phase 1 (no-args, simple main() invocation sequence) ===")

    for mod_name in sequence:
        print(f"\n--- STEP: {mod_name} ---")
        mod = _try_import(mod_name)
        if not mod:
            print(f"[WARN] Module '{mod_name}' not found under with_whom/ or top-level. Skipping.")
            continue

        ok = _call_main(mod)
        if not ok:
            print(f"[WARN] Calling main() for '{mod_name}' did not succeed. Continuing to next step.")
        else:
            print(f"[OK] Completed '{mod_name}' step.")

    print("\n=== Orchestration finished (attempted all modules) ===")
    # final quick report: list files found in canonical data dir
    print("\n[REPORT] Canonical data dir contents:")
    for p in sorted(DATA_DIR.glob("*.json")):
        print(f" - {p.name}")



if __name__ == "__main__":
    run_sequence()
