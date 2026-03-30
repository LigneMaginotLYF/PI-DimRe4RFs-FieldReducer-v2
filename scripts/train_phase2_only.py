#!/usr/bin/env python3
"""
train_phase2_only.py
====================
Run Phase 2: surrogate training in reduced space + evaluation.

Usage
-----
    python scripts/train_phase2_only.py [--config config.yaml]
"""
import argparse
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config_manager import ConfigManager
from src.phase2_surrogate import Phase2Surrogate
from src.phase2_evaluator import Phase2Evaluator


def main():
    parser = argparse.ArgumentParser(description="Run Phase 2: surrogate training + evaluation")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config")
    args = parser.parse_args()

    cm = ConfigManager(path=args.config)
    cm.warn_if_transient_mode()

    # === PHASE 2 TRAINING ===
    print("[Phase 2] Starting surrogate training in reduced parameter space ...")
    p2 = Phase2Surrogate(cm)
    surrogate = p2.run()
    print("[Phase 2] Training complete.")

    # === PHASE 2 EVALUATION ===
    print("[Phase 2] Starting evaluation ...")
    output_dir = cm.cfg["phase2"]["output_dir"]
    X_test = np.load(os.path.join(output_dir, "phase2_X_test.npy"))
    Y_test = np.load(os.path.join(output_dir, "phase2_Y_test.npy"))

    evaluator = Phase2Evaluator(cm)
    results = evaluator.run(X_test, Y_test, surrogate=surrogate, model_name="surrogate")
    m = results["metrics"]
    print(
        f"[Phase 2] R²={m.get('R2', float('nan')):.4f} | "
        f"RMSE={m.get('RMSE', float('nan')):.4e}"
    )
    print(f"[Phase 2] Results saved to {results['output_dir']}")


if __name__ == "__main__":
    main()
