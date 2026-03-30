#!/usr/bin/env python3
"""
train_phase3_only.py
====================
Run Phase 3: dimension reducer training + evaluation.

Requires a pre-trained Phase 2 surrogate if training_signal='surrogate'.

Usage
-----
    python scripts/train_phase3_only.py [--config config.yaml]
"""
import argparse
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config_manager import ConfigManager
from src.phase3_reducer import Phase3Reducer
from src.phase3_evaluator import Phase3Evaluator


def main():
    parser = argparse.ArgumentParser(description="Run Phase 3: reducer training + evaluation")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config")
    args = parser.parse_args()

    cm = ConfigManager(path=args.config)
    cm.warn_if_transient_mode()

    # === PHASE 3 TRAINING ===
    print("[Phase 3] Starting reducer training ...")
    p3_trainer = Phase3Reducer(cm)
    p3_trainer.run()
    print("[Phase 3] Training complete.")

    # === PHASE 3 EVALUATION ===
    print("[Phase 3] Starting evaluation ...")
    output_dir = cm.cfg["phase3"]["output_dir"]
    X_test = np.load(os.path.join(output_dir, "phase3_X_test_full.npy"))
    Y_test = np.load(os.path.join(output_dir, "phase3_Y_test_full.npy"))

    # Pass the surrogate (loaded during training) so the settlement comparison
    # plot shows all three curves: GT, Biot-decoded, and surrogate-decoded.
    evaluator = Phase3Evaluator(cm)
    results = evaluator.run(
        X_test, Y_test,
        reducer=p3_trainer,
        surrogate=p3_trainer.surrogate,
    )
    m = results["metrics"]
    print(
        f"[Phase 3] R²={m.get('R2', float('nan')):.4f} | "
        f"RMSE={m.get('RMSE', float('nan')):.4e}"
    )
    print(f"[Phase 3] Results saved to {results['output_dir']}")


if __name__ == "__main__":
    main()
