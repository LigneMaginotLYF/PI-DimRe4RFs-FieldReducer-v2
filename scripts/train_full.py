#!/usr/bin/env python3
"""
train_full.py
=============
Run Phase 2 (surrogate) + Phase 3 (reducer) sequentially with evaluation.

Usage
-----
    python scripts/train_full.py [--config config.yaml]
"""
import argparse
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config_manager import ConfigManager
from src.phase2_surrogate import Phase2Surrogate
from src.phase2_evaluator import Phase2Evaluator
from src.phase3_reducer import Phase3Reducer
from src.phase3_evaluator import Phase3Evaluator


def main():
    parser = argparse.ArgumentParser(description="Run Phase 2 + Phase 3 pipeline")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config")
    args = parser.parse_args()

    cm = ConfigManager(path=args.config)
    cm.warn_if_transient_mode()
    cfg = cm.cfg

    # ===================================================================
    print("=" * 60)
    print("[Phase 2] Starting surrogate training ...")
    p2_trainer = Phase2Surrogate(cm)
    surrogate = p2_trainer.run()
    print("[Phase 2] Training complete.")

    print("[Phase 2] Starting evaluation ...")
    output_dir_p2 = cfg["phase2"]["output_dir"]
    X_test_p2 = np.load(os.path.join(output_dir_p2, "phase2_X_test.npy"))
    Y_test_p2 = np.load(os.path.join(output_dir_p2, "phase2_Y_test.npy"))
    p2_evaluator = Phase2Evaluator(cm)
    p2_results = p2_evaluator.run(X_test_p2, Y_test_p2, surrogate=surrogate)
    p2_m = p2_results["metrics"]
    print(f"[Phase 2] R²={p2_m.get('R2', float('nan')):.4f} | RMSE={p2_m.get('RMSE', float('nan')):.4e}")
    print(f"[Phase 2] Results saved to {p2_results['output_dir']}")

    # ===================================================================
    print("=" * 60)
    print("[Phase 3] Starting reducer training ...")
    p3_trainer = Phase3Reducer(cm)
    p3_trainer.run()
    print("[Phase 3] Training complete.")

    print("[Phase 3] Starting evaluation ...")
    output_dir_p3 = cfg["phase3"]["output_dir"]
    X_test_p3 = np.load(os.path.join(output_dir_p3, "phase3_X_test_full.npy"))
    Y_test_p3 = np.load(os.path.join(output_dir_p3, "phase3_Y_test_full.npy"))
    p3_evaluator = Phase3Evaluator(cm)
    p3_results = p3_evaluator.run(X_test_p3, Y_test_p3, reducer=p3_trainer, surrogate=surrogate)
    p3_m = p3_results["metrics"]
    print(f"[Phase 3] R²={p3_m.get('R2', float('nan')):.4f} | RMSE={p3_m.get('RMSE', float('nan')):.4e}")
    print(f"[Phase 3] Results saved to {p3_results['output_dir']}")

    print("=" * 60)
    print("[Done] Full pipeline completed.")


if __name__ == "__main__":
    main()
