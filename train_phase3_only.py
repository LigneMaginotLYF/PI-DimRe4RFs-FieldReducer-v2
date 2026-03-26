#!/usr/bin/env python
"""Train Phase 3 (dimension reducer) and run evaluation.

Requires a pre-trained Phase 2 surrogate if training_signal='surrogate'.
"""
import argparse
import numpy as np
import os

from src.config_manager import ConfigManager
from src.phase3_reducer import Phase3Reducer
from src.phase3_evaluator import Phase3Evaluator


def main():
    parser = argparse.ArgumentParser(description="Run Phase 3: reducer training + evaluation")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config")
    args = parser.parse_args()

    cm = ConfigManager(path=args.config)

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

    evaluator = Phase3Evaluator(cm)
    results = evaluator.run(X_test, Y_test, reducer=p3_trainer)
    m = results["metrics"]
    print(
        f"[Phase 3] Evaluation complete. "
        f"R²={m.get('R2', float('nan')):.4f} | "
        f"RMSE={m.get('RMSE', float('nan')):.4e}"
    )
    print(f"[Phase 3] Results saved to {results['output_dir']}")


if __name__ == "__main__":
    main()
