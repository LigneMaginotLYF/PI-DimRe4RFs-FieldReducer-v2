#!/usr/bin/env python
"""Train Phase 2 (surrogate in reduced space) and run evaluation."""
import argparse

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
    import numpy as np
    output_dir = cm.cfg["phase2"]["output_dir"]
    import os
    X_test_path = os.path.join(output_dir, "phase2_X_test.npy")
    Y_test_path = os.path.join(output_dir, "phase2_Y_test.npy")
    X_test = np.load(X_test_path)
    Y_test = np.load(Y_test_path)

    evaluator = Phase2Evaluator(cm)
    results = evaluator.run(X_test, Y_test, surrogate=surrogate, model_name="surrogate")
    m = results["metrics"]
    print(
        f"[Phase 2] Evaluation complete. "
        f"R²={m.get('R2', float('nan')):.4f} | "
        f"RMSE={m.get('RMSE', float('nan')):.4e}"
    )
    print(f"[Phase 2] Results saved to {results['output_dir']}")


if __name__ == "__main__":
    main()
