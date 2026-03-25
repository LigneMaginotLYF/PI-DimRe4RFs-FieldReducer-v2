#!/usr/bin/env python
"""Train all 4 phases sequentially."""
import argparse

from src.config_manager import ConfigManager
from src.phase1_dataset import Phase1DatasetGenerator
from src.phase2_surrogate import Phase2Surrogate
from src.phase3_reducer import Phase3Reducer
from src.phase4_validation import Phase4Validator


def main():
    parser = argparse.ArgumentParser(description="Run full 4-phase pipeline")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config")
    args = parser.parse_args()

    cm = ConfigManager(path=args.config)

    # Phase 1
    print("Running Phase 1: Dataset Generation...")
    p1 = Phase1DatasetGenerator(cm)
    p1.run()
    X_train, Y_train, X_val, Y_val = p1.load()

    # Phase 2
    print("Running Phase 2: LUT and Surrogate Training...")
    p2 = Phase2Surrogate(cm)
    p2.run(X_train, Y_train)

    # Phase 3
    print("Running Phase 3: Dimension Reducer Training...")
    p3 = Phase3Reducer(cm)
    p3.run(X_train, Y_train, X_val, Y_val)

    # Phase 4
    print("Running Phase 4: Validation...")
    p4 = Phase4Validator(cm)
    results = p4.run(X_val, Y_val)
    print(f"All phases completed! Metrics: {results['metrics']}")


if __name__ == "__main__":
    main()
