#!/usr/bin/env python
"""Train Phase 3 only: dimension reducer training. Requires pre-trained Phase 2 surrogate."""
import argparse

from src.config_manager import ConfigManager
from src.phase1_dataset import Phase1DatasetGenerator
from src.phase3_reducer import Phase3Reducer


def main():
    parser = argparse.ArgumentParser(description="Run Phase 3: Dimension Reducer Training")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config")
    args = parser.parse_args()

    cm = ConfigManager(path=args.config)

    # Load Phase-1 data (must already exist)
    p1 = Phase1DatasetGenerator(cm)
    X_train, Y_train, X_val, Y_val = p1.load()

    # Train reducer (loads Phase-2 surrogate internally if training_signal='surrogate')
    p3 = Phase3Reducer(cm)
    p3.run(X_train, Y_train, X_val, Y_val)
    print("[Done] Phase 3 reducer trained and saved.")


if __name__ == "__main__":
    main()
