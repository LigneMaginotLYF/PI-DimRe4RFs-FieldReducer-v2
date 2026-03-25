#!/usr/bin/env python
"""Train Phases 1 and 2: dataset generation and surrogate training."""
import argparse

from src.config_manager import ConfigManager
from src.phase1_dataset import Phase1DatasetGenerator
from src.phase2_surrogate import Phase2Surrogate


def main():
    parser = argparse.ArgumentParser(description="Run Phases 1 and 2")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config")
    args = parser.parse_args()

    cm = ConfigManager(path=args.config)

    p1 = Phase1DatasetGenerator(cm)
    p1.run()
    X_train, Y_train, X_val, Y_val = p1.load()

    p2 = Phase2Surrogate(cm)
    p2.run(X_train, Y_train)
    print("[Done] Phase 2 surrogate trained and saved.")


if __name__ == "__main__":
    main()
