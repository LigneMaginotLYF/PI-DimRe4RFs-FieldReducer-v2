#!/usr/bin/env python
"""Run Phase 4 validation only. Requires trained Phase 2 surrogate and Phase 3 reducer."""
import argparse

from src.config_manager import ConfigManager
from src.phase1_dataset import Phase1DatasetGenerator
from src.phase4_validation import Phase4Validator


def main():
    parser = argparse.ArgumentParser(description="Run Phase 4: Validation")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config")
    args = parser.parse_args()

    cm = ConfigManager(path=args.config)
    cm.warn_if_transient_mode()

    # Load validation data from Phase 1
    p1 = Phase1DatasetGenerator(cm)
    _, _, X_val, Y_val = p1.load()

    p4 = Phase4Validator(cm)
    results = p4.run(X_val, Y_val)
    print(f"[Done] Metrics: {results['metrics']}")


if __name__ == "__main__":
    main()
