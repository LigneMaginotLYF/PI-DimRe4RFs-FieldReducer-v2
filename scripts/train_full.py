#!/usr/bin/env python3
"""
train_full.py
=============
Run all 4 phases sequentially:
  Phase 1: Generate dataset
  Phase 2: Train surrogate on LUT
  Phase 3: Train dimension reducer
  Phase 4: Validate and visualise

Usage
-----
    python scripts/train_full.py [--config config.yaml]
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
    p1 = Phase1DatasetGenerator(cm)
    p1.run()
    X_train, Y_train, X_val, Y_val = p1.load()

    # Phase 2
    p2 = Phase2Surrogate(cm)
    p2.run(X_train, Y_train)

    # Phase 3
    p3 = Phase3Reducer(cm)
    p3.run(X_train, Y_train, X_val, Y_val)

    # Phase 4
    p4 = Phase4Validator(cm)
    results = p4.run(X_val, Y_val)
    print(f"[Done] Metrics: {results['metrics']}")


if __name__ == "__main__":
    main()
