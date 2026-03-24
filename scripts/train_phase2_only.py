#!/usr/bin/env python3
"""
train_phase2_only.py
====================
Run Phases 1 and 2 only (dataset generation + surrogate training).
The trained surrogate can later be reloaded by Phase 3.

Usage
-----
    python scripts/train_phase2_only.py [--config config.yaml]
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
