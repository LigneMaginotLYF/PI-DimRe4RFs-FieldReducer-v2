#!/usr/bin/env python
"""Train Phase 1 only: generate the dataset."""
import argparse

from src.config_manager import ConfigManager
from src.phase1_dataset import Phase1DatasetGenerator


def main():
    parser = argparse.ArgumentParser(description="Run Phase 1: Dataset Generation")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config")
    args = parser.parse_args()

    cm = ConfigManager(path=args.config)
    p1 = Phase1DatasetGenerator(cm)
    p1.run()
    print("[Done] Phase 1 dataset generated and saved.")


if __name__ == "__main__":
    main()
