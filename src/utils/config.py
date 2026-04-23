# Shivang Patel
# CS 5330, Final Project: Deepfake Breakfix
# Load YAML config file

import yaml


def load_config(path="configs/default.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)
