import json
import os

def load_config(path=None):

    if path is None:
        # Locate project root (one level above this file)
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(project_root, "config", "train_config.json")

    # Load and return JSON config
    with open(path, "r") as f:
        return json.load(f)
