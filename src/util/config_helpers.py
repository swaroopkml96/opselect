import os
import yaml

def load_config():
    with open(os.path.join("configs", "config.yaml"), 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg