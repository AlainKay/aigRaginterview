#RAG loader using YAML config file
import yaml

# Default path to the pipeline configuration file
CONFIG_PATH = "config/rag_config.yaml"


def load_config():
    """
  load yaml configuration file
    """
    with open(CONFIG_PATH, 'r') as f:
        return yaml.safe_load(f)
