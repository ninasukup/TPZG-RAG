import faiss
import json
import logging
import yaml
from pathlib import Path
from box import ConfigBox
from box.exceptions import BoxValueError


def load_vector_db(index_file, metadata_file):
    try:
        print("Loading FAISS index...")
        index = faiss.read_index(str(index_file))

        metadata = []
        # Read JSONL in UTF-8 (BOM-tolerant) so “weird” bytes don’t blow up on Windows
        with open(str(metadata_file), "r", encoding="utf-8-sig") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    metadata.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise RuntimeError(f"Bad JSON on line {lineno}: {e.msg}") from e

        logging.info("Data Loaded From Vector DB Successfully (FAISS and JSONL)!")
        return index, metadata
    except Exception as e:
        raise RuntimeError(f"Failed to load vector DB: {e}")

    
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """
    Read Yaml file and return ConfigBox type
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logging.info(f"Yaml file: {path_to_yaml} loaded successfully!")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("Yaml file is empty")
    except Exception as e:
        raise e