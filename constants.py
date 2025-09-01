from pathlib import Path
from dotenv import load_dotenv
import os

# load_dotenv()
# API_KEYS = os.getenv('OPENAI_API_KEY')

# CONFIG_FILE_PATH = Path("config/config.yaml")
BASE_DIR = Path(__file__).parent

CONFIG_FILE_PATH = BASE_DIR / "config" / "config.yaml"

CONFIG_JSON_PATH = BASE_DIR / "config" / "config.json"

VECTORSTORE_PATH = BASE_DIR / "Neu_Knowledgebase" / "faiss_index.bin"

METADATA_PATH = BASE_DIR / "Neu_Knowledgebase" / "metadata.jsonl"