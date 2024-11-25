import os
import json
import logging
from pathlib import Path
from src.data_types import (SearchResult, 
                            Parameters)


def get_project_root() -> Path:
    """"""
    return Path(__file__).parent.parent

empty_result = SearchResult(templateId=0, templateText="").dict()

PROJECT_ROOT_DIR = get_project_root()

with open(os.path.join(PROJECT_ROOT_DIR, "data", "config.json"), "r") as jf:
    config_dict = json.load(jf)

parameters = Parameters.parse_obj(config_dict)


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S', )

logger = logging.getLogger()
logger.setLevel(logging.INFO)