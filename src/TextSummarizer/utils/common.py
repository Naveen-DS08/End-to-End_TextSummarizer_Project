import os 
from box.exceptions import BoxValueError 
from box import ConfigBox
import yaml
from src.TextSummarizer.logging import logger 
from ensure import ensure_annotations
from pathlib import Path 
from typing import Any 

@ensure_annotations 
def read_yaml(path_to_yaml:Path)-> ConfigBox:
    """
    Read yaml file and returns

    Args: 
        path_to_yaml(str): path

    Raises: 
        valueError: if yaml file is empty
        e: empty file
    Returns:
        ConfigBox: ConfigBox type
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file {path_to_yaml} loaded successfully")
            return ConfigBox(content) 
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e

@ensure_annotations 
def create_directories(path_to_directories: list, verbose=True):
    """
    Create list of directories
    Args: 
        path_to_directories: list of path
        Verbose: lgnore_log(bool, optional): ignore if multiple directories to be created.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"Created Directory at: {path}")