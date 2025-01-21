import os 
from urllib import request 
import zipfile 

from src.TextSummarizer.logging import logger
from src.TextSummarizer.config.configuration import DataIngestionConfig 

class DataIngestion:
    def __init__(self, 
                 config:DataIngestionConfig):
        self.config = config 

    def download_file(self):
        if not os.path.exists(self.config.local_data_file):
            filename, headers = request.urlretrieve(
                url = self.config.source_URL,
                filename= self.config.local_data_file
                )
            logger.info("File downloaded successfully")
        else:
            logger.info("File already exists!")

    def extract_zip_file(self):
        """
        Extracts the zip file into data directory 
        function returns None
        """
        unzip_file_path = self.config.unzip_dir
        os.makedirs(unzip_file_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, "r") as raw:
            raw.extractall(unzip_file_path)
        logger.info("File unzipped successfully")