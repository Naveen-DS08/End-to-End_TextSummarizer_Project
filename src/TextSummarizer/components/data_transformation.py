import os 
from src.TextSummarizer.logging import logger 
from src.TextSummarizer.config.configuration import DataTransformationConfig

from transformers import AutoTokenizer 
from datasets import load_from_disk

class DataTransformation:
    def __init__(self,
                 config:DataTransformationConfig):
        self.config = config 
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)

    def convert_examples_to_features(self, example_batch):
        input_encoding = self.tokenizer(example_batch["dialogue"], max_length=1024, truncation=True)

        with self.tokenizer.as_target_tokenizer():
            target_encoding = self.tokenizer(example_batch["summary"], max_length=1024, truncation=True)

        return {
            "input_ids": input_encoding["input_ids"],
            "attention_mask": input_encoding["attention_mask"],
            "labels": target_encoding["input_ids"]
            }
    
    def convert(self):
        dataset_samsum = load_from_disk(self.config.data_path)
        logger.info("Transforming data")
        dataset_samsum_pt = dataset_samsum.map(self.convert_examples_to_features, batched=True)
        # save to disk 
        dataset_samsum_pt.save_to_disk(os.path.join(self.config.root_dir, "samsum_dataset"))
        logger.info("Transformed dataset saved into disk successfully")