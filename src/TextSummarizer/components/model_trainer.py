import os 
from src.TextSummarizer.logging import logger
from src.TextSummarizer.entity import ModelTrainerConfig

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import TrainingArguments, Trainer 
from transformers import DataCollatorForSeq2Seq
import torch 
from datasets import load_from_disk

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config 

    def train(self):
        device = "cuda" if torch.cuda.is_availabel() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_ckpt)
        model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_ckpt).to(device)
        seq2seq_data_collector = DataCollatorForSeq2Seq(tokenizer=tokenizer, model= model_pegasus)

        # Loading the data 
        logger.info("Loading the dataset for training")
        dataset_samsum_pt = load_from_disk(self.config.data_path)

        trainer_args = TrainingArguments(
            output_dir=self.config.root_dir,
            num_train_epochs=self.config.num_train_epochs, 
            warmup_steps=self.config.warmup_steps,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            weight_decay=self.config.weight_decay, 
            logging_steps=self.config.logging_steps,
            evaluation_strategy=self.config.evaluation_strategy,
            eval_steps=self.config.eval_steps,
            save_steps=self.config.save_steps,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps
            )
        
        trainer = Trainer(
            model = model_pegasus, args= trainer_args,
            data_collator= seq2seq_data_collector, tokenizer= tokenizer, 
            train_dataset=dataset_samsum_pt["train"],
            eval_dataset= dataset_samsum_pt["validation"]
        )
        logger.info("Training gets started")
        trainer.train()
        logger.info("Training successfully completed")

        # save the model
        model_pegasus.save_pretrained(os.path.join(self.config.root_dir, "pegasus-samsum-model"))
        # save tokenizer 
        tokenizer.save_pretrained(os.path.join(self.config.root_dir, "tokenizer"))
        logger.info("Saved our trained model and tokenizer into local disk")