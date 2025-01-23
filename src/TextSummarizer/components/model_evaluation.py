import os 
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer 
from datasets import load_from_disk
import torch 
import pandas as pd 
from tqdm import tqdm 
import evaluate

from src.TextSummarizer.logging import logger 
from src.TextSummarizer.config.configuration import ModelEvaluationConfig

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config 

    def generate_batch_sized_chunks(self, list_of_elements, batch_size):
        "Split the dataset into similar batches that we can process simultaneously"
        for i in range(0, len(list_of_elements), batch_size):
            yield list_of_elements[1: i+batch_size]

    def calculate_metric_on_test_ds(self, dataset, metric, model, 
                                    tokenizer, batch_size=16, 
                                    device = "cuda" if torch.cuda.is_available() else "cpu", 
                                    column_text = "article", column_summary= "highlights"):
        article_batches = list(self.generate_batch_sized_chunks(dataset[column_text], batch_size))
        target_batches = list(self.generate_batch_sized_chunks(dataset[column_summary], batch_size)) 

        for article_batch, target_batch in tqdm(zip(article_batches, target_batches), 
                                                total=len(article_batches)):
            inputs = tokenizer(article_batch, max_length = 1024, truncation=True,
                               padding= "max_length", return_tensors = "pt" )
            
            summaries = model.generate(input_ids = input["input_ids"].to(device),
                                     attention_mask = inputs["attention_mask"].to(device),
                                     length_pennalty = 0.8, num_beams = 8, max_length=128 )
            "Parameter for lenth penalty ensures that the model does not generate sequences that ate too long"

            decoded_summary = [tokenizer.decode(s, skip_special_tokens=True,
                                                clean_up_tokenization_spaces = True) 
                                        for s in summaries]
            decoded_summary = [d.replace("", " ") for d in decoded_summary]

            metric.add_batch(predictions=decoded_summary, references = target_batch)
        
        # Return Rouge scores 
        score = metric.compute()
        return score 
    
    def evaluate(self):
        device = "cuda" if torch.cuda.is_availabel() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
        model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_path).to(device)

        # Loading data 
        dataset_samsum_pt = load_from_disk(self.config.data_path)

        rouge_names = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
        rouge_metric = evaluate.load("rouge")

        score = self.calculate_metric_on_test_ds(dataset_samsum_pt["test"][0:10],
                                                 rouge_metric, model_pegasus, tokenizer,
                                                 batch_size=2, column_test="dialogue", column_summary= "summary")
        
        rouge_dict = {rn: score[rn] for rn in rouge_names}

        df = pd.DataFrame(rouge_dict, index = ["pegasus"])
        df.to_csv(self.config.metric_file_path, index=False)