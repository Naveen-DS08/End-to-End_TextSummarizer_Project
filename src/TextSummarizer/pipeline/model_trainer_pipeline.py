from src.TextSummarizer.logging import logger 
from src.TextSummarizer.config.configuration import ConfigurationManager 
from src.TextSummarizer.components.model_trainer import ModelTrainer

class ModelTrainerPipeline:
    def __init__(self):
        pass 

    def initiate_model_trainer(self):
        config = ConfigurationManager()
        model_trainer_config = config.get_model_trainer_config()
        model_trainer = ModelTrainer(model_trainer_config)
        model_trainer.train()