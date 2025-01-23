from src.TextSummarizer.logging import logger 
from src.TextSummarizer.components.model_evaluation import ModelEvaluation 
from src.TextSummarizer.config.configuration import ConfigurationManager

class ModelEvaluationPipeline:
    def __init__(self):
        pass 

    def initiate_model_evaluation_pipeline(self):
        config = ConfigurationManager()
        model_evaluation_config = config.get_model_evaluation_config()
        model_evaluation = ModelEvaluation(model_evaluation_config)
        model_evaluation.evaluate()
