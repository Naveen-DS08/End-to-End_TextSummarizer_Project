from src.TextSummarizer.logging import logger
from src.TextSummarizer.pipeline.data_ingestion_pipeline import DataIngestionPipeline
from src.TextSummarizer.pipeline.data_transformation_pipeline import DataTransformationPipeline
from src.TextSummarizer.pipeline.model_trainer_pipeline import ModelTrainerPipeline


STAGE_NAME = "Data Ingestion stage"

try:
    logger.info(f"Stage -> {STAGE_NAME} initiated")
    data_ingestion_pipeline = DataIngestionPipeline()
    data_ingestion_pipeline.initiate_data_ingestion()
    logger.info(f"Stage -> {STAGE_NAME} Completed")
except Exception as e:
    logger.exception(e)
    raise(e)

STAGE_NAME = "Data Transformation stage"

try:
    logger.info(f"Stage -> {STAGE_NAME} initiated")
    data_transformation_pipeline = DataTransformationPipeline()
    data_transformation_pipeline.initiate_data_transformation()
    logger.info(f"Stage -> {STAGE_NAME} Completed")
except Exception as e:
    logger.exception(e)
    raise(e)

STAGE_NAME = "Model Trainer stage"

try:
    logger.info(f"Stage -> {STAGE_NAME} initiated")
    model_trainer_pipeline = ModelTrainerPipeline()
    model_trainer_pipeline.initiate_model_trainer()
    logger.info(f"Stage -> {STAGE_NAME} Completed")
except Exception as e:
    logger.exception(e)
    raise(e)