import os, sys
from wasteDetection.logger import logging
from wasteDetection.exception import AppException
from wasteDetection.entity.config_entity import ModelTrainerConfig
from wasteDetection.entity.artifacts_entity import ModelTrainerArtifact
from ultralytics import YOLO


class ModelTrainer:
    def __init__(self, 
                 model_trainer_config: ModelTrainerConfig):
        self.model_trainer_config = model_trainer_config
        
        
    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        logging.info("Entered intiate_model_trainer method of ModelTrainer class")
        
        try:
            # Unzipping and preparing data
            logging.info("Unzipping data")
            os.system("unzip data.zip -d yolov8n_train")
            os.system("rm data.zip")
            
            # Ensure the training directory exists
            os.makedirs("yolov8n_train", exist_ok=True)
            
            # Path to the data.yaml which should be relative or configured externally
            data_yaml_path = os.path.abspath("yolov8n_train/data.yaml")
            
            # Running the training process
            os.system(f"cd yolov8n_train && yolo task=detect mode=train \
                    model={self.model_trainer_config.weight_name} \
                    imgsz=640 \
                    batch={self.model_trainer_config.batch_size} \
                    epochs={self.model_trainer_config.no_epochs} \
                    data={data_yaml_path} \
                    name='yolov8n_results'")
            
            # Path for saving the best model
            best_model_path = "yolov8n_train/runs/detect/yolov8n_results/weights/best.pt"
            os.system(f"cp {best_model_path} yolov8n_train/")
            
            # Ensure the model trainer directory exists and copy the best model
            os.makedirs(self.model_trainer_config.model_trainer_dir, exist_ok=True)
            os.system(f"cp {best_model_path} {self.model_trainer_config.model_trainer_dir}/")
            
            # Cleanup to remove unnecessary files and directories
            os.system(f"rm -rf yolov8n_train/{self.model_trainer_config.weight_name}")
            os.system("rm -rf yolov8n_train/runs") 
            os.system("rm -rf yolov8n_train/train")
            os.system("rm -rf yolov8n_train/test")
            os.system("rm -rf yolov8n_train/valid")
            os.system("rm -rf yolov8n_train/data.yaml")
            
            # Creating artifact object for the trained model
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=os.path.join(self.model_trainer_config.model_trainer_dir, "best.pt")
            )
            
            logging.info("Exited initiate_model_trainer method of ModelTrainer class")
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            
            return model_trainer_artifact
        
        except Exception as e:
            raise AppException(e, sys)