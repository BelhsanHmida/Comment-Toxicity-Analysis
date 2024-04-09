import sys
sys.path.append(r'C:\Users\hp\Desktop\CommentToxicity\Comment-Toxicity-Classification')
import os
from dataclasses import dataclass

from  keras.layers import TextVectorization
import tensorflow as tf
import numpy as np 
import pandas as pd

from src.utils import save_objects
from src.logger import logging
from src.exceptions import CustomException
# Get the current directory of the script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add the parent directory to the Python path
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)

@dataclass
class DataIngestionConfig:
    train_path: str = os.path.join("artifact", "train.csv")
    test_path: str = os.path.join("artifact", "test.csv")
    val_path: str = os.path.join("artifact", "val.csv")
class DataIngestion:
    """
    This class shall be used to ingest the data
    """
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")

        try:
            df=pd.read_csv(r'C:\Users\hp\Desktop\CommentToxicity\Comment-Toxicity-Classification\artifact\Data.csv')
            logging.info('Read the dataset as dataframe')
            X= df['comment_text']
            y= df[df.columns[2:]].values
            logging.info('Text Vectorization started')
            MAX_FEATURES = 20  
            vectorizer = TextVectorization(max_tokens=MAX_FEATURES,
                              output_sequence_length=1800,
                              output_mode='int') 
            vectorizer.adapt(X.values)
            vectorized_text=vectorizer(X.values)
            trained_vectorizer_file_path=os.path.join("artifact","vectorizer.pkl")
            save_objects(
                file_path= trained_vectorizer_file_path , 
                obj=vectorizer
            )
            logging.info('Text Vectorization finished')
            data=tf.data.Dataset.from_tensor_slices((vectorized_text,y))
            data = data.cache()
            data = data.shuffle(160000)
            data = data.batch(16)
            data = data.prefetch(8)
            batch_X,batch_Y = data.as_numpy_iterator().next()
            train = data.take(int(len(data)*0.7))
            val   = data.skip(int(len(data)*0.7)).take(int(len(data)*0.2))
            test  = data.skip(int(len(data)*0.9)).take(int(len(data)*0.1))
            logging.info('Data split into train, val and test')

            os.makedirs(os.path.dirname(self.data_ingestion_config.train_path),exist_ok=True)
            df.to_csv(self.data_ingestion_config.train_path,index=False,header=True)

            os.makedirs(os.path.dirname(self.data_ingestion_config.test_path),exist_ok=True)
            df.to_csv(self.data_ingestion_config.test_path,index=False,header=True)
            
            os.makedirs(os.path.dirname(self.data_ingestion_config.val_path),exist_ok=True)
            df.to_csv(self.data_ingestion_config.val_path,index=False,header=True)

            logging.info('Data saved successfully')
            return train, test, val
        except Exception as e:
            logging.error("Error occurred while ingesting data")
            raise CustomException("Error occurred while ingesting data", e)    
        
if __name__ == "__main__":
    DataIngestion().initiate_data_ingestion()        