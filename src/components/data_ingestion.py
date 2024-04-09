import sys
import os
sys.path.append(r'C:\Users\hp\Desktop\New folder (3)\Comment-Toxicity-Classification')

from dataclasses import dataclass

import pandas as pd
import tensorflow as tf
from src.utils import save_objects, TextVectorizer
from src.logger import logging
from src.exceptions import CustomException
import pickle

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
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()

    def read_data(self, file_path):
        try:
            logging.info(f"Reading data from {file_path}")
            df = pd.read_csv(file_path)
            logging.info("Data read successfully")
            return df
        except Exception as e:
            logging.error(f"Error reading data from {file_path}: {e}")
            raise CustomException(f"Error reading data from {file_path}", e)
        
    def vectorize_text(self, X):
        logging.info("Performing text vectorization")
        vectorizer = TextVectorizer(max_tokens=20, output_sequence_length=1800, output_mode='int')
        vectorizer.fit(X)
        vectorizer.save(os.path.join("artifact", "vectorizer.pkl"))
        logging.info("Text vectorization completed")
        return vectorizer.transform(X)    
     
    def split_data(self, data):
        logging.info("Splitting data into train, validation, and test sets")
        train_size, val_size, test_size = 0.7, 0.2, 0.1
        num_samples = len(data)
        train_end = int(num_samples * train_size)
        val_end = int(num_samples * (train_size + val_size))
        train_data = data.take(train_end)
        val_data = data.skip(train_end).take(val_end - train_end)
        test_data = data.skip(val_end).take(num_samples - val_end)
        logging.info("Data split completed")
        return train_data, test_data, val_data
    

    def initiate_data_ingestion(self):
        try:
            df = self.read_data(r'C:\Users\hp\Desktop\New folder (2)\train.csv')
            df = df.sample(60)
            X = df['comment_text']
            y = df[df.columns[2:]].values
            vectorized_text = self.vectorize_text(X)
            data = tf.data.Dataset.from_tensor_slices((vectorized_text, y)).cache().shuffle(160000).batch(16).prefetch(8)
            train_data, test_data, val_data = self.split_data(data)
            return train_data, test_data, val_data
        
        except Exception as e:
            logging.error(f"Error occurred during data ingestion: {e}")
            raise CustomException("Error occurred during data ingestion", e)
     
        
if __name__ == "__main__":
    DataIngestion().initiate_data_ingestion()        
    logging.info("Data ingestion completed")