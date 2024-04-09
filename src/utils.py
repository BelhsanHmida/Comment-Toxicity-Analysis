import sys
import os
from sklearn.metrics import r2_score
from src.exceptions import CustomException
from src.logger import logging
from src.exceptions import CustomException
from src.logger import logging  
from keras.layers import TextVectorization
import numpy as np 
import pandas as pd
from dataclasses import dataclass
import pickle
import tensorflow as tf
import os
import sys
import pickle


import dill

def save_objects(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,'wb') as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e :
        raise CustomException(e,sys)    
    
def load_object(file_path) :
    try:
        with open(file_path,"rb")as file_obj:
            return dill.load(file_obj)
    except Exception as e :
       raise CustomException(e,sys)
    

class TextVectorizer:
    def __init__(self, max_tokens=20, output_sequence_length=1800, output_mode='int'):
        self.max_tokens = max_tokens
        self.output_sequence_length = output_sequence_length
        self.output_mode = output_mode
        self.vectorizer = TextVectorization(max_tokens=max_tokens, 
                                            output_sequence_length=output_sequence_length, 
                                            output_mode=output_mode)
        df = pd.read_csv(r'train.csv')
        df = df.sample(60)
        self.X = df['comment_text']
    def fit(self):
        logging.info("Fitting TextVectorizer to data")
        self.vectorizer.adapt(self.X)
        logging.info("TextVectorizer fitted successfully")

    def save(self, file_path):
        logging.info(f"Saving TextVectorizer to {file_path}")
        config = {
            "max_tokens": self.max_tokens,
            "output_sequence_length": self.output_sequence_length,
            "output_mode": self.output_mode
        }
        with open(file_path, "wb") as f:
            pickle.dump(config, f)
        logging.info("TextVectorizer saved successfully")

    @classmethod
    def load(cls, file_path):
        logging.info(f"Loading TextVectorizer from {file_path}")
        with open(file_path, "rb") as f:
            config = pickle.load(f)
        max_tokens = config["max_tokens"]
        output_sequence_length = config["output_sequence_length"]
        output_mode = config["output_mode"]
        vectorizer = cls(max_tokens=max_tokens, 
                         output_sequence_length=output_sequence_length, 
                         output_mode=output_mode)
        logging.info("TextVectorizer loaded successfully")
        return vectorizer

    def transform(self,Y):
        logging.info("Transforming text data")
        return self.vectorizer(Y)