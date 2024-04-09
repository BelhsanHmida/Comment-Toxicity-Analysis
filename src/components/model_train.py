import sys
sys.path.append(r'Comment-Toxicity-Classification')
import os
import tensorflow as tf
from dataclasses import dataclass

from keras.models import Sequential
from keras.layers import LSTM, Dropout, Bidirectional, Dense, Embedding
from keras.layers import Dropout, BatchNormalization

from src.logger import logging
from src.exceptions import CustomException
from src.utils import save_objects
from src.components.data_ingestion import DataIngestion
@dataclass
class modeltrainerconfig:
    trained_model_path=os.path.join('artifact','model.h5')
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = modeltrainerconfig()
    def initiate_model_trainer(self,train, val, test):    

        MAX_FEATURES = 2000000 
        model = Sequential()
        # Create the embedding layer
        model.add(Embedding(MAX_FEATURES + 1, 32))

        # Add dropout after the embedding layer
        model.add(Dropout(0.2))  # Adjust dropout rate as needed

        model.add(Bidirectional(LSTM(32, activation='tanh')))

        # Add dropout after the LSTM layer
        model.add(Dropout(0.2))  # Adjust dropout rate as needed

        model.add(Dense(128, activation='relu'))
        model.add(BatchNormalization())  # Add batch normalization

        model.add(Dense(256, activation='relu'))
        model.add(Dense(128, activation='relu'))

        # Add dropout before the final output layer
        model.add(Dropout(0.5))  # Adjust dropout rate as needed

        # Final Layer maps to output 
        model.add(Dense(6, activation='sigmoid'))
        self.model=model
        def ModelCompile():
            self.model.compile(optimizer='adam', loss='binary_crossentropy')
            
        def ModelFit():
            self.model.fit(train, epochs=1, validation_data=val)
            
        model.save(self.model_trainer_config.trained_model_path, save_format="h5")
        logging.info(f"Model saved at {self.model_trainer_config.trained_model_path}")

if __name__ =='__main__':
    train, test, val = DataIngestion().initiate_data_ingestion()
    model_trainer = ModelTrainer()
    model_trainer.initiate_model_trainer(train, val, test)