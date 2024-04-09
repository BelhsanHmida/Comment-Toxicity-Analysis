import sys
sys.path.append(r'C:\Users\hp\Desktop\New folder (3)\Comment-Toxicity-Classification')
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
class ModelTrainerConfig:
    trained_model_path=os.path.join('artifact','model.h5')
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    def initiate_model_trainer(self):    
        logging.info("Initiating model training")
        MAX_FEATURES = 20

        self.model = Sequential()
    
        self.model.add(Embedding(MAX_FEATURES + 1, 32))
        self.model.add(Dropout(0.2))
        self.model.add(Bidirectional(LSTM(32, activation='tanh')))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(128, activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(6, activation='sigmoid'))
        
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        logging.info("Model compiled successfully")

        self.model_summary = self.model.summary()
        logging.info(f"Model Summary: {self.model_summary} ")
        

    def fit_Model(self,train,val,test):
        logging.info("Fitting model")
        self.history = self.model.fit(train, epochs=3, validation_data=val)
        test_loss = self.history.history['loss']
        test_accuracy = self.history.history['accuracy']
        logging.info(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

    def save_model(self): 
        logging.info(f"Saving model at {self.model_trainer_config.trained_model_path}")   
        self.model.save(self.model_trainer_config.trained_model_path, save_format="h5")
        logging.info(f"Model saved at {self.model_trainer_config.trained_model_path}")

def main():
    train, test, val = DataIngestion().initiate_data_ingestion()
    model_trainer_config = ModelTrainerConfig()
    model_trainer = ModelTrainer()
    model_trainer.initiate_model_trainer()
    model_trainer.fit_Model(train, val, test)
    model_trainer.save_model()

if __name__ =='__main__':
    main()
    logging.info("Model training completed successfully")
