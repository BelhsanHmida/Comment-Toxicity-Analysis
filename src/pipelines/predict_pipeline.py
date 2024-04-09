import sys
#sys.path.append(r'C:\Users\hp\Desktop\New folder (3)\Comment-Toxicity-Classification')
import os
import pickle
import tensorflow as tf
import pandas as pd  
import numpy as np
from src.logger import logging
from src.exceptions import CustomException
from src.utils import TextVectorizer
class CommentToxicityPredictor:
    def __init__(self, model_path, vectorizer_path):
        self.model_path = model_path
        self.vectorizer_config_path = vectorizer_path
        self.model = None
        self.vectorizer = None

    def load_model_and_vectorizer(self):
        try:
            logging.info(f"Loading model from {self.model_path}")
            self.model = tf.keras.models.load_model(self.model_path)
             # Load the vectorizer configuration
            with open(self.vectorizer_config_path, "rb") as f:
                vectorizer_config = pickle.load(f)
            
            logging.info("Recreating the TextVectorization instance")
            self.vectorizer = TextVectorizer(
                     max_tokens=vectorizer_config["max_tokens"],
                     output_sequence_length=vectorizer_config["output_sequence_length"],
                     output_mode=vectorizer_config["output_mode"],
                      )
            self.vectorizer.fit()
            logging.info("Model and vectorizer loaded successfully")
        except Exception as e:  
            logging.error(f"Error loading model and vectorizer: {e}")
            raise CustomException("Error loading model and vectorizer", e)
            
    def predict_toxicity(self, df_comments):
        logging.info("Predicting toxicity of comments")
        try:
            if not self.model or not self.vectorizer:
                raise ValueError("Model and vectorizer must be loaded before prediction")
            
             # Transform the comments
            logging.info("Transforming comments using vectorizer")
            
            X = self.vectorizer.transform(df_comments['comment_text'].values)
            logging.info("Comment vectorization transformation successful")
            y_pred = self.model.predict(X)
            columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
            y_pred_df = pd.DataFrame(y_pred, columns=columns)
            return y_pred_df
        except Exception as e:  
            logging.error(f"Error predicting toxicity: {e}")
            raise CustomException("Error predicting toxicity", e)
       

if __name__ == "__main__":
    model_path = "artifact/model.h5"
    vectorizer_path = "artifact/vectorizer.pkl"

    # Sample DataFrame for testing
    df_comments = pd.DataFrame({"comment_text": ["This is a toxic comment", "This is not toxic"]})

    predictor = CommentToxicityPredictor(model_path, vectorizer_path)
    predictor.load_model_and_vectorizer()
    predictions = predictor.predict_toxicity(df_comments)
    print(predictions)