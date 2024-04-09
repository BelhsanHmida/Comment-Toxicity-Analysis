#After we trained our model from the model_train we can now mae predictions using the predict_pipeline.py
import sys
sys.path.append(r'Comment-Toxicity-Classification')
import os
import pickle
import tensorflow as tf
import pandas as pd  # Example library, import any other necessary libraries for your prediction pipeline
import numpy as np
class CommentToxicityPredictor:
  
    def predict_toxicity(self,df_comments):
        #model_path = "artifact/model.h5"  # Update with the path to your saved model file
        #self.model = tf.keras.models.load_model(model_path)
        
        #vectorizer_file = "Comment-Toxicity-Classification\artifact\vectorizer.pkl"
        #with open(vectorizer_file, "rb") as f:
         #   vectorizer = pickle.load(f)
 
        #self.X = vectorizer(df_comments['comment_text'].values)
        #y_pred = self.model.predict(self.X)
        y_pred = pd.DataFrame(np.random.rand(100, 6), columns=['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'])

        return y_pred
    
