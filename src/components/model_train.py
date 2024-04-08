import tensorflow as tf
import keras


# Load the model
model = tf.keras.models.load_model(r'C:\Users\hp\Desktop\Datascience  Projects\CommentToxicity\artifact\my_model1.keras')

# Optionally, you can also print a summary of the model
model.summary()