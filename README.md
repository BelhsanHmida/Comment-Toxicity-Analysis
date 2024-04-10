# 🚀 Youtube Toxicity Analysis App
This project is a YouTube comment analysis app that scrapes comments from a given YouTube video URL, classifies them according to a toxicity model, and provides an analysis of the comments based on their toxicity levels. The toxicity model utilized in this project is a Bidirectional LSTM model, and a vectorizer model is used to vectorize the text data.

## See it live and in action 📺
![Project Picture](https://github.com/BelhsanHmida/Comment-Toxicity-Classification/blob/main/Project%20Picture.PNG?raw=true)

## Features:
- Scrapes comments from a YouTube video URL.
- Classifies comments based on toxicity using a Bidirectional LSTM model.
- Provides analysis and visualization of comment toxicity levels.

# Startup Guide 🚀

1. Clone this Repository on your local machine
2. Create a virtual environment `conda create -n venv python=3.8` 
3. Activate it `conda activate venv`
4. Install initial deps `pip install Requirements.txt`
5. Iniate Model training by `python model_trainer.py`
6. Get Youtube Data Api v3  API_KEY from Google cloud console for free
7. Run the app `python app.py`

## Models:
-Toxicity Model: Bidirectional LSTM model trained to classify comment toxicity levels.
-Vectorizer Model: Used to vectorize the text data for input to the toxicity model.

# Technologies Used:
- Python
- Streamlit
- TensorFlow
- googleapiclient.discovery (for youtube comment scraping)

# Credits :
  This project was developed by Mohamed Hmida.
# License:
  This project is licensed under the MIT License.
