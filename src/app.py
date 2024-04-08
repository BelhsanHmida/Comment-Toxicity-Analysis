import streamlit as st
from pytube import YouTube

import pandas as pd 

def get_thumbnail_url(youtube_url):
    yt = YouTube(youtube_url)
    thumbnail_url = yt.thumbnail_url
    return thumbnail_url

def main():
    st.title("Comment Toxicity Classification App")

    # Text input field
    youtube_url = st.text_input("Enter YouTube Video URL:")

    # Submit button
    if st.button("Submit"):
        # Display entered YouTube URL
        st.write("You entered:", youtube_url)

        #

if __name__ == "__main__":
    main()
