import streamlit as st
from pytube import YouTube

 
youtube_url = "https://www.youtube.com/watch?v=9bZkp7q19f0"
def get_thumbnail_url(youtube_url):
    yt = YouTube(youtube_url)
    thumbnail_url = yt.thumbnail_url
    return thumbnail_url    
thumbnail_url = get_thumbnail_url(youtube_url)
print(
    'done'
)