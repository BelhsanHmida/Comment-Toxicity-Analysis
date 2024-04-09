import sys
sys.path.append(r'C:\Users\hp\Desktop\CommentToxicity\Comment-Toxicity-Classification')
import streamlit as st
from pytube import YouTube
from src.components.youtube_scraper import datascraping
from src.pipelines.predict_pipeline import CommentToxicityPredictor

def get_thumbnail_url(youtube_url):
    yt = YouTube(youtube_url)
    thumbnail_url = yt.thumbnail_url
    return thumbnail_url
status=0
def main():
    st.title("YouTube Comment Scraper")

    # Text input field for YouTube URL
    youtube_url = st.text_input("Enter YouTube Video URL:")

    # Submit button to display YouTube thumbnail and input field for API key
    
    if 'submit_clicked' not in st.session_state:
        st.session_state.submit_clicked = False

    # Submit button to display YouTube thumbnail
    if st.button("Submit"):
        # Set submit_clicked to True when Submit button is clicked
        st.session_state.submit_clicked = True

    # Display YouTube video thumbnail if Submit button was clicked
    if st.session_state.submit_clicked and not st.button('clear'):
        try:
            thumbnail_url = get_thumbnail_url(youtube_url)
            st.image(thumbnail_url, caption="YouTube Video Thumbnail")
            global status
            status=1
        except Exception as e:
            st.error("Error: Unable to retrieve thumbnail. Please check the URL.")

    if status==1:   # Input field for YouTube API key
        api_key = st.text_input("Enter YouTube API key:")

    # Scrape button to scrape comments
    if status==1 and st.button("Scrape Comments") :
        
        video_id = youtube_url.split('=')[-1]

        # Call the function to scrape comments
        df_comments = datascraping().get_scraped_data(api_key, video_id)

        # Display the scraped comments
        st.subheader("Scraped Comments:")
        st.dataframe(df_comments)

        toxicity = CommentToxicityPredictor().predict_toxicity(df_comments)   
        st.subheader("Comments toxicity Analysis: ")
        st.dataframe(toxicity)


if __name__ == "__main__":
    main()
