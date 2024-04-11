import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import streamlit as st
from pytube import YouTube
from src.components.youtube_scraper import datascraping
from src.pipelines.predict_pipeline import CommentToxicityPredictor
import plotly.graph_objects as go

def get_thumbnail_url(youtube_url):
    yt = YouTube(youtube_url)
    thumbnail_url = yt.thumbnail_url
    return thumbnail_url
status=0

def create_pie_plot(data, category):
    toxic_color = '#ff0000'      # Red color
    non_toxic_color = '#00ff00'  # Green color

    # Count the number of toxic and non-toxic comments
    num_toxic = data[category].sum()
    num_non_toxic = len(data) - num_toxic

    # Create the pie chart
    fig = go.Figure(go.Pie(
        labels=['Non-toxic', 'Toxic'],
        values=[num_non_toxic, num_toxic],
        hole=0.6,
        marker=dict(colors=[non_toxic_color, toxic_color]),
        textinfo='label+percent',
    ))

    # Update the layout
    fig.update_layout(
        title=f'Distribution of {category} Comments',
        margin=dict(l=10, r=10, t=50, b=50),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )

    return fig
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
        
        model_path = r"Comment-Toxicity-Classification\artifact\model.h5"
        vectorizer_path = r"Comment-Toxicity-Classification\artifact\vectorizer.pkl"
        toxicity = CommentToxicityPredictor(model_path,vectorizer_path)
        toxicity.load_model_and_vectorizer()
        toxicity =toxicity .predict_toxicity(df_comments)   

        st.subheader("Comments toxicity Analysis: ")
        st.dataframe(toxicity)
        
        # Plotting pie charts for each toxicity category
        for category in toxicity.columns[1:]:
            st.plotly_chart(create_pie_plot(toxicity, category))


if __name__ == "__main__":
    main()
