import sys
import os 
from dataclasses import dataclass
sys.path.append(r'C:\Users\hp\Desktop\Datascience  Projects\CommentToxicity\Comment-Toxicity-Classification')

import pandas as pd
import googleapiclient.discovery
 
from src.logger import logging
from src.utils  import save_objects
from src.exceptions import CustomException

api_key = "AIzaSyBmzmmaxaFwezU2siJ4Lh4ui0mbzkimM3k"
video_id = "Q0CbN8sfihY"

@dataclass
class datascraperConfig:
    scraped_data_path=os.path.join('artifact',"scraped_data.csv")
class datascraping:
    def __init__(self):
        self.scraped_data_config=datascraperConfig()
    def get_scraped_data(self,key,video_id):  
        try:
             

            api_service_name = "youtube"
            api_version = "v3"
            DEVELOPER_KEY = key

            youtube = googleapiclient.discovery.build(
                api_service_name, api_version, developerKey=DEVELOPER_KEY)

            request = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=200
            )
            response = request.execute()

            comments = []

            for item in response['items']:
                comment = item['snippet']['topLevelComment']['snippet']
                comments.append([
                    comment['authorDisplayName'],
                    comment['publishedAt'],
                    comment['updatedAt'],
                    comment['likeCount'],
                    comment['textDisplay']
                ])

            df = pd.DataFrame(comments, columns=['author', 'published_at', 'updated_at', 'like_count', 'text'])
            logging.info("Youtube Data Scraper: Comments extracted successfully.")
            scraper_path=os.path.join('artifact',"scraped_data.csv")

            save_objects(
                            file_path=scraper_path,
                            obj=df
                        )
        except Exception as e :
            raise CustomException(e,sys)
scraper = datascraping()

# Initiate scraping by calling the get_scraped_data method
scraper.get_scraped_data(api_key, video_id)