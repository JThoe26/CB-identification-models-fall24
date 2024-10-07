# Import the necessary libraries.
import os
import csv
# package install: 'pip install google-api-python-client'
from googleapiclient.discovery import build

# API key for YouTube Data API v3.
API_KEY = ''

# Function to convert a list of comments to csv.
def comments_to_csv(comments: list, filename: str):
    """
    Convert a list of comments to csv.
    :param comments: List of comments.
    :param filename: File path of the csv.
    """
    dir = os.path.dirname(filename)
    if not os.path.exists(dir):
        print(f"Creating directory: {dir}")
        os.makedirs(dir)
    print(f"Writing {len(comments)} comments to {filename}")
    with open(filename, 'a', newline='') as file:
            writer = csv.writer(file)
            for comment in comments:
                writer.writerow([comment])

# Function to get video ids from a video name query.
def get_video_ids(youtube, query: str, max_results=1):
    """
    Get video ids from a video name query.
    :param youtube: YouTube API object.
    :param query: Query to search for.
    :param max_results: Maximum number of results to return.
    :return: List of video ids.
    """
    search_response = youtube.search().list(
        q=query,
        part="id",
        type="video",
        maxResults=max_results
    ).execute()

    video_ids = [item['id']['videoId'] for item in search_response['items']]
    return video_ids

# Function to get comments by video id.
def get_comments_by_id(youtube, video_id: str, max_results=100):
    """
    Get comments by video id.
    :param youtube: YouTube API object.
    :param video_id: Video id to get comments from.
    :param max_results: Maximum number of results to return.
    :return: List of comments.
    """
    comments = []
    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=100,  # Maximum allowed per request
        order="relevance"
    )
    response = request.execute()

    # While response is not empty, append comments and fetch more if needed on next page.
    while response:
        for item in response['items']:
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            comments.append(comment)
            if len(comments) >= max_results:
                return comments

        if 'nextPageToken' in response:
            request = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=100,
                order="relevance",
                pageToken=response['nextPageToken']
            )
            response = request.execute()
        else:
            break

    return comments

# Function to search comments by video name.
def search_comments_by_name(query: str, max_results=100):
    """
    Search comments by video name.
    :param query: Query to search for.
    :return: List of comments.
    """
    youtube = build('youtube', 'v3', developerKey=API_KEY)

    video_ids = get_video_ids(youtube, query)

    comments = []
    for video_id in video_ids:
        comments.extend(get_comments_by_id(youtube, video_id, max_results))
        for comment in comments:
            print(comment)
    return comments

if __name__ == "__main__":
    # You can search by video name, or video id directly.
    results = search_comments_by_name("Video Name", 10)
    comments_to_csv(results, "./data/pathToCSV.csv")
    
