from sklearn.preprocessing import MultiLabelBinarizer
import ast

from lyricsgenius import Genius
import pandas as pd
from dotenv import load_dotenv
import os
import openai
from jinja2 import Environment, FileSystemLoader
import time
import re

def get_song_info(artist_name: str, num_of_songs: int):
    """
    Function returns a dataframe of song information from Genius API
    Song information that is returned includes:
        1. Song Title
        2. Artist
        3. Lyrics
    """

    load_dotenv()
    token = os.getenv("GENIUS_API_TOKEN")

    list_ids=[]
    list_lyrics = []
    list_artist = []
    list_song_title = []
    list_description = []

    api = Genius(token, timeout=20).search_artist(artist_name, num_of_songs, sort='popularity')

    songs = api.songs
    for song in songs:

        list_ids.append(song.id)
        list_song_title.append(song.title)
        list_artist.append(song.artist)
        list_lyrics.append(song.lyrics)
        try:
            # Some songs have descriptions, some not
            list_description.append(song.description)
        except:
            list_description.append("No description")

    df = pd.DataFrame({'song_id': list_ids, 'artist': list_artist, 'title': list_song_title,
                       'lyric': list_lyrics, 'description': list_description})

    return df


def classify_song(df: pd.DataFrame):
    load_dotenv()
    template_dir = os.path.dirname(os.path.abspath(__file__))

    env = Environment(loader=FileSystemLoader(template_dir))

    prompt = env.get_template('/prompts/gpt_prompt.j2')

    openai.api_key = os.getenv("OPEN_API_KEY")
    # Prepare a list to store the results
    results = []

    # Process each song in the DataFrame
    for index, row in df.iterrows():
        title = row["title"]
        lyrics = row["lyric"]
        artist = row["artist"]
        print("Classifying emotions for:", title)

        # First GPT call
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt.render(title=title, lyrics=lyrics),
            max_tokens=100
        )
        emotions = response.choices[0].text.strip()

        print("Emotions result: ")
        print(emotions)
        # Append results
        results.append({'title': title, 'artist': artist, 'emotions': emotions})

        # Delay for API rate limits
        time.sleep(20)

    # Create a DataFrame from the results
    results_df = pd.DataFrame(results)

    return results_df


def get_songs_from_edmondsDance(data_path: str, artist_name: list):
   """
   Function returns the subset of songs from EdmondsDance dataset
   The subset of songs are from the artists in the atrist_name list
   :param data_path: Path to EdmondsDance dataset
   :param artist_name: List of artist names
   :return: DataFrame of songs
   """
   df=pd.read_csv(data_path)
   # Create a regular expression pattern from the list for filtering
   pattern = '|'.join(artist_name)

   # Filter the DataFrame for rows where the "Artists" column contains any of the artists in the list
   subset_songs = df[df['Artists'].str.contains(pattern, case=False, na=False)]

   return subset_songs


def main():
    num_of_songs = 5
    atrist_name = ["Drake", "J Cole", "Nicki Minaj","Ice Cube",
                   "Lil Wayne", "Kanye West", "Jay Z", "Eminem","Kendrick Lamar",
                   "Lauren Hill","Mac Miller", "50 Cent", "Nas", "Biggie Smalls", "Tupac"]

    df = pd.DataFrame({'song_id': [], 'artists': [], 'song': [], 'lyrics': [], 'description': []})

    for artist in atrist_name:
        df_artist = get_song_info(artist, num_of_songs)
        df = pd.concat([df, df_artist], ignore_index=True)

    df.to_csv("songs.csv", index=False, header=True, sep=',')




if __name__ == '__main__':

    #Get songs from EdmondsDance dataset
    artist_name_from_edmDS = ["Drake", "J. Cole", "Nicki Minaj", "Kanye West", "Rae Sremmurd", "The Weeknd",
                   "XXXTENTACION", "Kid Cudi", "A$AP Rocky", "Migos", "Cardi B", "Post Malone","Big Sean",
                   "Travis Scott", "Chris Brown"]

    #songs_from_edmDS = get_songs_from_edmondsDance("EdmondsDance/EdmondsDance.csv", artist_name_from_edmDS)
    #songs_from_edmDS.to_csv("edmonds_hip_hop_songs.csv", index=False, header=True, sep=',')

    #Get songs from Genius API
    # Specify artist name to get songs from.
    atrist_name = ["Drake", "J Cole", "Nicki Minaj", "Ice Cube",
                   "Lil Wayne", "Kanye West", "Jay Z", "Eminem", "Kendrick Lamar",
                   "Lauren Hill", "Mac Miller", "50 Cent", "Nas", "Biggie Smalls", "Tupac",
                   "Rihanna", "Beyonce", "Mariah Carey", "Whitney Houston",
                   "Lil Nas X", "Doja Cat", "Usher", "Chris Brown", "The Weeknd",
                   "Missy Elliott", "Post Malone", "Pitbull", "Frank Ocean"]

    #df = pd.DataFrame({'song_id': [], 'artists': [], 'song': [], 'lyrics': [], 'description': []})

    #for artist in atrist_name:
        #df_artist = get_song_info(artist, num_of_songs=10)
       # df = pd.concat([df, df_artist], ignore_index=True)

    #df.to_csv("raw_songs.csv", index=False, header=True, sep=',')

    #df = pd.read_csv("raw_songs.csv")

    # Assing emotions using GPT-3 API
    #emotion_df = classify_song(df)

    #emotion_df.to_csv("emotions_raw.csv", index=False, header=True, sep=',')

    emotion_df = pd.read_csv("emotions_raw.csv")
    emotion_df['emotions'] = emotion_df['emotions'].str.extract(r"\[(.*?)\]").applymap(lambda x: x.lower() if isinstance(x, str) else x)

    print(emotion_df["emotions"][0])
    print(emotion_df.shape)






