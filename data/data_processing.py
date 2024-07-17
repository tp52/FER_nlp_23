import ast

import pandas as pd
import re


def clean_lyrics(df: pd.DataFrame):
    """
    Function cleans lyrics by removing words in brackets such as: Intro, corus, etc
    :param df: DataFrame of songs
    Returns DataFrame of cleaned lyrics
    """

    df["lyric"] = df["lyric"].str.lower()
    df["lyric"] = df["lyric"].str.replace(r"\[.*?\]", "", regex=True)
    df["lyric"] = df["lyric"].str.replace(r"^\d+ contributors.*\n", "", regex=True)
    df["lyric"] = df["lyric"].str.replace("\n", " ")
    df["lyric"] = df["lyric"].apply(lambda x: re.sub(r'\d+embed$', '', x))
    df["lyric"] = df["lyric"].str.strip()

    return df

def safe_literal_eval(x):
    try:
        return ast.literal_eval(x) if pd.notnull(x) else x
    except (SyntaxError, ValueError):
        return None
def emotion_transformation(df_emotions: pd.DataFrame):
    """
    Function transforms the emotions column into a one-hot encoded DataFrame
    :param df - original DataFrame with emotions column:
    :return df_aggregated - DataFrame with one-hot encoded emotions:
    """
    # Step 1: Split the emotions and explode the DataFrame


    # Correct the format of the emotions column
    df_emotions['emotions'] = df_emotions['emotions'].apply(safe_literal_eval)

    # Drop rows where emotions couldn't be evaluated
    df = df_emotions.dropna(subset=['emotions'])

    # Perform One-Hot Encoding
    ohe_df = pd.get_dummies(df['emotions'].apply(pd.Series).stack()).groupby(level=0).sum()

    # Concatenate the One-Hot Encoded DataFrame with the original DataFrame
    df = pd.concat([df, ohe_df], axis=1)

    # Drop the original "emotions" column
    df = df.drop('emotions', axis=1)

    #df_aggregated = df_aggregated.drop('emotions_', axis=1)
    print(df.columns)

    return df

if __name__ == '__main__':

    df_emotions = pd.read_csv("emotions_raw.csv")
    df_emotions["emotions"] = df_emotions["emotions"].str.lower()
    df_songs = pd.read_csv("raw_songs.csv")

    df_songs.drop(columns=['description','artists','song','lyrics'], inplace=True)

    df_songs = clean_lyrics(df_songs)

    # Merge the "emotion" column from the emotions DataFrame into the original DataFrame
    df = df_songs.merge(df_emotions, on=['title', 'artist'], how='right')
    #df.to_csv("songs_emotion.csv", index=False, header=True, sep=',')

    #Perform one-hot encoding on the emotions column
    df = emotion_transformation(df)

    # List of emotion columns
    emotion_columns = ['anger', 'confidence', 'desire', 'disgust', 'gratitude', 'joy', 'love', 'lust',
                       'sadness', 'shame']

    # Calculate the total count of each emotion across the entire DataFrame
    emotion_counts = df[emotion_columns].sum()

    # Display the emotion counts
    print(emotion_counts)

    df.to_csv("final-songs.csv", index=False, header=True, sep=',')
