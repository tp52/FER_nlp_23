import pandas as pd
import re

pd.set_option('display.max_rows', None)

# Function to extract emotions from a string
def extract_emotions(s):
    return re.findall(r"'([^']*)'", s)


def rename_label(df, list_of_labels, new_name):
    def replace_emotions(emotion_str):
        for label in list_of_labels:
            # Split the string into a list, replace, and join back into a string
            emotions_list = emotion_str.split(', ')
            emotions_list = [new_name if e.strip() == label else e for e in emotions_list]
            emotion_str = ', '.join(emotions_list)
        return emotion_str

    df["emotions"] = df["emotions"].apply(replace_emotions)
    return df


if __name__ == '__main__':

    df = pd.read_csv("emotions_raw.csv")
    df['emotions'] = df['emotions'].str.lower()

    old_love_labels = ["affection", "romance"]
    df = rename_label(df, old_love_labels, "love")

    old_sad_labels = ["despair", "pain", "regret"]
    df = rename_label(df, old_sad_labels, "sadness")

    old_joy_labels = ["happiness", "excitement"]
    df = rename_label(df, old_joy_labels, "joy")

    old_anger_labels = ["frustration"]
    df = rename_label(df, old_anger_labels, "anger")

    # Extracting and flattening all emotions
    all_emotions = [emotion for row in df['emotions'] for emotion in extract_emotions(row)]

    # Convert all_emotions to a Pandas Series
    emotions_series = pd.Series(all_emotions)
    # Getting the number of unique emotions
    # Redirecting output to a text file
    with open('emotion_frequencies.txt', 'w') as file:
        print(emotions_series.value_counts(), file=file)


    # Getting the frequency of each emotion
    emotion_frequencies = emotions_series.value_counts()



    # Filter out emotions with less than 5 occurrences
    filtered_emotions = emotion_frequencies[emotion_frequencies >= 5]
    print(filtered_emotions)





