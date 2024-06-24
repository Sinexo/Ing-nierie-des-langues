import pandas as pd
import json
from sklearn.preprocessing import MultiLabelBinarizer
from langdetect import detect, LangDetectException
import re

def clean_bbcode(raw_text):
    # Suppressionles balises BBCode
    cleaned_text = re.sub(r'\[\/*\w+.*?\]', '', raw_text)
    cleaned_text = ' '.join(cleaned_text.split())
    return cleaned_text

# Fonction pour filtrer les textes basés sur la langue, acceptant uniquement l'anglais
#Car pour une raison inconnu j'avais des champs en russe dans le csv
def filter_by_language(texts):
    filtered_texts = []
    for text in texts:
        try:
            if detect(text) == 'en':
                filtered_texts.append(text)
        except LangDetectException:
            pass
    return filtered_texts

with open('steam_data.json', 'r', encoding='utf-8') as file:
    json_data = json.load(file)

games_df = pd.json_normalize(json_data, sep='_')

mlb_genres = MultiLabelBinarizer()
# mlb_platforms = MultiLabelBinarizer()

games_df['genres_list'] = games_df['genres'].apply(lambda x: filter_by_language([genre['description'] for genre in x]))
# games_df['platforms_list'] = games_df['platforms'].apply(lambda x: x)

df_genres = pd.DataFrame(mlb_genres.fit_transform(games_df['genres_list']),
                         columns=mlb_genres.classes_,
                         index=games_df.index)
# df_platforms = pd.DataFrame(mlb_platforms.fit_transform(games_df['platforms_list']),
#                             columns=mlb_platforms.classes_,
#                             index=games_df.index)

games_df['developers'] = games_df['developers'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)
games_df['publishers'] = games_df['publishers'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)

full_games_df = pd.concat([games_df, df_genres], axis=1)
full_games_df.drop(['genres', 'genres_list'], axis=1, inplace=True)

reviews_df = pd.DataFrame()

for index, game_row in full_games_df.iterrows():
    if 'english_reviews' in game_row and isinstance(game_row['english_reviews'], list):
        temp_df = pd.json_normalize(game_row['english_reviews'])
        if 'review' in temp_df.columns:
            temp_df['review'] = temp_df['review'].apply(clean_bbcode)
        temp_df['language'] = 'English'
        for col in full_games_df.columns.difference(['english_reviews', 'french_reviews']):
            temp_df[col] = [game_row[col]] * len(temp_df)
        reviews_df = pd.concat([reviews_df, temp_df], ignore_index=True)


columns_to_drop = [
    'timestamp_created', 'timestamp_updated', 'recommendationid', 'steam_purchase',
    'received_for_free', 'written_during_early_access', 'hidden_in_steam_china', 'steam_china_location',
    'author.num_games_owned', 'author.num_reviews', 'author.playtime_forever',
    'author.playtime_last_two_weeks', 'author.playtime_at_review', 'author.last_played', 'author.steamid',
    'comment_count','timestamp_dev_responded','developer_response','voted_up','votes_up','votes_funny'
]
reviews_df.drop(columns=columns_to_drop, inplace=True)

reviews_df.to_csv('steam_data.csv', index=False, encoding='utf-8')
