import pandas as pd
import json
from sklearn.preprocessing import MultiLabelBinarizer
from langdetect import detect, LangDetectException

# Fonction pour filtrer les textes basés sur la langue, acceptant uniquement l'anglais et le français
def filter_by_language(texts):
    filtered_texts = []
    for text in texts:
        try:
            # Détecter la langue et filtrer
            if detect(text) in ['en', 'fr']:
                filtered_texts.append(text)
        except LangDetectException:
            pass  # Passer si la langue du texte ne peut pas être détectée
    return filtered_texts

# Charger les données depuis le fichier JSON
with open('steam_data.json', 'r',encoding='utf-8') as file:
    json_data = json.load(file)

# Transformer le JSON en DataFrame pour la manipulation
games_df = pd.json_normalize(json_data, sep='_')

# Création de colonnes one-hot pour les genres, catégories, et plateformes
mlb_genres = MultiLabelBinarizer()
mlb_categories = MultiLabelBinarizer()
mlb_platforms = MultiLabelBinarizer()

# Filtrage des genres et catégories
games_df['genres_list'] = games_df['genres'].apply(lambda x: filter_by_language([genre['description'] for genre in x]))
games_df['categories_list'] = games_df['categories'].apply(lambda x: filter_by_language([cat['description'] for cat in x]))
games_df['platforms_list'] = games_df['platforms'].apply(lambda x: x)  # Assumant que 'platforms' est déjà une liste

# Appliquer le MultiLabelBinarizer
df_genres = pd.DataFrame(mlb_genres.fit_transform(games_df['genres_list']),
                         columns=mlb_genres.classes_,
                         index=games_df.index)
df_categories = pd.DataFrame(mlb_categories.fit_transform(games_df['categories_list']),
                             columns=mlb_categories.classes_,
                             index=games_df.index)
df_platforms = pd.DataFrame(mlb_platforms.fit_transform(games_df['platforms_list']),
                            columns=mlb_platforms.classes_,
                            index=games_df.index)

# Nettoyage des champs 'developers', 'publishers' et 'price'
games_df['developers'] = games_df['developers'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)
games_df['publishers'] = games_df['publishers'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)
games_df['price'] = games_df['price'].replace('[$,€£]', '', regex=True)

# Fusionner le DataFrame original avec les nouvelles colonnes one-hot
full_games_df = pd.concat([games_df, df_genres, df_categories, df_platforms], axis=1)
full_games_df.drop(['genres', 'categories', 'platforms', 'genres_list', 'categories_list', 'platforms_list'], axis=1, inplace=True)

# Créer un DataFrame pour les avis avec les données du jeu associées
reviews_df = pd.DataFrame()

# Extrait et expand les avis en ligne individuelles
for index, game_row in full_games_df.iterrows():
    for review_type in ['english_reviews', 'french_reviews']:
        temp_df = pd.json_normalize(game_row[review_type])
        temp_df['language'] = review_type.split('_')[0]  # English or French
        for col in full_games_df.columns.difference(['english_reviews', 'french_reviews']):
            temp_df[col] = [game_row[col]] * len(temp_df)
        reviews_df = pd.concat([reviews_df, temp_df], ignore_index=True)

# Supprimer les colonnes non désirées des avis
columns_to_drop = [
    'timestamp_created', 'timestamp_updated', 'recommendationid', 'steam_purchase',
    'received_for_free', 'written_during_early_access', 'hidden_in_steam_china', 'steam_china_location',
    'author.num_games_owned', 'author.num_reviews', 'author.playtime_forever',
    'author.playtime_last_two_weeks', 'author.playtime_at_review', 'author.last_played', 'author.steamid',
    'comment_count','timestamp_dev_responded','developer_response','languages', 'voted_up','votes_up','votes_funny'
]
reviews_df.drop(columns=columns_to_drop, inplace=True)

# Exporter les données complètes en CSV avec encodage UTF-8
reviews_df.to_csv('steam_data.csv', index=False, encoding='utf-8-sig')