import pandas as pd
import json
from sklearn.preprocessing import MultiLabelBinarizer

with open('steam_games_data.json', 'r') as file:
    json_data = json.load(file)


# Transformer le JSON en DataFrame pour la manipulation
games_df = pd.json_normalize(json_data, sep='_')

# Création de colonnes one-hot pour les genres et catégories
mlb_genres = MultiLabelBinarizer()
mlb_categories = MultiLabelBinarizer()

games_df['genres_list'] = games_df['genres'].apply(lambda x: [genre['description'] for genre in x])
games_df['categories_list'] = games_df['categories'].apply(lambda x: [cat['description'] for cat in x])

df_genres = pd.DataFrame(mlb_genres.fit_transform(games_df['genres_list']),
                         columns=mlb_genres.classes_,
                         index=games_df.index)
df_categories = pd.DataFrame(mlb_categories.fit_transform(games_df['categories_list']),
                             columns=mlb_categories.classes_,
                             index=games_df.index)

# Fusionner le DataFrame original avec les nouvelles colonnes one-hot
full_games_df = pd.concat([games_df, df_genres, df_categories], axis=1)
full_games_df.drop(['genres', 'categories', 'genres_list', 'categories_list'], axis=1, inplace=True)

# Créer un DataFrame pour les avis avec les données du jeu associées
reviews_df = pd.DataFrame()

# Extrait et expand les avis en ligne individuelles
for index, game_row in full_games_df.iterrows():
    for review_type in ['english_reviews', 'french_reviews']:
        temp_df = pd.json_normalize(game_row[review_type])
        temp_df['language'] = review_type.split('_')[0]  # English or French
        for col in full_games_df.columns.difference(['english_reviews', 'french_reviews']):
            # Répéter la valeur pour chaque avis
            temp_df[col] = [game_row[col]] * len(temp_df)
        reviews_df = pd.concat([reviews_df, temp_df], ignore_index=True)

# Exporter les données complètes en CSV
reviews_df.to_csv('complete_reviews_data.csv', index=False)
