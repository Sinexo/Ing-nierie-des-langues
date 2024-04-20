import requests
import json
from bs4 import BeautifulSoup

def get_reviews(appid, n=40, language='english'):
    reviews = []
    cursor = '*'
    params = {
        'json': 1,
        'filter': 'all',
        'language': language,
        'day_range': 9223372036854775807,
        'review_type': 'negative',  # Assurez-vous que c'est bien le type de critique que vous voulez
        'purchase_type': 'all'
    }

    while len(reviews) < n:
        params['cursor'] = cursor
        response = requests.get(url=f'https://store.steampowered.com/appreviews/{appid}', params=params, headers={'User-Agent': 'Mozilla/5.0'})
        if response.ok:
            response_json = response.json()
            if 'reviews' in response_json:
                reviews.extend(response_json['reviews'])
                cursor = response_json['cursor']
                if len(response_json['reviews']) < 40:  # Suppose chaque lot retourne 20 avis ou moins
                    break
        else:
            print(f"Failed to fetch reviews for appid {appid}")
            break

    return reviews[:n]


def get_n_appids(n=200, filter_by='topsellers'):
    appids = []
    page = 0

    while len(appids) < n:
        page += 1
        response = requests.get(url=f'https://store.steampowered.com/search/?category1=998&filter={filter_by}&page={page}', headers={'User-Agent': 'Mozilla/5.0'})
        if response.ok:
            soup = BeautifulSoup(response.text, 'html.parser')
            for row in soup.find_all(class_='search_result_row'):
                appids.append(row['data-ds-appid'])
                if len(appids) >= n:
                    break
        else:
            print(f"Failed to fetch page {page} of app ids")
            break

    return appids[:n]

# ID des jes
appids = get_n_appids(2000)

# Collecte des détails des jeux et des avis
games_data = []
# Collecte des détails des jeux et des avis uniquement en anglais
games_data = []
for appid in appids:
    url = f"https://store.steampowered.com/api/appdetails?appids={appid}"
    details_response = requests.get(url)
    if details_response.ok:
        details_json = details_response.json()
        if details_json.get(str(appid), {}).get('success', False):
            data = details_json[str(appid)]['data']
            english_reviews = get_reviews(appid, 50, 'english')  # Récupération des avis négatifs anglais
            game_info = {
                'name': data.get('name', 'N/A'),
                # 'categories': data.get('categories', []), supprimer car peu pertinent dans la génération de commentaire
                'genres': data.get('genres', []),
                'description': data.get('short_description', 'No description available.'),
                # 'price': data.get('price_overview', {}).get('final_formatted', 'Free or not available'), supprimer car peu pertinent dans la génération de commentaire
                'developers': data.get('developers', ['Unknown']),
                'publishers': data.get('publishers', ['Unknown']),
                'english_reviews': english_reviews,
                # 'platforms': list(data.get('platforms', {}).keys()) supprimer car peu pertinent dans la génération de commentaire
            }
            games_data.append(game_info)
        else:
            print(f"No valid data for appid {appid}")
    else:
        print(f"Failed to retrieve game details for appid {appid}")



with open('steam_data.json', 'w', encoding='utf-8') as f:
    json.dump(games_data, f, ensure_ascii=False, indent=4)

print("Les données ont été enregistrées dans 'steam_data.json'.")
