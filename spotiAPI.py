
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# Authentification
auth_manager = SpotifyClientCredentials(client_id='', client_secret='')
sp = spotipy.Spotify(auth_manager=auth_manager)

# Récupérer les genres disponibles
def get_available_genres():
    return sp.recommendation_genre_seeds()['genres']

# Fonction pour obtenir des musiques par genre
def get_tracks_by_genre(genre, limit=50, total=400):
    offset = 0
    tracks = []
    while len(tracks) < total:
        results = sp.search(q=f'genre:{genre}', type='track', limit=limit, offset=offset)
        items = results['tracks']['items']
        if not items:
            break
        for item in items:
            if item is not None:
                track_info = {
                    'name': item['name'],
                    'artist': item['artists'][0]['name'],
                    'preview_url': item['preview_url']
                }

                with open(f"genres_sp/{genre}.txt", 'a', encoding='utf-8') as f:
                    f.write(f"{track_info['name']}\n")
                    f.write(f"{track_info['artist']}\n")
                    f.write(f"{track_info['preview_url']}\n")
                    f.write("\n")
                tracks.append(track_info)

        offset += limit
    return tracks[:total]  # Retourner seulement le nombre total demandé

# Obtenir les genres disponibles
genres = get_available_genres()

"""
# Enregistrer les genres dans un fichier texte
with open('genres.txt', 'w') as f:
    for genre in genres:
        f.write(f"{genre}\n")
"""

def load_genres_from_file(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f if line.strip()]

# Charger les genres modifiés
genres = load_genres_from_file('genres.txt')

# Sélectionner les genres désirés (par exemple, les premiers 5 genres disponibles)
selected_genres = genres[:]

# Obtenir les musiques pour chaque genre sélectionné
genre_tracks = {}
for genre in selected_genres:
    genre_tracks[genre] = get_tracks_by_genre(genre)
    with open(f"{genre}.txt", 'w',encoding='utf-8') as f:
        for track in genre_tracks[genre]:
            f.write(f"{track['name']}\n")
            f.write(f"{track['artist']}\n")
            f.write(f"{track['preview_url']}\n")
            f.write("\n")

