import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from collections import defaultdict


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
                    'artist_id': item['artists'][0]['id'],  # Ajouter l'ID de l'artiste
                    'preview_url': item['preview_url'],
                    'genres': []
                }
                tracks.append(track_info)
        offset += limit
    return tracks[:total]  # Retourner seulement le nombre total demandé

# Obtenir les genres disponibles
genres = get_available_genres()

# Enregistrer les genres dans un fichier texte
with open('CLUSTERING/genres_sl.txt', 'w', encoding='utf-8') as f:
    for genre in genres:
        f.write(f"{genre}\n")

def load_genres_from_file(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

# Charger les genres modifiés
genres = load_genres_from_file('CLUSTERING/genres_sl.txt')

# Sélectionner les genres désirés (par exemple, les premiers 5 genres disponibles)
selected_genres = genres[:]

# Obtenir les musiques pour chaque genre sélectionné
all_tracks = []
artist_id_to_name = {}
for genre in selected_genres:
    tracks = get_tracks_by_genre(genre)
    all_tracks.extend(tracks)
    for track in tracks:
        artist_id_to_name[track['artist_id']] = track['artist']

# Enregistrer tous les titres dans un fichier commun
with open("CLUSTERING/all_tracks.txt", 'w', encoding='utf-8') as f:
    for track in all_tracks:
        f.write(f"{track['name']}\n")
        f.write(f"{track['artist']}\n")
        f.write(f"{track['preview_url']}\n")
        f.write("\n")

# Enregistrer tous les artistes uniques dans un fichier
with open("CLUSTERING/artists.txt", 'w', encoding='utf-8') as f:
    for artist_id, artist_name in artist_id_to_name.items():
        f.write(f"{artist_name} ({artist_id})\n")

# Obtenir les genres musicaux pour chaque artiste
def get_genres_for_artists(artist_ids):
    genres = {}
    for i in range(0, len(artist_ids), 50):  # Limite de 50 artistes par requête
        batch_ids = artist_ids[i:i+50]
        results = sp.artists(batch_ids)
        for artist in results['artists']:
            if artist:
                genres[artist['id']] = artist.get('genres', [])
    return genres

# Regrouper les requêtes API pour obtenir les genres des artistes

artist_ids = list(artist_id_to_name.keys())
artist_genres = get_genres_for_artists(artist_ids)

# Associer les genres musicaux à chaque titre
genre_frequency = defaultdict(int)
for track in all_tracks:
    track['genres'] = artist_genres.get(track['artist_id'], [])
    for genre in track['genres']:
        genre_frequency[genre] += 1

# Enregistrer les titres avec les genres associés dans un fichier
with open("all_tracks_with_genres.txt", 'w', encoding='utf-8') as f:
    for track in all_tracks:
        f.write(f"{track['name']}\n")
        f.write(f"{track['artist']}\n")
        f.write(f"{track['preview_url']}\n")
        f.write(f"Genres: {', '.join(track['genres'])}\n")
        f.write("\n")

# Enregistrer les genres et leur fréquence dans un fichier
with open("genre_frequencies.txt", 'w', encoding='utf-8') as f:
    for genre, count in sorted(genre_frequency.items(), key=lambda item: item[1], reverse=True):
        f.write(f"{genre}: {count}\n")
