import os
import requests
from concurrent.futures import ThreadPoolExecutor

# Fonction pour télécharger une preview et l'enregistrer en tant que fichier WAV
def download_preview_and_save(track_info, genre):
    name = track_info[0][:-1].split(" - ")[0].replace("/","_").replace('"',"").replace(":","").replace(".","").replace("\\","_").replace("?","").replace("|","").replace("*","")
    artist = track_info[1][:-1].split(" - ")[0].replace("/","_").replace('"',"").replace(":","").replace(".","").replace("\\","_").replace("?","").replace("|","").replace("*","")
    preview_url = track_info[2][:-1]
    if preview_url and preview_url != 'None':
        response = requests.get(preview_url)
        if response.status_code == 200:
            # Créer un sous-dossier s'il n'existe pas déjà
            subfolder = os.path.join('genres_sp', genre)
            if not os.path.exists(subfolder):
                os.makedirs(subfolder)
            # Enregistrer le fichier WAV dans le sous-dossier
            filename = os.path.join(subfolder, f"{artist} - {name}.wav")
            with open(filename, 'wb') as f:
                f.write(response.content)
            print(f"Preview téléchargée et enregistrée : {filename}")
        else:
            print(f"Échec du téléchargement de la preview pour {name} - {artist}")

# Lire chaque bloc d'informations dans la liste "lines" et soumettre les tâches au pool de threads
def process_lines(lines, genre, executor):
    track_info = []
    for i, line in enumerate(lines):
        if line.strip() == '':
            if track_info:
                executor.submit(download_preview_and_save, track_info, genre)
            track_info = []
        else:
            track_info.append(line)
    # Soumettre la dernière tâche s'il y a des données restantes
    if track_info:
        executor.submit(download_preview_and_save, track_info, genre)

# Parcourir tous les fichiers texte dans le dossier "genres_sp"
def main():
    with ThreadPoolExecutor(max_workers=10) as executor:  # Adjust the number of workers as needed
        for filename in os.listdir('genres_sp'):
            if filename.endswith('.txt'):
                genre = os.path.splitext(filename)[0]
                with open(os.path.join('genres_sp', filename), 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    # Appeler la fonction pour lire et traiter chaque bloc d'informations
                    process_lines(lines, genre, executor)

if __name__ == "__main__":
    main()
