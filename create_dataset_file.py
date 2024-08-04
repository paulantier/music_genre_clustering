import os
import csv
from tqdm import tqdm

# Chemins vers les fichiers
image_folder = "MELs"
output_file = "dataset.csv"
all_classes_file = "genre_frequencies.txt"



# Define the mapping for genres
genre_mapping = {
    "classical": ["early music", "baroque", "classical era", "late romantic era", "romantic era", "early romantic era", "modern classical", "post-romantic era"],
    "mpb": ["mpb"],
    "reggaeton" : ["reggaeton", "reggaeton colombiano"],
    "hip hop": ["hip hop", "trap", "trap latino", "underground hip hop", "hardcore hip hop", "emo rap", "alternative hip hop", "latin hip hop", "french hip hop", "nigerian hip hop", "kids hip hop"],
    "rock": ["rock", "album rock", "soft rock", "classic rock", "rock-and-roll", "alternative rock", "indie rock", "modern rock", "psychedelic rock", "hard rock", "garage rock"],
    "disco": ["disco", "disco house", "nu disco"],
    "samba": ["samba", "samba reggae", "samba-jazz", "samba-enredo"],
    "bossa nova": ["bossa nova"],
    "bachata": ["bachata"],
    "salsa": ["salsa", "salsa puertorriquena"],
    "techno": ["techno", "minimal techno", "detroit techno", "acid techno"],
    "pop": ["pop", "pop dance",  "j-pop", "pop punk", "pop rock", "pop edm", "pop nacional", "british pop", "bubblegum pop", "latin pop", "pop rap", "pop emo", "pop reggaeton", "power pop", "pop rock brasileiro", "wonky", "indie pop", "synthpop", "french pop"],
    "electronic": ["electronic", "electronica", "idm", "edm", "downtempo", "chillwave", "electropop", "trance", "drum and bass", "hardcore techno"],
    "house": ["house", "deep house", "electro house", "progressive house", "tech house", "acid house"],
    "death metal": ["death metal", "brutal death metal", "melodic death metal"],
    "black metal": ["black metal", "symphonic black metal", "black thrash"],
    "afrobeat": ["afrobeat", "afrobeats"],
    "ambient": ["ambient", "ambient idm"],
    "country": ["country", "country rock", "honky tonk", "bluegrass", "traditional country", "alternative country", "country blues", "country pop", "outlaw country", "cowboy", "country gospel", "americana", "southern rock", "new country"],
    "blues": ["blues", "blues rock", "acoustic blues", "electric blues"],
    "folk": ["folk", "indie folk", "folk rock", "folk punk"],
    "funk": ["funk", "funk rock", "funk metal"],
    "grunge": ["grunge"],
    "punk": ["punk", "punk rock", "post-punk", "hardcore punk", "punk blues"],
    "ska": ["ska", "ska punk", "ska revival"],
    "reggae": ["reggae", "roots reggae", "reggae fusion", "dub reggae", "dancehall"],
    "jazz": ["jazz", "jazz fusion", "latin jazz", "vocal jazz", "bebop"],
    "dubstep": ["dubstep", "riddim dubstep", "deep dubstep", "brostep"],
    "tango": ["tango", "nuevo tango"],
    "breakbeat": ["breakbeat", "big beat", "psybreaks"],
    "heavy metal": ["heavy metal", "thrash metal", "glam metal", "speed metal"],
    "power metal": ["power metal", "symphonic metal"],
    "metalcore": ["metalcore", "melodic metalcore", "deathcore"],
}

new_genres = list(genre_mapping.keys())
print(new_genres)

def map_genre(genre):
    for key, values in genre_mapping.items():
        if genre in values:
            return key
    return "Other"




# Lire les classes possibles depuis le fichier all_classes.txt
all_classes = []
with open(all_classes_file, 'r') as f:
    for line in f:
        class_name, _ = line.split(':')
        all_classes.append(class_name.strip())

with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
    csvwriter = csv.writer(csvfile)
    # Lire chaque fichier image et son fichier texte associé
    for image_file in tqdm(os.listdir(image_folder)):
        if image_file.endswith(".png"):
            image_name = os.path.splitext(image_file)[0].replace('\ufeff\ufeff','')
            image_path = os.path.join(image_folder, image_file)
            label_path = os.path.join(image_folder, f"{image_name}.txt")
            
            # Lire les classes présentes dans le fichier imageX.txt
            with open(label_path, 'r') as label_file:
                image_classes = label_file.read().strip().split('\n')

            soft_label = [0] * len(new_genres)
            
            list_genres=[]
            new_list_genres = []
            for c in all_classes:
                list_genres.extend([c for image_class in image_classes if c in image_class])
            for genre in list_genres:
                new_genre = map_genre(genre)
                if new_genre!="Other":
                    new_list_genres.append(new_genre)
            for genre in new_list_genres:
                index = new_genres.index(genre)
                soft_label[index] += 1
            
            total = sum(soft_label)
            if total > 0:
                soft_label = [x / total for x in soft_label]
                csvwriter.writerow(soft_label + [image_name])

        


