genre_mapping = {
    "classical": ["classical", "early music", "baroque", "classical era", "late romantic era", "romantic era", "early romantic era", "modern classical", "post-romantic era"],
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


# Utilisation d'un dictionnaire pour stocker les genres déjà rencontrés
genre_count = {}
duplicates = []

# Parcours de chaque genre et de ses sous-genres
for genre, subgenres in genre_mapping.items():
    for subgenre in subgenres:
        if subgenre in genre_count:
            if subgenre not in duplicates:
                duplicates.append(subgenre)
        else:
            genre_count[subgenre] = 1

# Affichage des doublons détectés
if duplicates:
    print("Les doublons dans le mapping sont :")
    for duplicate in duplicates:
        print(duplicate)
else:
    print("Aucun doublon trouvé dans le mapping.")
