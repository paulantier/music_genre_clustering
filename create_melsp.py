import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Define the base paths
input_folder = 'downloaded_previews'
output_folder = 'MELs'

def process():
    
    # Process each file in the genre folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.wav'):
            file_path = os.path.join(input_folder, filename)
            
            # Load audio file
            y, sr = librosa.load(file_path, sr=None)
            
            # Compute Mel-spectrogram
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
            mel_spec_db = librosa.amplitude_to_db(mel_spec, ref=np.max)
            
            # Create the figure and remove axes
            fig, ax = plt.subplots()
            fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
            
            # Plot the Mel-spectrogram
            librosa.display.specshow(mel_spec_db, sr=sr, x_axis=None, ax=ax)
            
            # Remove axes
            ax.axis('off')
            
            # Save the plot as an image file
            output_file_path = os.path.join(output_folder, f'{os.path.splitext(filename)[0]}.png')
            fig.savefig(output_file_path, bbox_inches='tight', pad_inches=0)
            plt.close(fig)
            
            # Remove the .wav file after processing
            os.remove(file_path)

process()
