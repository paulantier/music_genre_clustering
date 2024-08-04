import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from tqdm import tqdm
from torchvision.models import resnet18, ResNet18_Weights
import csv
from torch.utils.data import Dataset
from PIL import Image
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
from collections import Counter
from matplotlib.patches import Patch
import gc

batch_size = 1

data_part = 'TRAIN' #ALL, TRAIN ou TEST

checkpoint_path = 'checkpoints_mel_sp/checkpoint_epoch_71.pth'

layer = "LAST" #LAST, INTER ou ALL

test_name = f"{data_part}Data_{layer}Features"

extract = True

plot = "2D" #False, 2D ou 3D

decomposition = "PCA" #PCA ou TSNE

soft_label_to_hard_label = "max_index" # "first_nonzero" or "max_index"

def first_nonzero(labels_array):
    indices = np.zeros(labels_array.shape[0], dtype=int)
    for i in range(labels_array.shape[0]):
        nonzero_indices = np.nonzero(labels_array[i])[0]
        if len(nonzero_indices) > 0:
            indices[i] = nonzero_indices[0]
    return indices

def max_index(labels_array):
    indices = np.zeros(labels_array.shape[0], dtype=int)
    for i in range(labels_array.shape[0]):
        max_indices = np.argwhere(labels_array[i] == np.amax(labels_array[i])).flatten()
        indices[i] = np.random.choice(max_indices)  # Choose randomly among max indices if there's a tie
    return indices



transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Path to the CSV file
csv_file = "dataset.csv"
len_labels=33

# Lists to store image names and soft labels
image_names = []
soft_labels = []

# Read the CSV file
with open(csv_file, 'r', encoding='utf-8') as csvfile:
    csvreader = csv.reader(csvfile)
    
    for row in csvreader:
        soft_labels.append([float(x) for x in row[:len_labels]])  # Soft labels are the res
        image_names.append(row[len_labels:])  # The image name is the last element

class ImageDataset(Dataset):
    def __init__(self, image_folder, image_names, soft_labels, transform=None):
        self.image_folder = image_folder
        self.image_names = image_names
        self.soft_labels = soft_labels
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        if len(img_name)==1:
            img_name = img_name[0]
        img_path = os.path.join(self.image_folder, img_name + '.png')
        
        # Open image
        image = Image.open(img_path).convert("RGB")
        
        # Apply transformations if any
        if self.transform:
            image = self.transform(image)
        
        # Get corresponding soft labels
        labels = torch.tensor(self.soft_labels[idx], dtype=torch.float32)
        
        return image, labels


def load_checkpoint(checkpoint_path, model):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    epoch = checkpoint.get('epoch', None)
    
    return epoch

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def soft_confusion_matrix(y_true, y_pred):
    """
    Compute the soft confusion matrix.
    
    :param y_true: Array of true soft labels, shape (n_samples, n_classes)
    :param y_pred: Array of predicted soft labels, shape (n_samples, n_classes)
    :return: Soft confusion matrix, shape (n_classes, n_classes)
    """
    n_classes = y_true.shape[1]
    matrix = np.zeros((n_classes, n_classes))
    
    for yt, yp in zip(y_true, y_pred):
        matrix += np.outer(yt, yp)
    
    return matrix

all_classes = ['classical', 'mpb', 'reggaeton', 'hip hop', 'rock', 'disco', 'samba', 'bossa nova', 'bachata', 'salsa', 'techno', 'pop', 'electronic', 'house', 'death metal', 'black metal', 'afrobeat', 'ambient', 'country', 'blues', 'folk', 'funk', 'grunge', 'punk', 'ska', 'reggae', 'jazz', 'dubstep', 'tango', 'breakbeat', 'heavy metal', 'power metal', 'metalcore']
if extract:
    print('extraction des caractéristiques')
    image_folder = "MELs"
    batch_size=16

    if data_part == "TRAIN" or data_part == "TEST":
        train_image_names, test_image_names, train_soft_labels, test_soft_labels = train_test_split(
            image_names, soft_labels, test_size=0.2, random_state=1
        )

        batch_size=16

        if data_part == "TRAIN":
            train_dataset = ImageDataset(image_folder, train_image_names, train_soft_labels, transform=transform)
            dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        else:
            test_dataset = ImageDataset(image_folder, test_image_names, test_soft_labels, transform=transform)
            dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    elif data_part == "ALL":
    # Create the train and test datasets
        dataset = ImageDataset(image_folder, image_names, soft_labels, transform=transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)


    def load_checkpoint(checkpoint_path, model):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])

    class ConvFeatureExtractor(nn.Module):
        def __init__(self, original_model, layer):
            super(ConvFeatureExtractor, self).__init__()
            self.features = nn.Sequential(*list(original_model.children())[:-layer])

        def forward(self, x):
            x = self.features(x)
            x = torch.flatten(x, 1)
            return x


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    # Instantiate the model
    class MyResModel(torch.nn.Module):
        def __init__(self):
            super(MyResModel, self).__init__()
            self.classifier = nn.Sequential(
                nn.Linear(512,128),
                nn.ReLU(),
                nn.Dropout(p=0.1),
                nn.Linear(128,len_labels),
            )

        def forward(self, x):
            return self.classifier(x)


    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = MyResModel()
    model = model.to(device)

    load_checkpoint(checkpoint_path, model)

    features1 = ConvFeatureExtractor(model, layer=1).to(device)
    features2 = ConvFeatureExtractor(model, layer=2).to(device)

    features1.eval()
    features2.eval()

    features_temp_files = []
    labels_temp_files = []

    def to_numpy(tensor, dtype=np.float16):
        return tensor.cpu().numpy().astype(dtype)

    # Utiliser torch.float16 pour réduire l'utilisation de la mémoire
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(dataloader)):
            images, labels = images.to(device), labels.to(device)
            
            if layer == "ALL":
                features = torch.cat((features1(images), features2(images)), 1)
            elif layer == "LAST":
                features = features1(images)
            elif layer == "INTER":
                features = features2(images)
            
            # Convertir en numpy et sous-échantillonner avec des types de données moins lourds
            features_np = to_numpy(features, dtype=np.float16)
            labels_np = to_numpy(labels, dtype=np.int16)

            # Sauvegarder les résultats dans des fichiers temporaires
            features_file = f'features_temp_{batch_idx}.npy'
            labels_file = f'labels_temp_{batch_idx}.npy'
            
            np.save(features_file, features_np)
            np.save(labels_file, labels_np)
            
            features_temp_files.append(features_file)
            labels_temp_files.append(labels_file)

            # Libérer la mémoire GPU et CPU
            del images, labels, features, features_np, labels_np
            torch.cuda.empty_cache()
            gc.collect()

    # Charger et concaténer les fichiers temporaires
    features_list = [np.load(file) for file in features_temp_files]
    labels_list = [np.load(file) for file in labels_temp_files]

    features_array = np.concatenate(features_list, axis=0)
    labels_array = np.concatenate(labels_list, axis=0)

    # Sauvegarder les tableaux finaux
    np.save(f'features_{test_name}.npy', features_array)
    np.save(f'labels_{test_name}.npy', labels_array)

    # Supprimer les fichiers temporaires
    for file in features_temp_files + labels_temp_files:
        os.remove(file)

if plot:
    print('projection des données')
    if plot=="2D":

        features_array = np.load(f'features_{test_name}.npy')

        labels_array = np.load(f'labels_{test_name}.npy')

        if decomposition=="TSNE":
            tsne = TSNE(n_components=2)
            results = tsne.fit_transform(features_array)
        elif decomposition=="PCA":
            pca = PCA(n_components=2)
            results = pca.fit_transform(features_array)

        # Define class names
        class_names = all_classes
        if soft_label_to_hard_label == "first_nonzero":
            labels_array = first_nonzero(labels_array)
        elif soft_label_to_hard_label == "max_index":
            labels_array = max_index(labels_array)

        plt.figure(figsize=(16, 10))
        scatter = plt.scatter(results[:, 0], results[:, 1], c=labels_array, cmap='jet', alpha=0.6, edgecolors='w', linewidth=0.5)

        # Annotate clusters with class names
        for i, class_name in enumerate(class_names):
            # Find the mean position of points in each cluster
            class_indices = np.where(labels_array == i)
            mean_x = np.mean(results[class_indices, 0])
            mean_y = np.mean(results[class_indices, 1])
            plt.text(mean_x, mean_y, class_name, fontsize=12, bbox=dict(facecolor='white', alpha=0.6))

        # Create legend
        handles = [Patch(color=plt.cm.jet(i / len(class_names)), label=class_name) for i, class_name in enumerate(class_names)]
        plt.legend(handles=handles, title="Classes", loc="best")

        plt.title('Music Genre Map')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.show()

    elif plot=="3D":

        features_array = np.load(f'features_{test_name}.npy')

        labels_array = np.load(f'labels_{test_name}.npy')

        if soft_label_to_hard_label == "first_nonzero":
            labels_array = first_nonzero(labels_array)
        elif soft_label_to_hard_label == "max_index":
            labels_array = max_index(labels_array)

        if decomposition=="TSNE":
            tsne = TSNE(n_components=3)
            results = tsne.fit_transform(features_array)
        elif decomposition=="PCA":
            pca = PCA(n_components=3)
            results = pca.fit_transform(features_array)

        # Define class names
        class_names = all_classes

        fig = plt.figure(figsize=(16, 10))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(results[:, 0], results[:, 1], results[:, 2], c=labels_array, cmap='jet', alpha=0.6, edgecolors='w', linewidth=0.5)

        # Annotate clusters with class names
        for i, class_name in enumerate(class_names):
            # Find the mean position of points in each cluster
            class_indices = np.where(labels_array == i)
            mean_x = np.mean(results[class_indices, 0])
            mean_y = np.mean(results[class_indices, 1])
            mean_z = np.mean(results[class_indices, 2])
            ax.text(mean_x, mean_y, mean_z, class_name, fontsize=12, bbox=dict(facecolor='white', alpha=0.6))

        plt.title('Music genre Map')
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_zlabel('')
        plt.show()
        plt.show()