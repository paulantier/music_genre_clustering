import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from torchvision.models import resnet18


batch_size = 1

data_folder = 'MELs_train'

checkpoint_path = 'checkpoints_mel_sp/checkpoint_epoch_22.pth'

layer = "LAST" #LAST, INTER ou ALL

test_name = "trainData_lastFeatures"

extract = True

plot = "2D" #False, 2D ou 3D

decomposition = "PCA" #PCA ou TSNE


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

dataset = datasets.ImageFolder(data_folder, transform=transform)


if extract:
    print('extraction de caractéristiques')
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

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
    model = resnet18().to(device)
    model.eval()

    load_checkpoint(checkpoint_path, model)

    features1 = ConvFeatureExtractor(model, layer=1).to(device)
    features2 = ConvFeatureExtractor(model, layer=2).to(device)

    features1.eval()
    features2.eval()

    liste_features = []
    labels_list = []

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)

            if layer == "ALL":
                features = torch.cat( (features1(images), features2(images)) , 1)
            elif layer == "LAST":
                features = features1(images)
            elif layer == "INTER":
                features = features2(images)
            liste_features.append(features.cpu().numpy())
            labels_list.append(labels.cpu().numpy())

    # Convert lists to arrays
    features_array = np.concatenate(liste_features, axis=0)
    labels_array = np.concatenate(labels_list, axis=0)

    np.save(f'features_{test_name}.npy', features_array)
    np.save(f'labels_{test_name}.npy', labels_array)




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
        class_names = dataset.classes

        plt.figure(figsize=(16, 10))
        scatter = plt.scatter(results[:, 0], results[:, 1], c=labels_array, cmap='jet', alpha=0.6, edgecolors='w', linewidth=0.5)

        # Annotate clusters with class names
        for i, class_name in enumerate(class_names):
            # Find the mean position of points in each cluster
            class_indices = np.where(labels_array == i)
            mean_x = np.mean(results[class_indices, 0])
            mean_y = np.mean(results[class_indices, 1])
            plt.text(mean_x, mean_y, class_name, fontsize=12, bbox=dict(facecolor='white', alpha=0.6))

        plt.title('Music Genre Map')
        plt.xlabel('')
        plt.ylabel('')
        plt.show()

    elif plot=="3D":

        features_array = np.load(f'features_{test_name}.npy')

        labels_array = np.load(f'labels_{test_name}.npy')

        if decomposition=="TSNE":
            tsne = TSNE(n_components=3)
            results = tsne.fit_transform(features_array)
        elif decomposition=="PCA":
            pca = PCA(n_components=3)
            results = pca.fit_transform(features_array)

        # Define class names
        class_names = dataset.classes

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